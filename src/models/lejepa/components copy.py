import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import timm
from transformers import AutoModel
from lightning.pytorch.callbacks import Callback
from torchmetrics.classification import MultilabelAUROC
from torchvision import transforms
from torchvision.transforms import functional as TF
from typing import Literal

# ==========================================
# MODEL COMPONENTS
# ==========================================

class SIGReg(nn.Module):
    """
    The Sketched Isotropic Gaussian Regularizer (SIGReg).
    Matches the GitHub implementation math but updated for DDP compliance and robustness.
    """
    def __init__(self, knots=17, d_max=3, proj_dim=128, sketch_dim=256):
        super().__init__()
        # Integration points and weights (Algorithm 1 / GitHub)
        t = torch.linspace(0, d_max, knots)
        dt = d_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        
        # Default sketch matrix (used if seed is None)
        A = torch.randn(proj_dim, sketch_dim)
        A = A / A.norm(p=2, dim=0)
        self.register_buffer("A_default", A)

    def forward(self, proj, seed: int = None):
        # proj expected shape: [Batch, Dim] or [Total_Views, Dim]
        
        # 1. Sketching directions A [D, M]
        if seed is not None:
            # Synchronized variety across DDP ranks
            g = torch.Generator(device=proj.device).manual_seed(seed)
            A = torch.randn(proj.size(-1), self.A_default.size(1), generator=g, 
                            device=proj.device, dtype=proj.dtype)
            A = A / A.norm(p=2, dim=0)
        else:
            A = self.A_default.to(proj.dtype)

        # 2. Project onto sketching directions: [N, M, T] (Batch, Directions, Knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        
        # 3. Empirical Characteristic Function (ECF) - DDP Compliant
        # We average across the 'N' dimension (first dimension in a 2D proj)
        cos_sum = x_t.cos().sum(dim=0) # [M, T]
        sin_sum = x_t.sin().sum(dim=0) # [M, T]
        local_n = proj.size(0)
        
        if dist.is_initialized():
            dist.all_reduce(cos_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(sin_sum, op=dist.ReduceOp.SUM)
            n_tensor = torch.tensor([float(local_n)], device=proj.device, dtype=proj.dtype)
            dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)
            global_n = n_tensor.item()
        else:
            global_n = float(local_n)

        # 4. Eeps-Pulley test statistic
        cos_mean = cos_sum / global_n
        sin_mean = sin_sum / global_n
        
        # Mean distance between empirical and theoretical characteristic functions
        err = (cos_mean - self.phi).square() + sin_mean.square()
        
        # Summation over integration points and scaling by total samples n
        # statistic shape: [M]
        statistic = (err @ self.weights) * global_n
        
        return statistic.mean()

class MultiModalEncoder(nn.Module):
    def __init__(
        self,
        img_model_name: str = "vit_small_patch16_224",
        txt_model_name: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        proj_dim: int = 128,
    ):
        super().__init__()
        self.img_backbone = timm.create_model(img_model_name, pretrained=False, num_classes=0, dynamic_img_size=True)
        self.backbone_dim = self.img_backbone.num_features
        self.img_proj = nn.Sequential(
            nn.Linear(self.backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )
        
        if txt_model_name is not None:
            self.txt_backbone = AutoModel.from_pretrained(txt_model_name, trust_remote_code=True)
            # Freeze the pretrained BERT backbone — it serves as a stable semantic
            # anchor.  Only the txt_proj projector (below) is trained, which is
            # sufficient: gradients from the LeJEPA invariance loss update txt_proj
            # to align the projection space without drifting the pretrained LM.
            for param in self.txt_backbone.parameters():
                param.requires_grad_(False)
            self.txt_backbone.eval()  # freeze BN/dropout statistics too
            self.txt_proj = nn.Sequential(
                nn.Linear(768, 2048),
                nn.BatchNorm1d(2048),
                nn.GELU(),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.GELU(),
                nn.Linear(2048, proj_dim)
            )
        else:
            self.txt_backbone = None
            self.txt_proj = None

    def forward(self, x, mode: Literal["img", "txt"]="img"):
        if mode == "img":
            return self.forward_img(x)
        elif mode == "txt":
            return self.forward_txt(x)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward_img(self, x):
        feats = self.img_backbone(x)
        return feats, self.img_proj(feats)

    def forward_txt(self, input_ids, attention_mask, **kwargs):
        assert self.txt_backbone is not None, "Text backbone not initialized"
        # Backbone is frozen — no_grad saves activation memory for the BERT pass.
        with torch.no_grad():
            outputs = self.txt_backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls_feat = outputs.last_hidden_state[:, 0, :].detach()
        return cls_feat, self.txt_proj(cls_feat)

# ==========================================
# TRANSFORM COMPONENTS
# ==========================================

class CXRAnatBiasedCrop(nn.Module):
    """Anatomically-biased RandomResizedCrop for Chest X-rays."""
    def __init__(
        self,
        output_size:   int   = 224,
        scale:         tuple = (0.40, 1.00),
        ratio:         tuple = (0.75, 1.33),
        cx_mu:         float = 0.50,
        cy_mu:         float = 0.42,
        sigma_x:       float = 0.15,
        sigma_y:       float = 0.12,
        blur_p:        float = 0.0,
        blur_kernel:   int   = 7,
        blur_sigma:    float = 1.4,
        brightness:    float = 0.0,
        contrast:      float = 0.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.scale       = scale
        self.ratio       = ratio
        self.cx_mu       = cx_mu
        self.cy_mu       = cy_mu
        self.sigma_x     = sigma_x
        self.sigma_y     = sigma_y
        self.blur_p      = blur_p

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

        if blur_p > 0:
            coords = torch.arange(blur_kernel).float() - blur_kernel // 2
            g = torch.exp(-(coords ** 2) / (2 * blur_sigma ** 2))
            g = g / g.sum()
            kernel_2d = g.outer(g).view(1, 1, blur_kernel, blur_kernel)
            self.register_buffer("blur_k", kernel_2d)
            self.blur_pad = blur_kernel // 2
        else:
            self.blur_k   = None
            self.blur_pad = 0

        self.light_aug = transforms.ColorJitter(brightness=brightness, contrast=contrast)

    def _sample_crop_params(self, w: int, h: int):
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            rel_area = torch.empty(1).uniform_(*self.scale).item()
            aspect   = math.exp(torch.empty(1).uniform_(*log_ratio).item())
            crop_w   = int(round(math.sqrt(w * h * rel_area * aspect)))
            crop_h   = int(round(math.sqrt(w * h * rel_area / aspect)))
            if crop_w <= w and crop_h <= h:
                cx   = float(torch.empty(1).normal_(self.cx_mu, self.sigma_x).clamp(0, 1))
                cy   = float(torch.empty(1).normal_(self.cy_mu, self.sigma_y).clamp(0, 1))
                left = max(0, min(int(cx * w - crop_w / 2), w - crop_w))
                top  = max(0, min(int(cy * h - crop_h / 2), h - crop_h))
                return top, left, crop_h, crop_w
        crop = min(w, h)
        return (h - crop) // 2, (w - crop) // 2, crop, crop

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        top, left, crop_h, crop_w = self._sample_crop_params(w, h)
        img = TF.resized_crop(
            img, top, left, crop_h, crop_w,
            size=[self.output_size, self.output_size],
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        img = self.light_aug(img)
        img = (img - self.mean) / self.std
        if self.blur_k is not None and torch.rand(1).item() < self.blur_p:
            C      = img.shape[0]
            kernel = self.blur_k.expand(C, 1, -1, -1)
            img    = F.conv2d(img.unsqueeze(0), kernel, padding=self.blur_pad, groups=C).squeeze(0)
        return img

class CXRMultiCropTransform(nn.Module):
    def __init__(
        self,
        global_size:  int   = 224,
        local_size:   int   = 96,
        scale_global: tuple = (0.40, 1.00),
        scale_local:  tuple = (0.05, 0.40),
        n_global:     int   = 2,
        n_local:      int   = 8,
        brightness:   float = 0.15,
        contrast:     float = 0.15,
    ):
        super().__init__()
        self.n_global = n_global
        self.n_local  = n_local
        self.global_crop = CXRAnatBiasedCrop(
            output_size=global_size, scale=scale_global,
            sigma_x=0.12, sigma_y=0.10, blur_p=0.0,
            brightness=brightness, contrast=contrast,
        )
        self.local_crop  = CXRAnatBiasedCrop(
            output_size=local_size,  scale=scale_local,
            sigma_x=0.22, sigma_y=0.18, blur_p=0.5,
            brightness=brightness, contrast=contrast,
        )

    def forward(self, img: torch.Tensor) -> dict:
        # Precision Island: Force float32 for geometric/color transforms to avoid 
        # bf16 kernel instability. Use modern torch.amp.autocast.
        autocast_kwargs = {"device_type": img.device.type, "enabled": False}
        with torch.amp.autocast(**autocast_kwargs):
            img_f = img.float()
            global_views = torch.stack([self.global_crop(img_f) for _ in range(self.n_global)])
            local_views  = torch.stack([self.local_crop(img_f)  for _ in range(self.n_local)])
        return {"global_views": global_views, "local_views": local_views}

# ==========================================
# EVALUATION UTILITIES & CALLBACKS
# ==========================================

def _padded_all_gather(local_tensor: torch.Tensor, device: torch.device, world_size: int) -> torch.Tensor:
    if not dist.is_initialized() or world_size <= 1: return local_tensor
    local_n, D = local_tensor.shape[0], local_tensor.shape[1]
    local_size = torch.tensor([local_n], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]
    max_n = max(all_sizes)
    if max_n == 0: return torch.zeros(0, D, device=device)
    pad_n = max_n - local_n
    if pad_n > 0:
        local_tensor = torch.cat([local_tensor, torch.zeros(pad_n, D, device=device, dtype=local_tensor.dtype)], dim=0)
    gathered = [torch.zeros(max_n, D, device=device, dtype=local_tensor.dtype) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    return torch.cat([gathered[i][:all_sizes[i]] for i in range(world_size)], dim=0)

class GeometryMarginCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        img_emb = outputs["img_emb"].detach()
        txt_emb = outputs["txt_emb"].detach()
        B = img_emb.size(0)
        if B < 2: return
        img_norm = F.normalize(img_emb.float(), p=2, dim=-1)
        txt_norm = F.normalize(txt_emb.float(), p=2, dim=-1)
        sim_matrix = img_norm @ txt_norm.T
        true_sim = sim_matrix.diag().mean()
        mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
        random_sim = sim_matrix[mask].mean()
        margin = true_sim - random_sim
        pl_module.log("val/geometry_margin", margin, sync_dist=True, batch_size=B)

class ClinicalTopologyCallback(Callback):
    def __init__(self, ks=[1, 5]):
        super().__init__()
        self.ks = ks
        self.img_features = []
        self.txt_features = []
        self.labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.img_features.append(outputs["img_emb"].detach())
        self.txt_features.append(outputs["txt_emb"].detach())
        self.labels.append(outputs["labels"].detach().float())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.img_features: return
        all_img = torch.cat(self.img_features, dim=0)
        all_txt = torch.cat(self.txt_features, dim=0)
        all_lbl = torch.cat(self.labels, dim=0)
        if trainer.world_size > 1:
            all_img = _padded_all_gather(all_img, pl_module.device, trainer.world_size)
            all_txt = _padded_all_gather(all_txt, pl_module.device, trainer.world_size)
            all_lbl = _padded_all_gather(all_lbl, pl_module.device, trainer.world_size)
        all_img = F.normalize(all_img.float(), p=2, dim=-1)
        all_txt = F.normalize(all_txt.float(), p=2, dim=-1)
        metrics = {}
        n_total = all_img.size(0)
        i2t_sim_matrix = all_img @ all_txt.T
        max_k = min(max(self.ks), n_total)
        if max_k > 0:
            _, topk_txt_indices = i2t_sim_matrix.topk(max_k, dim=1)
            target_labels = all_lbl.unsqueeze(1) 
            retrieved_labels = all_lbl[topk_txt_indices]
            pathology_indices = list(range(1, 13)) 
            target_path = (target_labels == 1.0)[:, :, pathology_indices]
            retrieved_path = (retrieved_labels == 1.0)[:, :, pathology_indices]
            semantic_match_matrix = (retrieved_path & target_path).any(dim=-1) 
            top1_retrieved = retrieved_labels[:, 0, :]
            r_intersection = ((top1_retrieved == 1.0) & (all_lbl == 1.0)).float().sum(-1)
            r_union = ((top1_retrieved == 1.0) | (all_lbl == 1.0)).float().sum(-1)
            metrics["val/Retrieval_Jaccard"] = (r_intersection / (r_union + 1e-8)).mean()
            for k in self.ks:
                k_c = min(k, max_k)
                sem_precision = semantic_match_matrix[:, :k_c].any(dim=1).float().mean()
                metrics[f"val/Semantic_Recall@{k}"] = sem_precision
        if n_total > 5:
            v2v_sim_matrix = all_img @ all_img.T 
            v2v_sim_matrix.fill_diagonal_(-1)
            _, knn_indices = v2v_sim_matrix.topk(k=5, dim=1)
            knn_labels = all_lbl[knn_indices]
            knn_labels_clean = knn_labels.clone()
            knn_labels_clean[knn_labels_clean == -99.0] = float('nan')
            with torch.no_grad():
                majority_vote = (torch.nanmean(knn_labels_clean, dim=1) >= 0.5).float()
            target_valid = (all_lbl != -99.0)
            intersection = ((majority_vote == 1.0) & (all_lbl == 1.0)).float().sum(dim=-1)
            union = ((majority_vote == 1.0) | (all_lbl == 1.0)).float().sum(dim=-1)
            jaccard = intersection / (union + 1e-8)
            metrics["val/kNN_Jaccard"] = jaccard.mean()
            label_matches = (majority_vote == all_lbl) & target_valid
            per_label_acc = (label_matches.float().sum(0) / target_valid.float().sum(0).clamp(min=1))
            metrics["val/kNN_Accuracy@5"] = per_label_acc.mean()
        pl_module.log_dict(metrics, sync_dist=True, batch_size=n_total)
        self.img_features.clear()
        self.txt_features.clear()
        self.labels.clear()

class LinearProbeCallback(Callback):
    def __init__(self, num_classes: int = 14):
        super().__init__()
        self.num_classes = num_classes
        self.auc_metric = None
    def _init_metric(self, pl_module):
        self.auc_metric = MultilabelAUROC(
            num_labels=len(self.target_indices), 
            average="macro", 
            ignore_index=-99
        ).to(pl_module.device)
    def on_fit_start(self, trainer, pl_module):
        self.target_indices = trainer.datamodule.target_indices
        self._init_metric(pl_module)
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 1 or outputs is None: return
        probs = outputs.get("probs")
        labels = outputs.get("labels")
        if probs is None or labels is None: return
        target = labels.clone()
        mask = (target != 0) & (target != 1)
        target[mask] = -99
        
        try:
            self.auc_metric.update(probs[:, self.target_indices].float(), target[:, self.target_indices].long())
        except Exception as e:
            if trainer.is_global_zero:
                print(f"DEBUG: probs.shape={probs.shape}, target.shape={target.shape}, target_indices={self.target_indices}")
                print(f"DEBUG: Update failed: {e}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.auc_metric is None: return
        try:
            mean_auc = self.auc_metric.compute()
            if not torch.isnan(mean_auc):
                pl_module.log("val/auc_5x200", mean_auc, prog_bar=True, sync_dist=True)
        except Exception as e:
            if trainer.is_global_zero:
                print(f"Warning: LinearProbe AUC computation failed: {e}")
        self._init_metric(pl_module)
