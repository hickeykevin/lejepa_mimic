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

import os
import time

def load_hf_model_safely(model_name: str, **kwargs) -> AutoModel:
    """Loads a HuggingFace model safely in a distributed environment.

    Ensures only the local rank 0 on each node downloads the model first to avoid
    race conditions on the filesystem. Other ranks wait until the model is cached.

    Args:
        model_name: The name or path of the HuggingFace model to load.
        **kwargs: Additional arguments passed to AutoModel.from_pretrained.

    Returns:
        The loaded HuggingFace AutoModel instance.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if local_rank == 0:
        # Local rank 0 performs the download/check
        AutoModel.from_pretrained(model_name, **kwargs)
        
    # All other ranks on the node wait for a short period to ensure 
    # the file-system lock is released before they try to read the cache.
    if local_rank != 0:
        time.sleep(2) # Give Rank 0 a head start to acquire the lock
        
    return AutoModel.from_pretrained(model_name, **kwargs)

class SIGReg(nn.Module):
    """Symmetric Isotropic Gaussian Regularization (SIGReg).

    This implementation computes the distance between the empirical characteristic
    function (ECF) of projected features and the theoretical characteristic function
    of a standard normal distribution. It is designed to work in distributed
    environments by synchronizing the ECF across nodes.

    Attributes:
        num_slices (int): Number of random 1D projections (slices) to use.
        target_variance (float): The desired variance for the Gaussian distribution.
        t (torch.Tensor): Integration nodes for the characteristic function.
        phi_theoretical (torch.Tensor): Theoretical characteristic function values.
    """
    def __init__(
        self, 
        num_slices: int = 256, 
        t_max: float = 5.0, 
        n_points: int = 17, 
        target_variance: float = 1.0
    ):
        """Initializes SIGReg with integration parameters and target variance.

        Args:
            num_slices: Number of random projections to sample.
            t_max: The range [-t_max, t_max] for the characteristic function integration.
            n_points: Number of integration points for the trapezoidal rule.
            target_variance: The variance of the target isotropic Gaussian distribution.
        """
        super().__init__()
        self.num_slices = num_slices
        self.target_variance = target_variance

        # 1. Precompute integration nodes (trapezoidal rule)
        t = torch.linspace(-t_max, t_max, n_points)
        self.register_buffer("t", t)

        # 2. Precompute the theoretical characteristic function (Standard Normal)
        # phi(t) = exp(-0.5 * sigma^2 * t^2)
        phi_theoretical = torch.exp(-0.5 * target_variance * t**2)
        self.register_buffer("phi_theoretical", phi_theoretical)

    def forward(self, x: torch.Tensor, global_step: int, world_size: int = 1) -> torch.Tensor:
        """Computes the SIGReg loss (Epps-Pulley statistic) for the input batch.

        Args:
            x: Input feature tensor of shape [N, D] or [V, N, D].
            global_step: The current trainer global step, used to seed random projections.
            world_size: Total number of distributed ranks for reduction.

        Returns:
            A scalar tensor representing the mean SIGReg statistic across all slices.
        """
        # Support both [N, D] and [V, N, D] by flattening view dimensions
        if x.dim() == 3:
            x = x.flatten(0, 1)
            
        N, D = x.shape
        device = x.device

        # 1. Deterministic Random Slicing (Synced seeded projection)
        g = torch.Generator(device=device).manual_seed(int(global_step))
        A = torch.randn(D, self.num_slices, generator=g, device=device)
        A /= A.norm(p=2, dim=0)

        # 2. Project data onto random 1D slices
        x_proj = x @ A

        # 3. Compute Empirical Characteristic Function (ECF)
        x_t = x_proj.unsqueeze(-1) * self.t 
        
        # We sum instead of mean initially to handle heterogeneous N correctly across ranks
        ecf_sum = torch.exp(1j * x_t).sum(dim=0) # [S, T]

        # 4. Multi-GPU Reduction
        # Synchronize the ECF sum and the total N across all ranks
        total_N = torch.tensor(float(N), device=device)
        
        if world_size > 1:
            dist.all_reduce(ecf_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_N, op=dist.ReduceOp.SUM)

        # Average back to get the global ECF
        ecf = ecf_sum / (total_N + 1e-8)

        # 5. Compute Weighted L2 Distance (Epps-Pulley Statistic)
        err = (ecf - self.phi_theoretical).abs().square()
        weighted_err = err * self.phi_theoretical

        # 6. Integration using Trapezoidal Rule
        T = torch.trapezoid(weighted_err, self.t, dim=-1)
        
        # Return the mean statistic across all random slices, scaled by global N
        return T.mean() * total_N


class MultiModalEncoder(nn.Module):
    """A dual-tower encoder for joint image and text representation learning.

    This class encapsulates a vision backbone (via timm) and a text backbone
    (via HuggingFace Transformers), each followed by a projection head to
    map features into a shared latent space.

    Attributes:
        img_backbone (nn.Module): The vision model used for image encoding.
        backbone_dim (int): The output dimension of the vision backbone.
        img_proj (nn.Sequential): The MLP projector for image features.
        txt_backbone (nn.Module, optional): The language model for text encoding.
        txt_proj (nn.Sequential, optional): The MLP projector for text features.
    """
    def __init__(
        self,
        img_model_name: str = "vit_small_patch16_224",
        txt_model_name: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        proj_dim: int = 128,
    ):
        """Initializes the multi-modal encoder with specific backbones.

        Args:
            img_model_name: Name of the timm vision model to create.
            txt_model_name: Name or path of the HuggingFace text model to load.
            proj_dim: The final dimension of the projected shared latent space.
        """
        super().__init__()
        self.img_backbone = timm.create_model(img_model_name, pretrained=False, num_classes=0, dynamic_img_size=True)
        self.backbone_dim = self.img_backbone.num_features
        self.img_proj = nn.Sequential(
            nn.Linear(self.backbone_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim)
        )
        
        if txt_model_name is not None:
            self.txt_backbone = load_hf_model_safely(txt_model_name, trust_remote_code=True)
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

    def forward(self, x: torch.Tensor, mode: Literal["img", "txt"] = "img"):
        """Dispatches the forward pass based on the input modality.

        Args:
            x: Input data (images for 'img' mode, tokenized text for 'txt' mode).
            mode: The modality of the input ('img' or 'txt').

        Returns:
            A tuple of (backbone_features, projected_embeddings).
        """
        if mode == "img":
            return self.forward_img(x)
        elif mode == "txt":
            return self.forward_txt(x)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward_img(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes an image batch and projects it into the shared space.

        Args:
            x: Input image tensor of shape [B, C, H, W].

        Returns:
            A tuple of (vision_features, image_projections).
        """
        feats = self.img_backbone(x)
        return feats, self.img_proj(feats)

    def forward_txt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes tokenized text and projects it into the shared space.

        Args:
            input_ids: Tokenized text IDs.
            attention_mask: Attention mask for the text sequence.
            **kwargs: Additional arguments for the text backbone.

        Returns:
            A tuple of (text_features, text_projections).
        """
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
    """Anatomically-biased RandomResizedCrop for Chest X-rays.

    This transform samples crop parameters biased towards the center of the
    chest (lungs/heart) using Gaussian distributions for the crop centers.

    Attributes:
        output_size (int): Size of the final output square image.
        scale (tuple): Range of relative areas to sample for crops.
        ratio (tuple): Range of aspect ratios to sample for crops.
        cx_mu, cy_mu (float): Gaussian means for the crop center coordinates.
        sigma_x, sigma_y (float): Gaussian standard deviations for crop centers.
        blur_p (float): Probability of applying Gaussian blur.
    """
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
        """Initializes the biased crop transform with specific CXR priors.

        Args:
            output_size: Output resolution.
            scale: Range of scale for RandomResizedCrop.
            ratio: Range of aspect ratios.
            cx_mu: Mean x-coordinate for the crop center.
            cy_mu: Mean y-coordinate for the crop center.
            sigma_x: Standard deviation for the x-coordinate.
            sigma_y: Standard deviation for the y-coordinate.
            blur_p: Probability of blurring the image.
            blur_kernel: Size of the Gaussian blur kernel.
            blur_sigma: Standard deviation of the Gaussian blur.
            brightness: Brightness jitter strength.
            contrast: Contrast jitter strength.
        """
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

    def _sample_crop_params(self, w: int, h: int) -> tuple[int, int, int, int]:
        """Samples the crop coordinates using anatomical priors.

        Args:
            w: Width of the input image.
            h: Height of the input image.

        Returns:
            A tuple of (top, left, height, width).
        """
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
        """Applies the anatomically-biased crop and augmentations to an image.

        Args:
            img: Input image tensor of shape [C, H, W].

        Returns:
            Transformed and normalized image tensor.
        """
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
    """Generates multiple global and local views for self-supervised learning.

    This transform is designed for frameworks like DINO or LeJEPA that utilize
    different scales (global/local) of the same image to learn invariant features.

    Attributes:
        n_global (int): Number of global views to generate.
        n_local (int): Number of local views to generate.
        global_crop (CXRAnatBiasedCrop): Transform for global views.
        local_crop (CXRAnatBiasedCrop): Transform for local views.
    """
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
        """Initializes the multi-crop transform with view counts and sizes.

        Args:
            global_size: Resolution for global views.
            local_size: Resolution for local views.
            scale_global: Scale range for global views.
            scale_local: Scale range for local views.
            n_global: Count of global views.
            n_local: Count of local views.
            brightness: Brightness jitter strength.
            contrast: Contrast jitter strength.
        """
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

    def forward(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        """Generates the multi-scale views for a single image.

        Args:
            img: Input image tensor.

        Returns:
            A dictionary containing 'global_views' and 'local_views' tensors.
        """
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
    """Performs an all-gather operation on tensors with potentially different sizes.

    Handles heterogeneous batch sizes across GPUs by padding tensors to the
    maximum size before gathering and then trimming them back to original sizes.

    Args:
        local_tensor: The tensor to gather from the current rank.
        device: The torch device to use.
        world_size: Total number of distributed processes.

    Returns:
        The concatenated tensor gathered from all ranks.
    """
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
    """Logs the margin between positive and random image-text similarity.

    The 'Geometry Margin' measures how much more similar an image is to its
    corresponding text than to a random text in the batch.
    """
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Computes and logs the similarity margin for the current batch.

        Args:
            trainer: The PyTorch Lightning Trainer.
            pl_module: The LightningModule being validated.
            outputs: Dictionary of outputs from the validation step.
            batch: The current batch of data.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the dataloader.
        """
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
    """Evaluates the model based on clinical semantic retrieval and topology.

    Calculates metrics like Semantic Recall, Jaccard Similarity, and kNN Accuracy
    using clinical labels to assess how well the latent space organizes pathologies.

    Attributes:
        ks (list[int]): Top-k values for retrieval metrics.
        img_features (list): Accumulator for image embeddings.
        txt_features (list): Accumulator for text embeddings.
        labels (list): Accumulator for clinical labels.
    """
    def __init__(self, ks=[1, 5]):
        super().__init__()
        self.ks = ks
        self.img_features = []
        self.txt_features = []
        self.labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Accumulates embeddings and labels from the current validation batch.

        Args:
            trainer: The PyTorch Lightning Trainer.
            pl_module: The LightningModule being validated.
            outputs: Dictionary of outputs from the validation step.
            batch: The current batch of data.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the dataloader.
        """
        self.img_features.append(outputs["img_emb"].detach())
        self.txt_features.append(outputs["txt_emb"].detach())
        self.labels.append(outputs["labels"].detach().float())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Computes global semantic metrics across all validation samples.

        Performs distributed gathering of embeddings, calculates retrieval
        metrics (Semantic Recall@K, Jaccard), and evaluates kNN performance.

        Args:
            trainer: The PyTorch Lightning Trainer.
            pl_module: The LightningModule being validated.
        """
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
    """Evaluates the frozen encoder using an online linear probe.

    Fits a linear layer on top of the representations to predict pathology
    labels, calculating AUROC to monitor the quality of learned features.

    Attributes:
        num_classes (int): Total number of pathology labels.
        auc_metric (MultilabelAUROC): Metric instance for evaluation.
    """
    def __init__(self, num_classes: int = 14):
        super().__init__()
        self.num_classes = num_classes
        self.auc_metric = None
    def _init_metric(self, pl_module):
        """Initializes or resets the MultilabelAUROC metric.

        Args:
            pl_module: The LightningModule used to determine the device.
        """
        self.auc_metric = MultilabelAUROC(
            num_labels=len(self.target_indices), 
            average="macro", 
            ignore_index=-99
        ).to(pl_module.device)
    def on_fit_start(self, trainer, pl_module):
        """Initializes metric tracking at the start of training.

        Retrieves the target label indices from the datamodule.

        Args:
            trainer: The PyTorch Lightning Trainer.
            pl_module: The LightningModule being trained.
        """
        self.target_indices = trainer.datamodule.target_indices
        self._init_metric(pl_module)
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Updates the AUC metric with probabilities from the linear probe.

        Args:
            trainer: The PyTorch Lightning Trainer.
            pl_module: The LightningModule being validated.
            outputs: Dictionary of outputs from the validation step.
            batch: The current batch of data.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the dataloader.
        """
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
        """Computes and logs the final AUC for the validation epoch.

        Args:
            trainer: The PyTorch Lightning Trainer.
            pl_module: The LightningModule being validated.
        """
        if self.auc_metric is None: return
        try:
            mean_auc = self.auc_metric.compute()
            if not torch.isnan(mean_auc):
                pl_module.log("val/auc_5x200", mean_auc, prog_bar=True, sync_dist=True)
        except Exception as e:
            if trainer.is_global_zero:
                print(f"Warning: LinearProbe AUC computation failed: {e}")
        self._init_metric(pl_module)
