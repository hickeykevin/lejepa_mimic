import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch.distributed as dist
import timm

from .components import (
    SIGReg,
    MultiModalEncoder,
    CXRMultiCropTransform,
    GeometryMarginCallback,
    ClinicalTopologyCallback,
    LinearProbeCallback
)

# ==========================================
# LIGHTNING MODULES
# ==========================================

class LeJEPA(L.LightningModule):
    """Minimalist implementation of LeJEPA (Learned Joint-Embedding Predictive Architecture).

    This implementation is faithful to Algorithm 2 from the paper and follows the
    axiomatic "No Predictor" design. It regularises representations using Sketched
    Isotropic Gaussian Regularisation (SIGReg) to prevent collapse while maintaining
    high downstream utility.

    Attributes:
        backbone (MultiModalEncoder): The multi-modal encoder (image/text).
        probe (nn.Sequential): Online linear probe for tracking representation quality.

    Design Properties:
        - Single ViT/CNN backbone (no momentum encoder/EMA teacher).
        - 3-layer MLP projector with BatchNorm.
        - SIGReg: Distribution matching objective using Epps-Pulley test.
        - Online linear probe on detached representations.
    """

    def __init__(
        self,
        model_name:             str   = "vit_small_patch16_224",
        txt_model_name:         str   = None,
        proj_dim:               int   = 128,
        num_classes:            int   = 14,
        lr:                     float = 2e-3,
        weight_decay:           float = 5e-2,
        lamb:                   float = 0.10,
        warmup_epochs:          int   = 5,
        sketch_dim:             int   = 128,
        probe_lr:               float = 1.0e-3,
        probe_wd:               float = 1.0e-7,
        sigreg_target_variance: float = 1.0,
        **kwargs,
    ):
        """Initializes the LeJEPA LightningModule.

        Args:
            model_name: Name of the image backbone (timm-compatible).
            txt_model_name: Name of the text backbone (HuggingFace-compatible).
            proj_dim: Dimension of the projection space.
            num_classes: Number of classes for the online linear probe.
            lr: Peak learning rate for the backbone and projector.
            weight_decay: Weight decay for the backbone and projector.
            lamb: Weight of SIGReg loss relative to the invariance loss (lambda).
            warmup_epochs: Number of epochs for linear LR warmup.
            sketch_dim: Number of random projections for SIGReg.
            probe_lr: Learning rate for the online linear probe.
            probe_wd: Weight decay for the online linear probe.
            sigreg_target_variance: Target variance for EEPP distribution matching.
            **kwargs: Additional hyperparameters (e.g., scheduler configs).
        """
        super().__init__()
        self.save_hyperparameters()

        # ── Backbone ──
        self.backbone = MultiModalEncoder(
            img_model_name=model_name,
            txt_model_name=txt_model_name,
            proj_dim=proj_dim,
        )
        backbone_dim = self.backbone.backbone_dim

        # ── Online linear probe ──
        # Detached features: ensures the probe doesn't affect SSL gradients
        self.probe = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, V, C, H, W]  (multi-view) or [N, C, H, W] (single-view val)
        Returns:
            emb:  [N*V, backbone_dim]   raw backbone embeddings
            proj: [V, N, proj_dim]      projected embeddings (view-first)
        """
        if x.dim() == 4:
            # Single-view inference path (validation)
            emb, proj = self.backbone(x)           # [N, D]
            return emb, proj

        N, V = x.shape[:2]
        # Flatten views into batch, run once through backbone
        flat = x.flatten(0, 1)              # [N*V, C, H, W]
        emb, proj = self.backbone.forward_img(flat)           # [N*V, D]
        proj = proj.reshape(N, V, -1).transpose(0, 1)  # [V, N, proj_dim]
        return emb, proj

    def _sigreg(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 1. Setup (Fixed seed per step ensures 'A' is the same on all GPUs)
        dev = dict(device=x.device)
        g = torch.Generator(**dev).manual_seed(int(self.global_step))
        A = torch.randn(x.size(1), self.hparams.sketch_dim, generator=g, **dev)
        A /= A.norm(p=2, dim=0)

        # 2. Integration Points
        t = torch.linspace(-5, 5, 17, **dev)
        exp_f = torch.exp(-0.5 * self.hparams.sigreg_target_variance * t**2)
        
        # 3. Compute local sums in float32 for safety
        x_t = (x @ A).unsqueeze(2) * t
        cos_sum = x_t.cos().sum(0, dtype=torch.float32) # [M, T]
        sin_sum = x_t.sin().sum(0, dtype=torch.float32) # [M, T]
        local_n = torch.tensor([float(x.size(0))], device=x.device, dtype=torch.float32)

        # 4. SINGLE Synchronization Point
        if self.trainer.world_size > 1:
            # Flatten and concatenate all data into one buffer
            sync_buffer = torch.cat([cos_sum.flatten(), sin_sum.flatten(), local_n])
            dist.all_reduce(sync_buffer, op=dist.ReduceOp.SUM)
            
            # Unpack the reduced data
            M, T = cos_sum.shape
            cos_sum = sync_buffer[:M*T].reshape(M, T)
            sin_sum = sync_buffer[M*T:2*M*T].reshape(M, T)
            global_n = sync_buffer[-1].item()
        else:
            global_n = local_n.item()

        # 5. Distance Calculation
        err = (cos_sum / global_n - exp_f).square() + (sin_sum / global_n).square()
        return (torch.trapz(err * exp_f, t, dim=1) * global_n).mean()



    # def _sigreg(self, x: torch.Tensor) -> torch.Tensor:
    #     """Verbatim Algorithm 1 from the paper."""
    #     global_step = int(self.global_step)
    #     num_slices  = self.hparams.sketch_dim

    #     # slice sampling -- synced across devices
    #     dev = dict(device=x.device)
    #     g = torch.Generator(**dev)
    #     g.manual_seed(global_step)

    #     proj_shape = (x.size(1), num_slices)
    #     A = torch.randn(proj_shape, generator=g, **dev)
    #     A /= A.norm(p=2, dim=0)

    #     # -- Epps-Pulley stat. --
    #     # integration points
    #     t = torch.linspace(-5, 5, 17, **dev)
    #     # theoretical CF for N(0, 1) and Gauss. window
    #     exp_f = torch.exp(-0.5 * self.hparams.sigreg_target_variance * t**2)
        
    #     # empirical CF -- gathered across devices
    #     x_t = (x @ A).unsqueeze(2) * t      # (N, M, T)
    #     ecf = (1j * x_t).exp().mean(0)      # (M, T)
        
    #     if self.trainer.world_size > 1:
    #         dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
        
    #     # weighted L2 distance
    #     err = (ecf - exp_f).abs().square().mul(exp_f)
    #     N = x.size(0) * self.trainer.world_size
    #     T = torch.trapz(err, t, dim=1) * N
    #     return T.mean()

    def _compute_probe_loss(self, z, batch, study_map=None):
        """Compute binary cross-entropy for the online linear probe on detached embeddings."""
        if "labels" not in batch or batch["labels"] is None:
            return torch.tensor(0.0, device=self.device, dtype=z.dtype)
            
        y = batch["labels"]
        # Expand labels to match view-level embeddings via study_map (image-to-study index)
        if study_map is not None:
            y = y[study_map]
            
        valid_mask = (y == 0.0) | (y == 1.0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device, dtype=z.dtype)
            
        # Linear probe trains on detached backbone/projector features
        logits = self.probe(z.detach())
        return F.binary_cross_entropy_with_logits(logits[valid_mask], y[valid_mask].type_as(logits))

    def on_train_start(self):
        """Override to ensure the vision encoder stays in eval mode during training."""
        self.backbone.eval()

    # ── Training step ────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        imgs = batch["image"]       # [N, V, C, H, W]
        B    = imgs.shape[0]

        emb, proj = self(imgs)      # emb: [N*V, D], proj: [V, N, proj_dim]
        
        # ── Invariance loss: each view vs. mean of all views (symmetric) ──
        target   = proj.mean(0)                             # [N, proj_dim]
        inv_loss = (target.unsqueeze(0) - proj).square().mean()
        
        # ── SIGReg: Per-view average ─────────────────────────────────────
        # proj: [V, N, proj_dim]
        sigreg_losses = [self._sigreg(view_proj) for view_proj in proj]
        sigreg_loss   = torch.stack(sigreg_losses).mean()

        # ── Combined LeJEPA loss ─────────────────────────────────────────
        lejepa_loss = self.hparams.lamb * sigreg_loss + (1 - self.hparams.lamb) * inv_loss
        
        # ── Online probe (detached; doesn't contaminate SSL gradient) ────
        study_map = torch.tensor([[i] * imgs.shape[1] for i in range(B)], device=self.device).flatten()
        probe_loss = self._compute_probe_loss(emb, batch, study_map=study_map)
        
        loss = lejepa_loss + probe_loss

        self.log_dict({
            "train/inv":        inv_loss,
            "train/sigreg":     sigreg_loss,
            "train/lejepa":     lejepa_loss,
            "train/probe":      probe_loss,
            "train/loss":       loss,
        }, prog_bar=False, batch_size=B, on_step=True, on_epoch=True)

        return loss

    # ── Validation step ──────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch["image"]   # [N, C, H, W]  (val always single-view)
        B    = imgs.shape[0]

        emb, _ = self(imgs)     # [N, D]
        
        probe_loss = torch.tensor(0.0, device=self.device)
        probe_probs = None
        if "labels" in batch and batch["labels"] is not None:
            y            = batch["labels"]
            probe_logits = self.probe(emb)
            probe_probs  = torch.sigmoid(probe_logits)
            valid        = (y == 0.0) | (y == 1.0)
            if valid.any():
                probe_loss = F.binary_cross_entropy_with_logits(
                    probe_logits[valid], y[valid].type_as(probe_logits)
                )

        if dataloader_idx == 0:
            self.log("val/probe_loss", probe_loss, sync_dist=True,
                     add_dataloader_idx=False, batch_size=B)

        return {
            "emb":    emb,
            "probs":  probe_probs,
            "labels": batch.get("labels"),
        }

    def configure_callbacks(self):
        return [
            LinearProbeCallback(num_classes=self.hparams.num_classes)
        ]

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    def configure_optimizers(self):
        # Two parameter groups: backbone + projector (higher WD) vs. probe (near-zero WD)
        g1 = {"params": self.backbone.parameters(),
              "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay}
        g2 = {"params": self.probe.parameters(),
              "lr": self.hparams.probe_lr, "weight_decay": self.hparams.probe_wd}
        opt = torch.optim.AdamW([g1, g2])

        # Steps per epoch: read from trainer once fit starts
        total_steps     = max(1, int(self.trainer.estimated_stepping_batches))
        max_epochs      = max(1, self.trainer.max_epochs or 1)
        steps_per_epoch = max(1, total_steps // max_epochs)
        
        warmup_steps    = steps_per_epoch * self.hparams.warmup_epochs
        
        # Safety for short runs: ensure warmup doesn't consume the entire run 
        # so that CosineAnnealing (s2) has a valid T_max > 0.
        if warmup_steps >= total_steps:
            warmup_steps = total_steps // 2

        cosine_steps = max(1, total_steps - warmup_steps)
        s1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
        s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_steps, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
        return [opt], [{"scheduler": scheduler, "interval": "step"}]


class DINOLeJEPA(LeJEPA):
    """DINO-style LeJEPA using multi-crop augmentation.

    This variant implements an asymmetric invariance objective inspired by DINO:
    two global views define a semantic "teacher" center, while multiple local
    views (smaller crops) are pulled toward that center.

    Key Properties:
        - Asymmetric Pull: Student views predict a teacher center.
        - Multi-Crop: Leverages different scales for better feature locality.
        - On-Device Augmentation: Crops are applied to GPU tensors for speed.
    """

    def __init__(
        self,
        model_name:    str   = "vit_small_patch16_224",
        txt_model_name: str  = None,
        proj_dim:      int   = 256,
        num_classes:   int   = 14,
        lr:            float = 2e-3,
        weight_decay:  float = 5e-2,
        lamb:          float = 0.05,
        warmup_epochs: int   = 5,
        sketch_dim:    int   = 256,
        # Multi-crop geometry
        n_global:      int   = 2,
        n_local:       int   = 8,
        global_size:   int   = 224,
        local_size:    int   = 96,
        scale_global:  tuple = (0.40, 1.00),
        scale_local:   tuple = (0.05, 0.40),
        brightness:    float = 0.15,
        contrast:      float = 0.15,
        **kwargs,
    ):
        """Initializes the DINOLeJEPA variant.

        Args:
            model_name: Image backbone name.
            txt_model_name: Text backbone name (optional).
            proj_dim: Projection dimension.
            num_classes: Linear probe classes.
            lr: Peak learning rate.
            weight_decay: Weight decay.
            lamb: SIGReg loss weight.
            warmup_epochs: Learning rate warmup period.
            sketch_dim: SIGReg sketching dimension.
            n_global: Number of global views (large crops).
            n_local: Number of local views (small crops).
            global_size: Resolution of global views.
            local_size: Resolution of local views.
            scale_global: Area scale for global crops.
            scale_local: Area scale for local crops.
            brightness: Color jitter brightness.
            contrast: Color jitter contrast.
            **kwargs: Additional hyperparameters.
        """
        super().__init__(
            model_name=model_name,
            txt_model_name=txt_model_name,
            proj_dim=proj_dim,
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            lamb=lamb,
            warmup_epochs=warmup_epochs,
            sketch_dim=sketch_dim,
            n_global=n_global,
            n_local=n_local,
            global_size=global_size,
            local_size=local_size,
            scale_global=scale_global,
            scale_local=scale_local,
            brightness=brightness,
            contrast=contrast,
            **kwargs,
        )
        self.multicrop = CXRMultiCropTransform(
            global_size=global_size,
            local_size=local_size,
            scale_global=scale_global,
            scale_local=scale_local,
            n_global=n_global,
            n_local=n_local,
            brightness=brightness,
            contrast=contrast,
        )

    def training_step(self, batch, batch_idx):
        raw_imgs = batch["image"]   # list[Tensor [C, H, W]], variable size
        B        = len(raw_imgs)
        ng       = self.hparams.n_global
        nl       = self.hparams.n_local
        gs       = self.hparams.global_size
        ls       = self.hparams.local_size

        g_list, l_list = [], []
        for img in raw_imgs:
            crops = self.multicrop(img.to(self.device))
            g_list.append(crops["global_views"])   # [ng, 3, gs, gs]
            l_list.append(crops["local_views"])    # [nl, 3, ls, ls]

        global_views = torch.stack(g_list)   # [B, ng, 3, gs, gs]
        local_views  = torch.stack(l_list)   # [B, nl, 3, ls, ls]

        # ── 2. Encode global views → teacher centre ───────────────────
        emb_g, proj_g = self(global_views)   # emb_g: [B*ng, D], proj_g: [ng, B, d] (view-first)
        proj_g = proj_g.transpose(0, 1)      # → [B, ng, d] (batch-first)
        
        center = proj_g.mean(dim=1) # [B, d]

        _, proj_l = self(local_views)            # proj_l: [nl, B, d] (view-first)
        proj_l = proj_l.transpose(0, 1)          # → [B, nl, d] (batch-first)

        # ── 4. Asymmetric invariance loss ─────────────────────────────
        # Student views are pulled toward the global teacher centre.
        all_views_proj = torch.cat([proj_g, proj_l], dim=1) # [B, ng+nl, d]
        inv_loss = (center.unsqueeze(1) - all_views_proj).pow(2).mean()

        # ── 5. SIGReg over full view pool (per-view average) ──────────────
        # Combined views: [B, ng+nl, d] -> [ng+nl, B, d]
        all_views_vfirst = all_views_proj.transpose(0, 1)
        sigreg_losses = [self._sigreg(view_proj) for view_proj in all_views_vfirst]
        sigreg_loss   = torch.stack(sigreg_losses).mean()

        # ── 6. Combined LeJEPA objective ───────────────────────────────
        lejepa_loss = (1 - self.hparams.lamb) * inv_loss + self.hparams.lamb * sigreg_loss

        # ── 7. Online probe (detached image representations) ──────────────
        # Map labels to all global views in the batch
        study_map = torch.tensor([[i] * ng for i in range(B)], device=self.device).flatten()
        probe_loss = self._compute_probe_loss(emb_g, batch, study_map=study_map)

        loss = lejepa_loss + probe_loss

        self.log_dict({
            "train/inv":    inv_loss,
            "train/sigreg": sigreg_loss,
            "train/lejepa": lejepa_loss,
            "train/probe":  probe_loss,
            "train/loss":   loss,
        }, prog_bar=False, batch_size=B, on_step=True, on_epoch=True)

        return loss

    # ── validation_step, configure_callbacks, configure_optimizers, _sigreg ─
    # All inherited from LeJEPA without modification.




class TextAnchoredLeJEPA(LeJEPA):
    """Multimodal LeJEPA using text as a semantic anchor.

    This variant treats a frozen clinical text backbone (e.g., BiomedVLP-CXR-BERT)
    as a fixed semantic oracle. Image representations are pulled toward their
    corresponding study reports via an asymmetric invariance loss.

    Key Properties:
        - Semantic Anchoring: Text serves as a stable target that does not collapse.
        - Asymmetric Target: Image backbone learns from fixed text representations.
        - Efficient Training: Avoids drifting pretrained LM weights.
    """

    def __init__(
        self,
        model_name:         str   = "vit_small_patch16_224",
        txt_model_name:     str   = "microsoft/BiomedVLP-CXR-BERT-specialized",
        proj_dim:           int   = 256,
        num_classes:        int   = 14,
        lr:                 float = 2e-3,
        weight_decay:       float = 5e-2,
        lamb:               float = 0.10,
        warmup_epochs:      int   = 5,
        sketch_dim:         int   = 128,
        cross_modal_sigreg: bool  = False,
        **kwargs
    ):
        """Initializes the TextAnchoredLeJEPA variant.

        Args:
            model_name: Image backbone name.
            txt_model_name: Text backbone name.
            proj_dim: Projection dimension.
            num_classes: Linear probe classes.
            lr: Peak learning rate.
            weight_decay: Weight decay.
            lamb: SIGReg loss weight.
            warmup_epochs: Learning rate warmup period.
            sketch_dim: SIGReg sketching dimension.
            cross_modal_sigreg: If True, apply SIGReg to the *joint* pool of
                image and text projections to force cross-modal distribution
                matching. If False, apply SIGReg to each modality separately.
            **kwargs: Additional hyperparameters.
        """
        super().__init__(
            model_name=model_name,
            txt_model_name=txt_model_name,
            proj_dim=proj_dim,
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            lamb=lamb,
            warmup_epochs=warmup_epochs,
            sketch_dim=sketch_dim,
            cross_modal_sigreg=cross_modal_sigreg,
            **kwargs
        )

    def training_step(self, batch, batch_idx):
        images      = batch["images"]      # [N_total, C, H, W]
        study_map   = batch["study_map"]    # [N_total] mapping images to studies
        text_tokens = batch["text"]         # dict with input_ids, etc. [B, seq_len]
        B = text_tokens["input_ids"].shape[0]

        # 1. Encode text (Anchor)
        _, proj_txt = self.backbone.forward_txt(**text_tokens) # [B, D_proj]
        
        # 2. Encode full images
        emb_img, proj_img = self(images) # [N_total, D], [N_total, proj_dim]

        # 3. Asymmetric invariance loss
        # Each image is pulled toward its study's text anchor
        mapped_proj_txt = proj_txt[study_map] # [N_total, d]
        inv_loss = (mapped_proj_txt - proj_img).pow(2).mean()

        # 4. SIGReg — joint cross-modal pool or per-modality, depending on flag
        sigreg_loss = self._compute_sigreg_loss(proj_img, proj_txt)

        # 5. Combined loss
        lejepa_loss = (1 - self.hparams.lamb) * inv_loss + self.hparams.lamb * sigreg_loss

        # 6. Online probe (detached image-level embeddings)
        probe_loss = self._compute_probe_loss(emb_img.detach(), batch, study_map=study_map)

        loss = lejepa_loss + probe_loss

        self.log_dict({
            "train/inv":    inv_loss,
            "train/sigreg": sigreg_loss,
            "train/lejepa": lejepa_loss,
            "train/probe":  probe_loss,
            "train/loss":   loss,
        }, prog_bar=False, batch_size=B, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images      = batch["images"]      # [N_total, C, H, W]
        study_map   = batch["study_map"]    # [N_total]
        text_tokens = batch["text"]        # [B, seq_len]
        B = text_tokens["input_ids"].shape[0]

        emb_txt, proj_txt = self.backbone.forward_txt(**text_tokens)
        emb_img, proj_img = self(images)
        
        mapped_proj_txt = proj_txt[study_map]
        inv_loss = (mapped_proj_txt - proj_img).pow(2).mean()
        
        # Consistent SIGReg for validation monitoring
        sigreg_loss = self._compute_sigreg_loss(proj_img, proj_txt)
        
        val_loss = (1 - self.hparams.lamb) * inv_loss + self.hparams.lamb * sigreg_loss

        # Probe loss on study-aggregated image representations
        study_img_emb = []
        for study_idx in range(B):
            mask = (study_map == study_idx)
            if mask.any():
                study_img_emb.append(emb_img[mask].mean(0))
            else:
                study_img_emb.append(torch.zeros_like(emb_img[0]))
        study_img_emb = torch.stack(study_img_emb)
        
        probe_loss = self._compute_probe_loss(study_img_emb, batch)
        probe_probs = torch.sigmoid(self.probe(study_img_emb.detach()))

        if dataloader_idx == 0:
            self.log_dict({
                "val/inv": inv_loss,
                "val/sigreg": sigreg_loss,
                "val/probe_loss": probe_loss,
                "val/loss": val_loss
            }, sync_dist=True, batch_size=B, add_dataloader_idx=False)
            
        return {
            "loss":    val_loss,
            "img_emb": study_img_emb,
            "txt_emb": emb_txt, 
            "probs":   probe_probs,
            "labels":  batch.get("labels"),
        }

    # ── SIGReg dispatch helper ────────────────────────────────────────────────
    def _compute_sigreg_loss(
        self,
        proj_img: torch.Tensor,
        proj_txt: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SIGReg either jointly (cross-modal) or per-modality.

        Joint mode concatenates image and text projections into a single pool
        so that the Epps-Pulley test is run over the combined distribution,
        forcing both modalities to share the same isotropic Gaussian in the
        projection space — not just marginally, but jointly.

        Per-modality mode (default) applies SIGReg independently to each
        modality and averages, matching the original paper formulation.
        """
        if self.hparams.cross_modal_sigreg:
            # Joint pool: [N_img + B, proj_dim]
            joint_proj = torch.cat([proj_img, proj_txt], dim=0)
            return self._sigreg(joint_proj)
        else:
            sig_txt = self._sigreg(proj_txt)
            sig_img = self._sigreg(proj_img)
            return (sig_txt + sig_img) / 2

    def configure_optimizers(self):
        """
        Three parameter groups:
          g1 – image backbone + img_proj  (main SSL objective)
          g2 – txt_proj only              (trainable projection on frozen BERT)
          g3 – online probe               (near-zero WD, separate LR)

        txt_backbone is excluded entirely: it is frozen so it never needs
        a gradient or optimizer state, saving ~400 MB of GPU memory.
        """
        enc = self.backbone
        g1 = {
            "params": list(enc.img_backbone.parameters()) + list(enc.img_proj.parameters()),
            "lr": self.hparams.lr,
            "weight_decay": self.hparams.weight_decay,
        }
        g2 = {
            # txt_backbone is frozen; only train the projection head.
            # Low WD: the frozen BERT output is already stable, so the
            # projector doesn't need heavy regularization.
            "params": list(enc.txt_proj.parameters()),
            "lr": self.hparams.lr,
            "weight_decay": 1e-4,
        }
        g3 = {
            "params": self.probe.parameters(),
            "lr": self.hparams.probe_lr,
            "weight_decay": self.hparams.probe_wd,
        }
        opt = torch.optim.AdamW([g1, g2, g3])

        total_steps     = max(1, int(self.trainer.estimated_stepping_batches))
        max_epochs      = max(1, self.trainer.max_epochs or 1)
        steps_per_epoch = max(1, total_steps // max_epochs)
        warmup_steps    = steps_per_epoch * self.hparams.warmup_epochs
        if warmup_steps >= total_steps:
            warmup_steps = total_steps // 2
        cosine_steps = max(1, total_steps - warmup_steps)

        s1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
        s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_steps, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
        return [opt], [{"scheduler": scheduler, "interval": "step"}]


class MultiViewTextAnchoredLeJEPA(TextAnchoredLeJEPA):
    """Hybrid variant combining text anchoring with multi-crop augmentation.

    Synthesises the DINOLeJEPA (augmentation) and TextAnchoredLeJEPA (semantic
    anchoring) strategies. Global crops are pulled toward a frozen clinical text
    anchor, while local crops provide high-resolution local features that help
    regularise the embedding manifold.

    Key Properties:
        - Dual Objective: Semantic anchoring + local-to-global reconstruction.
        - Feature Diversity: Multi-crop leverages multiple scales of the same image.
        - Robust Regularisation: SIGReg handles the combined modalities and views.
    """

    def __init__(
        self,
        model_name:         str   = "vit_small_patch16_224",
        txt_model_name:     str   = "microsoft/BiomedVLP-CXR-BERT-specialized",
        proj_dim:           int   = 256,
        num_classes:        int   = 14,
        lr:                 float = 2e-3,
        weight_decay:       float = 5e-2,
        lamb:               float = 0.05,
        warmup_epochs:      int   = 5,
        sketch_dim:         int   = 256,
        cross_modal_sigreg: bool  = False,
        # Multi-crop geometry
        n_global:           int   = 2,
        n_local:            int   = 6,
        global_size:        int   = 224,
        local_size:         int   = 96,
        scale_global:       tuple = (0.40, 1.00),
        scale_local:        tuple = (0.05, 0.40),
        brightness:         float = 0.15,
        contrast:           float = 0.15,
        **kwargs,
    ):
        """Initializes the MultiViewTextAnchoredLeJEPA variant.

        Args:
            model_name: Image backbone name.
            txt_model_name: Text backbone name.
            proj_dim: Projection dimension.
            num_classes: Linear probe classes.
            lr: Peak learning rate.
            weight_decay: Weight decay.
            lamb: SIGReg loss weight.
            warmup_epochs: Learning rate warmup period.
            sketch_dim: SIGReg sketching dimension.
            cross_modal_sigreg: If True, apply joint SIGReg.
            n_global: Number of global crops per image.
            n_local: Number of local crops per image.
            global_size: Resolution of global crops.
            local_size: Resolution of local crops.
            scale_global: Area scale for global crops.
            scale_local: Area scale for local crops.
            brightness: Magnitude of brightness jitter.
            contrast: Magnitude of contrast jitter.
            **kwargs: Additional hyperparameters.
        """
        super().__init__(
            model_name=model_name,
            txt_model_name=txt_model_name,
            proj_dim=proj_dim,
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            lamb=lamb,
            warmup_epochs=warmup_epochs,
            sketch_dim=sketch_dim,
            cross_modal_sigreg=cross_modal_sigreg,
            n_global=n_global,
            n_local=n_local,
            global_size=global_size,
            local_size=local_size,
            scale_global=scale_global,
            scale_local=scale_local,
            brightness=brightness,
            contrast=contrast,
            **kwargs,
        )
        self.multicrop = CXRMultiCropTransform(
            global_size=global_size,
            local_size=local_size,
            scale_global=scale_global,
            scale_local=scale_local,
            n_global=n_global,
            n_local=n_local,
            brightness=brightness,
            contrast=contrast,
        )

    # ── Training step ────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        raw_imgs    = batch["images"]    # list[Tensor [C, H, W]], variable size
        study_map   = batch["study_map"] # [N_total] image → study index
        text_tokens = batch["text"]      # dict [B, seq_len]
        B           = text_tokens["input_ids"].shape[0]
        ng          = self.hparams.n_global

        # 1. Encode text anchor (frozen BERT + trainable txt_proj)
        _, proj_txt = self.backbone.forward_txt(**text_tokens)  # [B, proj_dim]

        # 2. On-device multi-crop augmentation per image
        #    raw_imgs is a flat list of N_total images (all images across the batch)
        g_list, l_list = [], []
        for img in raw_imgs:
            crops = self.multicrop(img.to(self.device))
            g_list.append(crops["global_views"])  # [ng, C, gs, gs]
            l_list.append(crops["local_views"])   # [nl, C, ls, ls]

        global_views = torch.stack(g_list)  # [N_total, ng, C, gs, gs]
        local_views  = torch.stack(l_list)  # [N_total, nl, C, ls, ls]
        N_total      = global_views.shape[0]

        # 3. Encode global crops — shape after forward: [N_total*ng, proj_dim]
        flat_global = global_views.flatten(0, 1)           # [N_total*ng, C, gs, gs]
        emb_global, proj_global = self.backbone.forward_img(flat_global)
        # proj_global: [N_total*ng, proj_dim]; emb_global: [N_total*ng, D]

        # 4. Encode local crops
        flat_local = local_views.flatten(0, 1)             # [N_total*nl, C, ls, ls]
        _, proj_local = self.backbone.forward_img(flat_local)  # [N_total*nl, proj_dim]

        # 5. Invariance loss: global crops → text anchor
        #    Expand study_map to cover all copies of each image's global crops
        study_map_global = study_map.repeat_interleave(ng)  # [N_total*ng]
        mapped_proj_txt  = proj_txt[study_map_global]       # [N_total*ng, proj_dim]
        inv_loss = (mapped_proj_txt - proj_global).pow(2).mean()

        # 6. SIGReg over the full view pool (global + local) and text projections
        all_img_proj = torch.cat([proj_global, proj_local], dim=0)  # [N_total*(ng+nl), proj_dim]
        sigreg_loss  = self._compute_sigreg_loss(all_img_proj, proj_txt)

        # 7. Combined loss
        lejepa_loss = (1 - self.hparams.lamb) * inv_loss + self.hparams.lamb * sigreg_loss

        # 8. Online probe on study-mean global backbone embeddings (detached)
        #    Average all global crops per study → one embedding per study
        emb_global_per_study = emb_global.reshape(N_total, ng, -1).mean(dim=1)  # [N_total, D]
        probe_loss = self._compute_probe_loss(
            emb_global_per_study.detach(), batch, study_map=study_map
        )

        loss = lejepa_loss + probe_loss

        self.log_dict({
            "train/inv":    inv_loss,
            "train/sigreg": sigreg_loss,
            "train/lejepa": lejepa_loss,
            "train/probe":  probe_loss,
            "train/loss":   loss,
        }, prog_bar=False, batch_size=B, on_step=True, on_epoch=True)

        return loss

    # ── Validation step ───────────────────────────────────────────────────────
    # Inherit TextAnchoredLeJEPA.validation_step unchanged — validation always
    # uses single-view pre-transformed images from the datamodule, so no
    # multi-crop augmentation is needed or desired here.

    # ── configure_optimizers ─────────────────────────────────────────────────
    # Fully inherited from TextAnchoredLeJEPA — the same three param groups
    # (img_backbone+img_proj, txt_proj, probe) apply here.

    # ── configure_callbacks ──────────────────────────────────────────────────
    # Inherited from LeJEPA (LinearProbeCallback).