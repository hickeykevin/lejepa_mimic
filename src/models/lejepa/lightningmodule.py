import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch.distributed as dist
import timm
# from loss import EppsPulley

from .components import (
    MultiModalEncoder,
    CXRMultiCropTransform,
    GeometryMarginCallback,
    ClinicalTopologyCallback,
    LinearProbeCallback,
    SIGReg
)

# ==========================================
# LIGHTNING MODULES
# ==========================================

class LeJEPA(L.LightningModule):
    """
    Minimalist LeJEPA implementation, faithful to Algorithm 2 from the paper.

    Key design properties (matching the official minimal example):
    - Single ViT backbone, NO momentum encoder / EMA / stop-gradient
    - 3-layer BN MLP projector (backbone_dim → 2048 → 2048 → proj_dim)
    - SIGReg: fresh random projection A drawn every forward call
    - Symmetric inv_loss: (proj.mean(0) - proj).square().mean()
    - Online linear probe on detached backbone [CLS] embeddings
    - LinearLR warmup → CosineAnnealingLR schedule
    - V (number of views) read from the datamodule at fit time
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        txt_model_name: str = None,
        proj_dim: int = 128,
        num_classes: int = 14,     # CheXpert labels for the probe
        lr: float = 2e-3,
        weight_decay: float = 5e-2,
        lamb: float                     = 0.10,     # λ weight of SIGReg relative to JEPA/inv
        sketch_dim: int                = 128,      # Sketching dimension for EEPP loss
        probe_lr: float                = 1.0e-3,   # Separate LR for the online probe
        probe_wd: float                = 1.0e-7,   # Separate WD for the online probe
        sigreg_target_variance: float  = 1.0,      # EEPP target variance
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Backbone ────────────────────────────────────────────────────────
        # num_classes=0 → raw [CLS] embedding, no classification head
        self.backbone = MultiModalEncoder(
            img_model_name=model_name,
            txt_model_name=txt_model_name,
            proj_dim=proj_dim,
        )
        backbone_dim = self.backbone.backbone_dim   # 384 for small, 768 for base

        # ── Online linear probe ──────────────────────────────────────────
        # Trains on detached backbone features; does not pollute the SSL gradient
        self.probe = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, num_classes),
        )

        self.sigreg_loss = SIGReg(
            num_slices=sketch_dim,
            t_max=3,
            n_points=17,
            target_variance=1.0,
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
        """Final synchronization point to ensure multi-node stability."""
        if self.trainer.world_size > 1:
            print(f"[Rank {self.global_rank}] Reached final sync barrier. Waiting for other nodes...")
            dist.barrier()
            print(f"[Rank {self.global_rank}] Sync complete. Starting training loop.")

    # ── Universal batch adapter ─────────────────────────────────────────────
    @staticmethod
    def prepare_batch(batch, device):
        """
        Converts a UniversalMimicCxrDataModule batch into independent image
        samples, matching the original LeJEPA assumption that each sample is
        a single image (not a study).

        Flattens [B, V_max, C, H, W] → [N_total, C, H, W] and replicates
        study-level labels to image-level so _compute_probe_loss receives
        [N_total, num_classes] instead of [B_studies, num_classes].

        Args:
            batch:  dict with keys 'images' [B, V_max, C, H, W],
                    'view_mask' [B, V_max], 'text', 'labels'
            device: torch.device to move tensors to
        Returns:
            dict with keys:
              'image'  : FloatTensor [N_total, C, H, W]
              'labels' : FloatTensor [N_total, num_classes] or None
        """
        if "view_mask" not in batch:
            return batch

        images    = batch["images"]     # [B, V_max, C, H, W]
        view_mask = batch["view_mask"]  # [B, V_max]
        labels    = batch.get("labels") # [B, num_classes] or None

        flat_imgs, flat_labels = [], []
        for b in range(images.size(0)):
            for v in range(images.size(1)):
                if view_mask[b, v]:
                    flat_imgs.append(images[b, v].to(device))
                    if labels is not None:
                        flat_labels.append(labels[b].to(device))  # replicate per image

        return {
            "image":  torch.stack(flat_imgs),                              # [N_total, C, H, W]
            "labels": torch.stack(flat_labels) if flat_labels else None,   # [N_total, num_classes]
        }

    # ── Training step ────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        # Accept both universal and legacy batch formats
        if "view_mask" in batch:
            batch = self.prepare_batch(batch, self.device)
        imgs = batch["image"]       # [N_total, C, H, W] (universal) or [N, V, C, H, W] (legacy)
        B    = imgs.shape[0]

        emb, proj = self(imgs)      # routes to single-view or multi-view path in forward()
        
        # ── Invariance loss: each view vs. mean of all views (symmetric) ──
        target   = proj.mean(0)                             # [N, proj_dim]
        inv_loss = (target.unsqueeze(0) - proj).square().mean()
        
        # ── SIGReg: single call over all views; handles ddp sync internally ──
        # proj is [V, N, D] (multi-view) or [N, D] (single-view)
        sigreg_loss = self.sigreg_loss(proj, global_step=self.global_step, world_size=self.trainer.world_size)

        # ── Combined LeJEPA loss ─────────────────────────────────────────
        lejepa_loss = self.hparams.lamb * sigreg_loss + (1 - self.hparams.lamb) * inv_loss
        
        # ── Online probe ────────────────────────────────────────────────
        # Universal format: labels are already [N_total, num_classes] — no study_map needed.
        # Legacy multi-view format: labels are [N, num_classes] and emb is [N*V, D],
        # so we replicate each study's label V times via study_map.
        if imgs.dim() == 5:
            # Legacy [N, V, C, H, W]: build study_map to expand labels
            V = imgs.shape[1]
            study_map = torch.tensor([[i] * V for i in range(B)], device=self.device).flatten()
            probe_loss = self._compute_probe_loss(emb, batch, study_map=study_map)
        else:
            # Universal [N_total, C, H, W]: labels already replicated, probe directly
            probe_loss = self._compute_probe_loss(emb, batch)
        
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
        # Universal format: pick the first valid view per study as the val image
        if "view_mask" in batch:
            images    = batch["images"]      # [B, V_max, C, H, W]
            view_mask = batch["view_mask"]   # [B, V_max]
            # For each study, select the first valid view
            first_valid = view_mask.int().argmax(dim=1)  # [B]
            imgs = images[torch.arange(images.size(0)), first_valid].to(self.device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(self.device)
            batch = {"image": imgs, "labels": labels}

        imgs = batch["image"]   # [B, C, H, W] — one canonical view per study
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
    """
    DINO-style LeJEPA: 2 global views define a teacher centre; 8 local views
    are pulled toward that centre via the invariance loss (asymmetric DINO).
    SIGReg over all views prevents representation collapse.

    The datamodule runs in raw_mode=True — the dataset returns full-resolution
    Tensors under 'raw_image'. RawImageCollate keeps them as a list.
    In training_step the images are moved to the GPU and then cropped via
    CXRMultiCropTransform, which fully supports CUDA tensors.
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
            **kwargs,
        )
        self.hparams.update(dict(
            n_global=n_global, n_local=n_local,
            global_size=global_size, local_size=local_size,
            scale_global=scale_global, scale_local=scale_local,
            brightness=brightness, contrast=contrast,
        ))
        self.multicrop = CXRMultiCropTransform(
            global_size=global_size, local_size=local_size,
            scale_global=scale_global, scale_local=scale_local,
            n_global=n_global, n_local=n_local,
            brightness=brightness, contrast=contrast,
        )

    @staticmethod
    def prepare_batch(batch, device):
        """
        Converts a UniversalMimicCxrDataModule batch into a flat list of
        individual image tensors. Each image becomes its own sample — DINOLeJEPA
        then applies multicrop (2 global + N local) to each image independently.

        Study-level labels are replicated per image so _compute_probe_loss
        receives [N_total, num_classes] aligned with the flattened images.

        Returns:
            dict with keys:
              'image'  : list[Tensor [C, H, W]] — one entry per image (N_total)
              'labels' : FloatTensor [N_total, num_classes] or None
        """
        if "view_mask" not in batch:
            return batch

        images    = batch["images"]     # [B, V_max, C, H, W]
        view_mask = batch["view_mask"]  # [B, V_max]
        labels    = batch.get("labels") # [B, num_classes] or None

        flat_imgs, flat_labels = [], []
        for b in range(images.size(0)):
            for v in range(images.size(1)):
                if view_mask[b, v]:
                    flat_imgs.append(images[b, v].to(device))
                    if labels is not None:
                        flat_labels.append(labels[b].to(device))  # replicate per image

        return {
            "image":  flat_imgs,
            "labels": torch.stack(flat_labels) if flat_labels else None,  # [N_total, num_classes]
        }

    def training_step(self, batch, batch_idx):
        if "view_mask" in batch:
            batch = self.prepare_batch(batch, self.device)
        raw_imgs = batch["image"]  # list[Tensor [C, H, W]] — one entry per image
        # B = number of images (each image is an independent sample for multicrop)
        B = len(raw_imgs)
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
        sigreg_losses = [self.sigreg_loss(view_proj, global_step=self.global_step, world_size=self.trainer.world_size) for view_proj in all_views_vfirst]
        sigreg_loss   = torch.stack(sigreg_losses).mean()

        # ── 6. Combined LeJEPA objective ───────────────────────────────
        lejepa_loss = (1 - self.hparams.lamb) * inv_loss + self.hparams.lamb * sigreg_loss

        # ── 7. Online probe (detached image representations) ──────────────
        # emb_g is [B*ng, D]; map each global-view embedding back to its source
        # image (sample). batch["labels"] is already [N_total, num_classes].
        study_map = torch.tensor(
            [i for i in range(B) for _ in range(ng)],
            device=self.device,
        )
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
    """
    Multimodal LeJEPA using text as the global anchor.

    The BiomedVLP-CXR-BERT backbone is frozen and acts as a fixed semantic
    oracle — only its projection head (txt_proj) is trained.  This design:
      - eliminates text-encoder collapse (SIGReg keeps txt_proj output isotropic)
      - avoids drifting the pretrained LM representations under SSL pressure
      - is the multimodal equivalent of stop-grad: a frozen anchor is a
        provably stable target, making an explicit detach() on proj_txt
        unnecessary (gradient can only flow through txt_proj anyway)

    Full image embeddings belonging to a study are pulled toward the study's
    text embedding. Diversity comes from multiple images per study (when
    available). If a study has only one image, it reduces to cross-modal
    alignment.
    """
    def __init__(
        self,
        model_name:    str   = "vit_small_patch16_224",
        txt_model_name: str  = "microsoft/BiomedVLP-CXR-BERT-specialized",
        proj_dim:      int   = 256,
        num_classes:   int   = 14,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            txt_model_name=txt_model_name,
            proj_dim=proj_dim,
            num_classes=num_classes,
            **kwargs
        )

    @staticmethod
    def prepare_batch(batch, device):
        """
        Converts a UniversalMimicCxrDataModule batch into the flattened
        (images, study_map, text) format expected by TextAnchoredLeJEPA.

        Variable view counts are handled naturally: each image gets a slot
        in the flat tensor and a corresponding study_map index, so studies
        with 1 image and studies with 5 images coexist in the same batch.

        Returns:
            dict with keys:
              'images'    : FloatTensor [N_total, C, H, W]
              'study_map' : LongTensor  [N_total]
              'text'      : tokenized dict on device  [B, seq_len]
              'labels'    : FloatTensor [B, num_cls] or None
        """
        if "view_mask" not in batch:
            # Legacy StudyCollateFn format — already in the right shape
            return batch

        images    = batch["images"]    # [B, V_max, C, H, W]
        view_mask = batch["view_mask"] # [B, V_max]

        flat_imgs, study_map = [], []
        for b in range(images.size(0)):
            for v in range(images.size(1)):
                if view_mask[b, v]:
                    flat_imgs.append(images[b, v])
                    study_map.append(b)

        flat_imgs = torch.stack(flat_imgs).to(device)          # [N_total, C, H, W]
        study_map = torch.tensor(study_map, dtype=torch.long, device=device)
        text      = {k: v.to(device) for k, v in batch["text"].items()}
        labels    = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)

        return {
            "images":    flat_imgs,
            "study_map": study_map,
            "text":      text,
            "labels":    labels,
        }

    def training_step(self, batch, batch_idx):
        # Accept both universal and legacy batch formats
        if "view_mask" in batch:
            batch = self.prepare_batch(batch, self.device)
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

        # 4. SIGReg per-modality independently
        sig_txt = self.sigreg_loss(proj_txt, global_step=self.global_step, world_size=self.trainer.world_size)
        sig_img = self.sigreg_loss(proj_img, global_step=self.global_step, world_size=self.trainer.world_size)
        sigreg_loss = (sig_txt + sig_img) / 2

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
        # Accept both universal and legacy batch formats
        if "view_mask" in batch:
            batch = self.prepare_batch(batch, self.device)
        images      = batch["images"]      # [N_total, C, H, W]
        study_map   = batch["study_map"]    # [N_total]
        text_tokens = batch["text"]        # [B, seq_len]
        B = text_tokens["input_ids"].shape[0]

        emb_txt, proj_txt = self.backbone.forward_txt(**text_tokens)
        emb_img, proj_img = self(images)
        
        mapped_proj_txt = proj_txt[study_map]
        inv_loss = (mapped_proj_txt - proj_img).pow(2).mean()
        
        # SIGReg for validation monitoring — no distributed reduction during val
        # (validation batches are not guaranteed to be aligned across ranks)
        sig_txt = self.sigreg_loss(proj_txt, global_step=self.global_step, world_size=1)
        sig_img = self.sigreg_loss(proj_img, global_step=self.global_step, world_size=1)
        sigreg_loss = (sig_txt + sig_img) / 2
        
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