"""
Universal DataModule for MIMIC-CXR.

This module provides a single, model-agnostic batch format for all LeJEPA
variants. The DataModule is responsible only for I/O and basic normalization.
Model-specific formatting (flattening, multi-crop, etc.) is handled inside
each LightningModule via its own `prepare_batch()` method.

Batch Format (output of every dataloader):
    images:    FloatTensor [B, V_max, C, H, W]   – padded to max views in batch
    view_mask: BoolTensor  [B, V_max]             – True = real image, False = pad
    text:      dict(input_ids, attention_mask, …) – tokenized reports [B, seq]
    labels:    FloatTensor [B, num_classes]        – CheXpert labels (optional)
    study_ids: list[str]                           – study IDs for bookkeeping
"""

import warnings
from typing import Any, Dict, List, Optional

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from .base_datamodule import BaseMimicDataModule
from .dataset import StudyMimicCxrDataset
from .datamodule import _build_5x200_df

BIOMEDVLP_TOKENIZER = "microsoft/BiomedVLP-CXR-BERT-specialized"

class UniversalStudyCollate:
    """
    Pads variable-length image lists to [B, V_max, C, H, W] and produces
    a boolean view_mask so downstream models can ignore padding slots.

    This is the ONLY collate function needed; models do their own formatting
    inside `prepare_batch()`.
    """
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        captions   = [item["caption"]   for item in batch]
        study_ids  = [item["study_id"]  for item in batch]

        # Determine the maximum number of views in this batch
        view_counts = [len(item["images"]) for item in batch]
        V_max       = max(view_counts)
        B           = len(batch)

        # Peek at the image shape from the first non-empty study
        ref_image = batch[0]["images"][0]
        C, H, W   = ref_image.shape

        # Build padded image tensor and mask
        images    = torch.zeros(B, V_max, C, H, W, dtype=ref_image.dtype)
        view_mask = torch.zeros(B, V_max, dtype=torch.bool)
        for i, item in enumerate(batch):
            for v, img in enumerate(item["images"]):
                images[i, v]    = img
                view_mask[i, v] = True

        # Tokenize reports
        tokens = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        collated: Dict[str, Any] = {
            "images":    images,      # [B, V_max, C, H, W]
            "view_mask": view_mask,   # [B, V_max]  True = valid image
            "text":      tokens,      # dict  [B, seq_len]
            "study_ids": study_ids,
        }

        # Labels are optional (use_labels=True must be set in dataset)
        labels = [item.get("labels") for item in batch]
        if all(l is not None for l in labels):
            collated["labels"] = torch.stack(labels)  # [B, num_classes]

        return collated


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class UniversalMimicCxrDataModule(BaseMimicDataModule):
    """
    Universal Study-level DataModule for MIMIC-CXR.
    Always yields 224x224 images in the canonical format.
    """

    def __init__(
        self,
        metadata_csv: str,
        reports_root: str,
        images_root: str,
        batch_size:  int  = 16,
        num_workers: int  = 4,
        use_labels:  bool = False,
        labels_csv:  Optional[str] = None,
        max_length:  int  = 512,
    ):
        super().__init__(
            metadata_csv=metadata_csv,
            use_labels=use_labels,
            labels_csv=labels_csv,
        )
        self.reports_root = str(reports_root)
        self.images_root  = str(images_root)
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.max_length   = max_length

        # Hardcoded 224x224 transforms for everything
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        self.val_transforms = self.train_transforms

        # Tokenizer is fixed to BiomedVLP-CXR-BERT.
        from transformers import AutoTokenizer
        self.tokenizer  = AutoTokenizer.from_pretrained(
            BIOMEDVLP_TOKENIZER, trust_remote_code=True
        )
        self.collate_fn = UniversalStudyCollate(self.tokenizer, max_length=max_length)

        self.save_hyperparameters()

    # ── Setup ───────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        train_df, val_df = self._get_split_dfs()

        self.train_ds = StudyMimicCxrDataset(
            train_df, self.reports_root, self.images_root,
            transform=self.train_transforms, use_labels=self.use_labels,
        )
        self.val_ds = StudyMimicCxrDataset(
            val_df, self.reports_root, self.images_root,
            transform=self.val_transforms, use_labels=self.use_labels,
        )

        # 5×200 Balanced Validation Subset ───────────────────────────────────
        if self.use_labels:
            val_studies_df      = val_df.drop_duplicates("study_id")
            val_5x200_study_df  = _build_5x200_df(val_studies_df)
            val_5x200_df        = val_df[
                val_df["study_id"].isin(val_5x200_study_df["study_id"])
            ].reset_index(drop=True)
        else:
            val_study_ids  = val_df["study_id"].unique()
            sample_ids     = pd.Series(val_study_ids).sample(
                min(1000, len(val_study_ids)), random_state=42
            )
            val_5x200_df   = val_df[
                val_df["study_id"].isin(sample_ids)
            ].reset_index(drop=True)

        self.val_5x200_ds = StudyMimicCxrDataset(
            val_5x200_df, self.reports_root, self.images_root,
            transform=self.val_transforms, use_labels=self.use_labels,
        )

    # ── DataLoaders ─────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,                          # DDP epoch alignment
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> List[DataLoader]:
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        return [
            DataLoader(self.val_ds,       **loader_kwargs),
            DataLoader(self.val_5x200_ds, **loader_kwargs),
        ]
