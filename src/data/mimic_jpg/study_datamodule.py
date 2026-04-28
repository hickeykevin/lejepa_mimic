import warnings
from typing import Any, Dict, List, Optional

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from .base_datamodule import BaseMimicDataModule
from .dataset import StudyMimicCxrDataset
from .datamodule import _build_5x200_df
from .utils import MIMIC_5X200_FINDINGS

BIOMEDVLP_TOKENIZER = "microsoft/BiomedVLP-CXR-BERT-specialized"


class StudyCollateFn:
    """Handles variable image counts per study and report tokenization."""
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        flat_images, study_map, captions, raw_texts, study_ids, labels = [], [], [], [], [], []

        for i, item in enumerate(batch):
            study_images = item["images"]
            flat_images.extend(study_images)
            study_map.extend([i] * len(study_images))
            captions.append(item["caption"])
            raw_texts.append(item["raw_text"])
            study_ids.append(item["study_id"])
            if "labels" in item:
                labels.append(item["labels"])

        tokens = self.tokenizer(
            captions, padding=True, truncation=True,
            return_tensors="pt", max_length=self.max_length
        )

        collated = {
            "images":    torch.stack(flat_images) if flat_images else torch.empty(0),
            "study_map": torch.tensor(study_map, dtype=torch.long),
            "text":      tokens,
            "captions":  captions,
            "raw_text":  raw_texts,
            "study_ids": study_ids,
        }
        if labels:
            collated["labels"] = torch.stack(labels)
        return collated


class StudyMimicCxrDataModule(BaseMimicDataModule):
    """
    Study-level DataModule for MIMIC-CXR.
    Always uses BiomedVLP-CXR-BERT as the tokenizer.
    """
    def __init__(
        self,
        metadata_csv: str,
        reports_root: str,
        images_root: str,
        train_transforms: Optional[Any] = None,
        val_transforms: Optional[Any] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        use_labels: bool = False,
        labels_csv: Optional[str] = None,
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__(
            metadata_csv=metadata_csv,
            use_labels=use_labels,
            labels_csv=labels_csv,
            **kwargs
        )
        self.reports_root = str(reports_root)
        self.images_root = str(images_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        if self.train_transforms is None:
            warnings.warn(
                "No train_transforms provided. Falling back to minimal Resize+Normalize. "
                "Set train_transforms explicitly in your data config.",
                UserWarning, stacklevel=2
            )
            self.train_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if self.val_transforms is None:
            self.val_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Hardcoded tokenizer — BiomedVLP-CXR-BERT is the fixed text backbone.
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BIOMEDVLP_TOKENIZER, trust_remote_code=True)
        self.collate_fn = StudyCollateFn(self.tokenizer, max_length=max_length)
        self.save_hyperparameters(ignore=["train_transform", "val_transform"])

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

        # 5x200 Balanced Subset – only possible when label columns are present.
        if self.use_labels:
            val_studies_df = val_df.drop_duplicates("study_id")
            val_5x200_studies_df = _build_5x200_df(val_studies_df)
            val_5x200_df = val_df[val_df["study_id"].isin(val_5x200_studies_df["study_id"])].reset_index(drop=True)
        else:
            val_study_ids = val_df["study_id"].unique()
            sample_ids = pd.Series(val_study_ids).sample(min(1000, len(val_study_ids)), random_state=42)
            val_5x200_df = val_df[val_df["study_id"].isin(sample_ids)].reset_index(drop=True)

        self.val_5x200_ds = StudyMimicCxrDataset(
            val_5x200_df, self.reports_root, self.images_root,
            transform=self.val_transforms, use_labels=self.use_labels,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> List[DataLoader]:
        loader_params = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=True,
        )
        return [
            DataLoader(self.val_ds, **loader_params),
            DataLoader(self.val_5x200_ds, **loader_params),
        ]