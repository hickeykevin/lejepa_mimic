import warnings
from typing import Any, List, Optional

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from .base_datamodule import BaseMimicDataModule
from .dataset import MimicCxrDataset
from .utils import MIMIC_5X200_FINDINGS


def _build_5x200_df(val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a balanced validation subset of up to 200 samples per finding
    from the 5 key CheXpert findings. Samples are drawn without replacement
    across findings to avoid duplicates.
    """
    subset_dfs = []
    remaining_df = val_df.copy()
    for finding in MIMIC_5X200_FINDINGS:
        if finding not in remaining_df.columns:
            continue
        finding_df = remaining_df[remaining_df[finding] == 1.0]
        n_to_sample = min(200, len(finding_df))
        if n_to_sample > 0:
            sample = finding_df.sample(n=n_to_sample, random_state=42)
            subset_dfs.append(sample)
            remaining_df = remaining_df.drop(sample.index)
    return pd.concat(subset_dfs).reset_index(drop=True) if subset_dfs else val_df.sample(min(1000, len(val_df)))


class MimicCxrDataModule(BaseMimicDataModule):
    """
    Image-level DataModule for MIMIC-CXR.
    """

    def __init__(
        self,
        metadata_csv: str,
        reports_root: str,
        images_root: str,
        train_transforms: Optional[Any] = None,
        val_transforms: Optional[Any] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        use_labels: bool = False,
        labels_csv: Optional[str] = None,
    ):
        super().__init__(
            metadata_csv=metadata_csv,
            use_labels=use_labels,
            labels_csv=labels_csv
        )
        self.reports_root = str(reports_root)
        self.images_root = str(images_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        if self.train_transforms is None:
            warnings.warn(
                "No train_transform provided. Falling back to minimal Resize+Normalize. "
                "Set train_transform explicitly in your data config.",
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

        self.save_hyperparameters(ignore=["train_transforms", "val_transforms"])

    def setup(self, stage: Optional[str] = None) -> None:
        train_df, val_df = self._get_split_dfs()

        self.train_ds = MimicCxrDataset(
            train_df, self.reports_root, self.images_root,
            transform=self.train_transforms, use_labels=self.use_labels,
        )
        self.val_ds = MimicCxrDataset(
            val_df, self.reports_root, self.images_root,
            transform=self.val_transforms, use_labels=self.use_labels,
        )

        # 5x200 Balanced Subset – only possible when label columns are present.
        if self.use_labels:
            val_5x200_df = _build_5x200_df(val_df)
        else:
            val_5x200_df = val_df.sample(min(1000, len(val_df)), random_state=42)

        self.val_5x200_ds = MimicCxrDataset(
            val_5x200_df, self.reports_root, self.images_root,
            transform=self.val_transforms, use_labels=self.use_labels,
        )

    def train_dataloader(self) -> DataLoader:
        # STRATEGY PATTERN: ask the transform if it requires special collation.
        collate_fn = None
        if hasattr(self.train_transforms, "get_collate_fn"):
            collate_fn = self.train_transforms.get_collate_fn()

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> List[DataLoader]:
        loader_kwargs = dict(
            batch_size=min(self.batch_size, 10),
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        return [
            DataLoader(self.val_ds, **loader_kwargs),
            DataLoader(self.val_5x200_ds, **loader_kwargs),
        ]