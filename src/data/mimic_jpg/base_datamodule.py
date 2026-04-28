import torch
import pandas as pd
import lightning.pytorch as pl
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from .utils import MIMIC_LABEL_COLUMNS


class BaseMimicDataModule(pl.LightningDataModule, ABC):
    """
    Abstract Base Class for MIMIC-CXR DataModules.
    Strictly handles reproducible subject-level splitting logic.
    """
    def __init__(
        self,
        metadata_csv: str,
        use_labels: bool = False,
        labels_csv: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.metadata_csv = metadata_csv
        self.use_labels = use_labels
        self.labels_csv = labels_csv

        # Placeholders for datasets created in setup()
        self.train_ds = None
        self.val_ds = None
        self.val_5x200_ds = None

    def _get_split_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Helper to load and split the clinical metadata by subject."""
        df = pd.read_csv(self.metadata_csv)

        if self.use_labels and self.labels_csv:
            if not all(c in df.columns for c in MIMIC_LABEL_COLUMNS):
                labels_df = pd.read_csv(self.labels_csv)
                df = df.merge(labels_df, on=["subject_id", "study_id"], how="left")
            df[MIMIC_LABEL_COLUMNS] = df[MIMIC_LABEL_COLUMNS].fillna(-99.0)

        unique_subjects = df["subject_id"].unique()

        # pl.seed_everything() is called in main.py
        perm = torch.randperm(len(unique_subjects))
        unique_subjects = unique_subjects[perm.numpy()]

        split_idx = int(len(unique_subjects) * 0.9)
        train_subjects = unique_subjects[:split_idx]
        val_subjects = unique_subjects[split_idx:]

        train_df = df[df["subject_id"].isin(train_subjects)].reset_index(drop=True)
        val_df = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)

        return train_df, val_df

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    @property
    def target_indices(self) -> torch.Tensor:
       from .utils import MIMIC_LABEL_COLUMNS, MIMIC_5X200_FINDINGS
       return torch.tensor([MIMIC_LABEL_COLUMNS.index(f) for f in MIMIC_5X200_FINDINGS if f in MIMIC_LABEL_COLUMNS])
