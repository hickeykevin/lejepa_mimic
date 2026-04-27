import torch
import pandas as pd
import numpy as np
import warnings
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, Dict, List, Optional

from .utils import extract_findings_and_impression, get_best_caption, MIMIC_LABEL_COLUMNS


class RawImageCollate:
    """
    Collate function that keeps designated keys as Python lists, without stacking.
    Used for variable-sized tensors (e.g. raw images for DINOLeJEPA GPU-side cropping).
    """
    def __init__(self, list_keys: Optional[List[str]] = None):
        self.list_keys = list_keys or ["image"]

    def __call__(self, batch):
        from torch.utils.data._utils.collate import default_collate

        # Pop keys that must remain as lists (variable-size tensors)
        list_contents = {}
        for key in self.list_keys:
            if key in batch[0]:
                list_contents[key] = [item.pop(key) for item in batch]

        collated = default_collate(batch)

        # Reattach the list-based keys
        for key, val in list_contents.items():
            collated[key] = val

        return collated


class MimicCxrDataset(Dataset):
    """
    Image-level dataset for MIMIC-CXR.
    Each sample is one DICOM image with its associated report text.
    """
    LABEL_COLUMNS = MIMIC_LABEL_COLUMNS

    def __init__(
        self,
        df: pd.DataFrame,
        reports_root: str,
        images_root: str,
        transform: Optional[Any] = None,
        use_labels: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.reports_root = str(reports_root)
        self.images_root = str(images_root)
        self.transform = transform
        self.use_labels = use_labels

        if self.use_labels:
            self.label_cols = [c for c in self.LABEL_COLUMNS if c in self.df.columns]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        subject_id = str(row["subject_id"])
        study_id = str(row["study_id"])
        dicom_id = str(row["dicom_id"])
        subj_prefix = f"p{subject_id[:2]}"

        # 1. Text Loading
        report_path = f"{self.reports_root}/files/{subj_prefix}/p{subject_id}/s{study_id}.txt"
        findings, impression, text = "", "", ""
        try:
            with open(report_path, "r", errors="ignore") as f:
                content = f.read()
            if content:
                findings, impression = extract_findings_and_impression(content)
                text = f"{findings} {impression}".strip()
        except FileNotFoundError:
            pass

        # 2. Image Loading
        image_path = f"{self.images_root}/files/{subj_prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        try:
            img = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            try:
                img = Image.open(image_path.replace(".jpg", ".png")).convert("RGB")
            except Exception:
                raise FileNotFoundError(f"Could not load image: {image_path}")

        # 3. Output Assembly
        caption = get_best_caption(findings, impression)
        if not caption:
            caption = text or "no report available"

        outputs = {
            "image":      self.transform(img) if self.transform else img,
            "caption":    caption,
            "findings":   findings,
            "impression": impression,
            "study_id":   study_id,
        }

        if self.use_labels:
            labels = row[self.label_cols].values.astype(np.float32)
            outputs["labels"] = torch.from_numpy(labels)

        return outputs


class StudyMimicCxrDataset(Dataset):
    """
    Study-level dataset for MIMIC-CXR.
    Each sample represents one study, returning all associated images and the report.
    """
    LABEL_COLUMNS = MIMIC_LABEL_COLUMNS

    def __init__(
        self,
        df: pd.DataFrame,
        reports_root: str,
        images_root: str,
        transform: Optional[Any] = None,
        use_labels: bool = False,
    ):
        self.df = df
        self.reports_root = str(reports_root)
        self.images_root = str(images_root)
        self.transform = transform
        self.use_labels = use_labels

        self.study_groups = self.df.groupby("study_id")
        self.study_ids = list(self.study_groups.groups.keys())
        self.study_to_subject = (
            self.df.drop_duplicates("study_id")
            .set_index("study_id")["subject_id"]
            .to_dict()
        )

        if self.use_labels:
            self.label_cols = [c for c in self.LABEL_COLUMNS if c in self.df.columns]

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        study_id = self.study_ids[idx]
        subject_id = str(self.study_to_subject[study_id])
        study_id_str = str(study_id)
        subj_prefix = f"p{subject_id[:2]}"

        # 1. Text Loading
        findings, impression, text = "", "", ""
        report_path = f"{self.reports_root}/files/{subj_prefix}/p{subject_id}/s{study_id_str}.txt"
        try:
            with open(report_path, "r", errors="ignore") as f:
                content = f.read()
            if content:
                findings, impression = extract_findings_and_impression(content)
                text = f"{findings} {impression}".strip()
        except FileNotFoundError:
            pass

        caption = get_best_caption(findings, impression)
        if not caption:
            caption = text or "no report available"

        # 2. Images Loading
        group = self.study_groups.get_group(study_id)
        images = []
        for _, row in group.iterrows():
            dicom_id = str(row["dicom_id"])
            image_path = f"{self.images_root}/files/{subj_prefix}/p{subject_id}/s{study_id_str}/{dicom_id}.jpg"
            try:
                img = Image.open(image_path).convert("RGB")
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                try:
                    img = Image.open(image_path.replace(".jpg", ".png")).convert("RGB")
                except Exception:
                    raise FileNotFoundError(f"Could not load image: {image_path}")

            if self.transform:
                img = self.transform(img)
            images.append(img)

        outputs = {
            "images":     images,
            "caption":    caption,
            "raw_text":   text,
            "findings":   findings,
            "impression": impression,
            "study_id":   study_id_str,
            "subject_id": subject_id,
        }

        if self.use_labels:
            labels = group[self.label_cols].iloc[0].values.astype(np.float32)
            outputs["labels"] = torch.from_numpy(labels)

        return outputs
