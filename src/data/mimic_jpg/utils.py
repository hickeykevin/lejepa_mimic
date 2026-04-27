import re
from typing import Tuple, List

# Single source of truth for CheXpert label columns used across MIMIC-CXR datasets.
MIMIC_LABEL_COLUMNS: List[str] = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# The 5 diseases used for the balanced 5x200 validation subset.
MIMIC_5X200_FINDINGS: List[str] = [
    "Atelectasis", "Cardiomegaly", "Edema", "Pleural Other", "Pleural Effusion"
]


def radtext_clean(text: str) -> str:
    """Performs basic cleaning of radiology report text."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_findings_and_impression(report_text: str) -> Tuple[str, str]:
    """Extracts FINDINGS and IMPRESSION sections from a MIMIC-CXR report."""
    findings, impression = "", ""
    findings_match = re.search(r"FINDINGS:\s*(.*?)(?:\n[A-Z ]+:|\Z)", report_text, re.DOTALL | re.IGNORECASE)
    impression_match = re.search(r"IMPRESSION:\s*(.*?)(?:\n[A-Z ]+:|\Z)", report_text, re.DOTALL | re.IGNORECASE)
    
    if findings_match:
        findings = findings_match.group(1).strip()
    if impression_match:
        impression = impression_match.group(1).strip()
    
    return radtext_clean(findings), radtext_clean(impression)

def get_best_caption(findings: str, impression: str) -> str:
    """Heuristic to select the most descriptive text as a caption."""
    if len(findings.split()) >= 5:
        return findings
    if impression:
        return impression
    return findings
