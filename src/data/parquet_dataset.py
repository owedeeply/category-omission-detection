from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)


CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Fracture",
    "Pulmonary Nodule/Mass",
    "Pleural Effusion",
    "Pneumothorax",
    "Emphysema",
    "Edema",
]

CLASS_SLUGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "fracture",
    "pulmonary_nodule_mass",
    "pleural_effusion",
    "pneumothorax",
    "emphysema",
    "edema",
]

REQUIRED_BASE_COLUMNS = [
    "sample_id",
    "study_id",
    "image_path",
    "perturbed_report_text",
    "split",
]

OPTIONAL_COLUMNS = [
    "clean_text",
    "original_report_text",
    "sample_type",
    "omitted_classes",
    "omitted_target_seed",
    "deleted_evidence_sentences",
    "unmatched_evidence_sentences",
    "pnm_keyword_removed_sentences",
    "has_unmatched_evidence",
    "input_word_count",
    "n_omitted_targets",
]


def label_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{slug}" for slug in CLASS_SLUGS]


WEAK_LABEL_COLUMNS = label_columns("weak_label")
OMIT_LABEL_COLUMNS = label_columns("omit_label")
OMIT_MASK_COLUMNS = label_columns("omit_mask")
WEAK_VALID_MASK_COLUMNS = label_columns("weak_valid_mask")
REPORT_LABEL_COLUMNS = label_columns("report_label")
OMIT_PAIR_WEIGHT_COLUMNS = label_columns("omit_pair_weight")
TAIL_CLASS_WEIGHT_COLUMNS = label_columns("tail_class_weight")
ALL_LABEL_COLUMNS = WEAK_LABEL_COLUMNS + OMIT_LABEL_COLUMNS + OMIT_MASK_COLUMNS


def validate_parquet_schema(df: pd.DataFrame, require_labels: bool = True) -> None:
    required = list(REQUIRED_BASE_COLUMNS)
    if require_labels:
        required += ALL_LABEL_COLUMNS + WEAK_VALID_MASK_COLUMNS + REPORT_LABEL_COLUMNS + OMIT_PAIR_WEIGHT_COLUMNS + TAIL_CLASS_WEIGHT_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Parquet file is missing required columns: {missing}")
    present_weak_valid_masks = [
        column for column in WEAK_VALID_MASK_COLUMNS if column in df.columns
    ]
    if present_weak_valid_masks and len(present_weak_valid_masks) != len(
        WEAK_VALID_MASK_COLUMNS
    ):
        missing_weak_valid = [
            column for column in WEAK_VALID_MASK_COLUMNS if column not in df.columns
        ]
        raise ValueError(
            f"Parquet file has partial weak_valid_mask columns; missing: {missing_weak_valid}"
        )
    present_report_labels = [
        column for column in REPORT_LABEL_COLUMNS if column in df.columns
    ]
    if present_report_labels and len(present_report_labels) != len(
        REPORT_LABEL_COLUMNS
    ):
        missing_report = [
            column for column in REPORT_LABEL_COLUMNS if column not in df.columns
        ]
        raise ValueError(
            f"Parquet file has partial report_label columns; missing: {missing_report}"
        )


def infer_is_clean_series(df: pd.DataFrame) -> pd.Series:
    if "is_clean" in df.columns:
        return pd.to_numeric(df["is_clean"], errors="coerce").fillna(0.0).astype(float)
    if "n_omitted_targets" in df.columns:
        n_omitted = pd.to_numeric(df["n_omitted_targets"], errors="coerce").fillna(0.0)
        return (n_omitted <= 0).astype(float)
    if "sample_type" in df.columns:
        sample_type = df["sample_type"].fillna("").astype(str).str.lower()
        return sample_type.str.startswith("no_omission").astype(float)
    if all(column in df.columns for column in OMIT_LABEL_COLUMNS):
        omit_sum = df[OMIT_LABEL_COLUMNS].sum(axis=1)
        return (omit_sum <= 0).astype(float)
    return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)


def read_metadata(
    parquet_path: str | Path, require_labels: bool = True
) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    validate_parquet_schema(df, require_labels=require_labels)
    if "is_clean" not in df.columns:
        df = df.copy()
        df["is_clean"] = infer_is_clean_series(df)
    return df.reset_index(drop=True)


class CxrOmissionDataset(Dataset):
    """Read user-provided Parquet rows without filtering, resampling, or rewriting data."""

    def __init__(
        self,
        parquet_path: str | Path,
        image_root: str | Path,
        transform: Any | None = None,
        require_labels: bool = True,
        image_path_layout: str = "auto",
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.require_labels = require_labels
        self.image_path_layout = str(image_path_layout or "auto").lower()
        if self.image_path_layout not in {"auto", "split_flat"}:
            raise ValueError("image_path_layout must be 'auto' or 'split_flat'.")
        self.df = read_metadata(self.parquet_path, require_labels=require_labels)
        self.has_weak_valid_masks = all(
            column in self.df.columns for column in WEAK_VALID_MASK_COLUMNS
        )
        self.has_omit_pair_weights = all(
            column in self.df.columns for column in OMIT_PAIR_WEIGHT_COLUMNS
        )
        self.has_tail_class_weights = all(
            column in self.df.columns for column in TAIL_CLASS_WEIGHT_COLUMNS
        )
        self._warn_missing_optional_weight_columns()

    def __len__(self) -> int:
        return len(self.df)

    def _candidate_image_paths(self, image_path: str, split: str) -> list[Path]:
        image_path_str = str(image_path)
        normalized_image_path = image_path_str.replace("\\", "/")
        path = Path(normalized_image_path)
        image_name = Path(normalized_image_path).name
        split_name = str(split)
        split_name_lower = split_name.lower()

        candidates: list[Path] = [
            self.image_root / split_name / image_name,
            self.image_root / split_name_lower / image_name,
        ]
        for candidate_split in ("train", "val", "test"):
            candidates.append(self.image_root / candidate_split / image_name)
        if self.image_path_layout == "split_flat":
            return self._dedupe_paths(candidates)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(self.image_root / path)
        return self._dedupe_paths(candidates)

    @staticmethod
    def _dedupe_paths(candidates: list[Path]) -> list[Path]:
        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                deduped.append(candidate)
                seen.add(key)
        return deduped

    def _resolve_image_path(self, image_path: str, split: str) -> Path:
        candidates = self._candidate_image_paths(image_path, split)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _read_image(self, image_path: str, split: str) -> Image.Image:
        resolved = self._resolve_image_path(image_path, split)
        if not resolved.exists():
            candidates = "\n".join(
                str(path) for path in self._candidate_image_paths(image_path, split)
            )
            raise FileNotFoundError(
                f"Could not locate image for image_path={image_path!r}, split={split!r}. "
                f"Tried:\n{candidates}"
            )
        with Image.open(resolved) as img:
            return img.copy()

    def _read_labels(
        self,
        row: pd.Series,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not self.require_labels:
            zeros = torch.zeros(len(CLASSES), dtype=torch.float32)
            ones = torch.ones(len(CLASSES), dtype=torch.float32)
            return (
                zeros,
                zeros.clone(),
                zeros.clone(),
                zeros.clone(),
                zeros.clone(),
                ones,
                ones.clone(),
            )
        weak = row[WEAK_LABEL_COLUMNS].to_numpy(dtype=np.float32)
        omit = row[OMIT_LABEL_COLUMNS].to_numpy(dtype=np.float32)
        mask = row[OMIT_MASK_COLUMNS].to_numpy(dtype=np.float32)
        if self.has_weak_valid_masks:
            weak_valid_mask = row[WEAK_VALID_MASK_COLUMNS].to_numpy(dtype=np.float32)
        else:
            weak_valid_mask = np.ones(len(CLASSES), dtype=np.float32)
        if all(column in row.index for column in REPORT_LABEL_COLUMNS):
            report = row[REPORT_LABEL_COLUMNS].to_numpy(dtype=np.float32)
        else:
            report = np.zeros(len(CLASSES), dtype=np.float32)
        omit_pair_weight = self._read_optional_weight_vector(
            row, OMIT_PAIR_WEIGHT_COLUMNS
        )
        tail_class_weight = self._read_optional_weight_vector(
            row, TAIL_CLASS_WEIGHT_COLUMNS
        )
        return (
            torch.from_numpy(weak),
            torch.from_numpy(omit),
            torch.from_numpy(mask),
            torch.from_numpy(weak_valid_mask),
            torch.from_numpy(report),
            torch.from_numpy(omit_pair_weight),
            torch.from_numpy(tail_class_weight),
        )

    def _read_optional_weight_vector(
        self, row: pd.Series, columns: list[str]
    ) -> np.ndarray:
        values = [
            float(row[column]) if column in row.index else 1.0 for column in columns
        ]
        return np.asarray(values, dtype=np.float32)

    def _warn_missing_optional_weight_columns(self) -> None:
        missing_pair = [
            column
            for column in OMIT_PAIR_WEIGHT_COLUMNS
            if column not in self.df.columns
        ]
        missing_tail = [
            column
            for column in TAIL_CLASS_WEIGHT_COLUMNS
            if column not in self.df.columns
        ]
        if missing_pair:
            LOGGER.warning(
                "Parquet %s is missing omit_pair_weight columns; missing columns will default to 1.0: %s",
                self.parquet_path,
                missing_pair,
            )
        if missing_tail:
            LOGGER.warning(
                "Parquet %s is missing tail_class_weight columns; missing columns will default to 1.0: %s",
                self.parquet_path,
                missing_tail,
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image_path = str(row["image_path"])
        split = str(row["split"])
        image = self._read_image(image_path, split)
        if self.transform is not None:
            image = self.transform(image)

        (
            weak_labels,
            omit_labels,
            omit_masks,
            weak_valid_masks,
            report_labels,
            omit_pair_weight,
            tail_class_weight,
        ) = self._read_labels(row)
        metadata: dict[str, Any] = {}
        for column in OPTIONAL_COLUMNS:
            if column in row:
                metadata[column] = row[column]

        return {
            "image": image,
            "text": str(row["perturbed_report_text"]),
            "sample_id": row["sample_id"],
            "study_id": row["study_id"],
            "image_path": image_path,
            "is_clean": float(row["is_clean"]),
            "split": split,
            "weak_labels": weak_labels,
            "omit_labels": omit_labels,
            "omit_masks": omit_masks,
            "weak_valid_masks": weak_valid_masks,
            "report_labels": report_labels,
            "omit_pair_weight": omit_pair_weight,
            "tail_class_weight": tail_class_weight,
            "has_weak_valid_masks": self.has_weak_valid_masks,
            "has_labels": self.require_labels,
            "metadata": metadata,
        }
