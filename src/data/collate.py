from __future__ import annotations

from typing import Any

import torch


def cxr_omission_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "texts": [item["text"] for item in batch],
        "weak_labels": torch.stack([item["weak_labels"] for item in batch], dim=0),
        "omit_labels": torch.stack([item["omit_labels"] for item in batch], dim=0),
        "omit_masks": torch.stack([item["omit_masks"] for item in batch], dim=0),
        "weak_valid_masks": torch.stack(
            [item["weak_valid_masks"] for item in batch], dim=0
        ),
        "report_labels": torch.stack([item["report_labels"] for item in batch], dim=0),
        "omit_pair_weight": torch.stack(
            [item["omit_pair_weight"] for item in batch], dim=0
        ),
        "tail_class_weight": torch.stack(
            [item["tail_class_weight"] for item in batch], dim=0
        ),
        "is_clean": torch.tensor(
            [item["is_clean"] for item in batch], dtype=torch.float32
        ),
        "sample_id": [item["sample_id"] for item in batch],
        "study_id": [item["study_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "split": [item["split"] for item in batch],
        "has_weak_valid_masks": all(
            bool(item["has_weak_valid_masks"]) for item in batch
        ),
        "has_labels": all(bool(item["has_labels"]) for item in batch),
        "metadata": [item["metadata"] for item in batch],
    }
