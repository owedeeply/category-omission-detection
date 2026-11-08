from __future__ import annotations

import math

import pandas as pd
import torch

from src.data.parquet_dataset import CLASS_SLUGS


def _sqrt_clipped(pos: int, neg: int, clip: tuple[float, float]) -> float:
    if pos <= 0:
        return 1.0
    value = math.sqrt(max(neg, 0) / max(pos, 1))
    return float(min(max(value, clip[0]), clip[1]))


def compute_omit_pos_weight(
    train_df: pd.DataFrame,
    clip: tuple[float, float] = (1.0, 5.0),
) -> torch.Tensor:
    weights: list[float] = []
    for slug in CLASS_SLUGS:
        label = train_df[f"omit_label_{slug}"].astype(int)
        mask = train_df[f"omit_mask_{slug}"].astype(int) == 1
        valid = label[mask]
        pos = int((valid == 1).sum())
        neg = int((valid == 0).sum())
        weights.append(_sqrt_clipped(pos, neg, clip))
    return torch.tensor(weights, dtype=torch.float32)


def compute_weak_pos_weight(
    train_df: pd.DataFrame,
    clip: tuple[float, float] = (1.0, 3.0),
    mask_with_omit_mask: bool = False,
    mask_column_prefix: str | None = None,
) -> torch.Tensor:
    weights: list[float] = []
    for slug in CLASS_SLUGS:
        label = train_df[f"weak_label_{slug}"].astype(int)
        if mask_column_prefix:
            mask_column = f"{mask_column_prefix}_{slug}"
            if mask_column not in train_df.columns:
                raise ValueError(f"Missing weak pos-weight mask column: {mask_column}")
            mask = train_df[mask_column].astype(int) == 1
            label = label[mask]
        elif mask_with_omit_mask:
            mask = train_df[f"omit_mask_{slug}"].astype(int) == 1
            label = label[mask]
        pos = int((label == 1).sum())
        neg = int((label == 0).sum())
        weights.append(_sqrt_clipped(pos, neg, clip))
    return torch.tensor(weights, dtype=torch.float32)
