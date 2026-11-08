from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    raw_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    masked_loss = raw_loss * mask
    return masked_loss.sum() / mask.sum().clamp_min(1.0)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="mean",
        pos_weight=pos_weight,
    )
