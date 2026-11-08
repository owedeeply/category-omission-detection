from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F




def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    return (base * ((1.0 - pt) ** gamma)).mean()


def masked_asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma_pos: float = 0.0,
    gamma_neg: float = 2.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    raw_loss = asymmetric_loss_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        gamma_pos=gamma_pos,
        gamma_neg=gamma_neg,
        eps=eps,
    )
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    loss = raw_loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma_pos: float | torch.Tensor = 0.0,
    gamma_neg: float | torch.Tensor = 2.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    gamma_pos = torch.as_tensor(gamma_pos, device=logits.device, dtype=logits.dtype)
    gamma_neg = torch.as_tensor(gamma_neg, device=logits.device, dtype=logits.dtype)

    pos_loss = -targets * torch.log(probs.clamp_min(eps)) * ((1.0 - probs) ** gamma_pos)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
        pos_loss = pos_loss * pos_weight
    neg_loss = (
        -(1.0 - targets) * torch.log((1.0 - probs).clamp_min(eps)) * (probs**gamma_neg)
    )
    return pos_loss + neg_loss
