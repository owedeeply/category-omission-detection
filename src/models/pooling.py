from __future__ import annotations

import torch
from torch import nn


class QueryAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, query_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self.scorer(x).squeeze(-1)
        if query_mask is not None:
            if query_mask.shape != logits.shape:
                raise ValueError(
                    f"query_mask shape must match query logits, got {query_mask.shape} and {logits.shape}."
                )
            logits = logits.masked_fill(~query_mask.bool(), -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        if query_mask is not None:
            weights = weights * query_mask.to(
                device=weights.device, dtype=weights.dtype
            )
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return (x * weights.unsqueeze(-1)).sum(dim=2)
