from __future__ import annotations
import torch
from torch import nn


class WeakVisualHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        pooling: str = "feature_mean",
        lse_sharpness: float = 1.0,
        normalize_by_valid_queries: bool = True,
    ) -> None:
        super().__init__()
        self.pooling = str(pooling).lower()
        self.lse_sharpness = float(lse_sharpness)
        self.normalize_by_valid_queries = bool(normalize_by_valid_queries)
        if self.pooling not in {"feature_mean", "prototype_logsumexp"}:
            raise ValueError(
                f"WeakVisualHead pooling must be 'feature_mean' or 'prototype_logsumexp', got {pooling!r}."
            )
        if self.lse_sharpness <= 0.0:
            raise ValueError(
                f"WeakVisualHead lse_sharpness must be positive, got {lse_sharpness}."
            )
        if self.pooling == "prototype_logsumexp" and (
            not self.normalize_by_valid_queries
        ):
            raise ValueError(
                "prototype_logsumexp requires normalize_by_valid_queries=True to avoid bias toward classes with more valid queries."
            )
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, visual: torch.Tensor, query_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if visual.ndim != 3:
            raise ValueError(
                f"feature_mean WeakVisualHead expects [B,C,D] features, got {tuple(visual.shape)}."
            )
        return self.net(visual).squeeze(-1)


class PrototypeCoverageGatedOmissionHead(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        prototype_pooling: str = "attention",
        coverage_pooling: str = "attention",
        use_coverage_gate: bool = True,
        use_weak_visual_support: bool = False,
        weak_visual_support_floor: float = 0.1,
        detach_weak_visual_support: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_coverage_gate = bool(use_coverage_gate)
        self.use_weak_visual_support = bool(use_weak_visual_support)
        self.weak_visual_support_floor = float(weak_visual_support_floor)
        self.detach_weak_visual_support = bool(detach_weak_visual_support)
        if self.use_weak_visual_support and (not self.use_coverage_gate):
            raise ValueError("weak visual support requires use_coverage_gate=True.")
        if not 0.0 <= self.weak_visual_support_floor <= 1.0:
            raise ValueError(
                f"weak_visual_support_floor must be between 0 and 1, got {self.weak_visual_support_floor}."
            )
        self.prototype_pooling = prototype_pooling.lower()
        if self.prototype_pooling not in {"mean", "attention"}:
            raise ValueError(f"Unsupported prototype_pooling={prototype_pooling!r}.")
        self.coverage_pooling = coverage_pooling.lower()
        if self.coverage_pooling not in {"attention", "logsumexp"}:
            raise ValueError(f"Unsupported coverage_pooling={coverage_pooling!r}.")
        gate_mid_dim = max(hidden_dim // 2, 128)
        self.poscov_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, gate_mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_mid_dim, 1),
        )
        omit_in_dim = hidden_dim * (5 if self.use_coverage_gate else 4) + 1
        omit_mid_dim = max(hidden_dim + hidden_dim // 2, hidden_dim)
        omit_small_dim = max(hidden_dim // 2, 128)
        self.omit_mlp = nn.Sequential(
            nn.LayerNorm(omit_in_dim),
            nn.Linear(omit_in_dim, omit_mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(omit_mid_dim, omit_small_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(omit_small_dim, 1),
        )
        pool_in_dim = hidden_dim * 3
        self.prototype_pool_mlp = nn.Sequential(
            nn.LayerNorm(pool_in_dim),
            nn.Linear(pool_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        visual_ck: torch.Tensor,
        text_ck: torch.Tensor,
        weak_logits: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if visual_ck.shape != text_ck.shape:
            raise ValueError(
                f"visual_ck and text_ck must have the same shape, got {visual_ck.shape} and {text_ck.shape}."
            )
        if visual_ck.ndim != 4:
            raise ValueError(
                f"visual_ck must have shape [B, C, K, D], got {visual_ck.shape}."
            )
        if query_mask is None:
            query_mask = torch.ones(
                visual_ck.shape[:3], device=visual_ck.device, dtype=torch.bool
            )
        else:
            if query_mask.shape != visual_ck.shape[:3]:
                raise ValueError(
                    f"query_mask must have shape [B, C, K], got {query_mask.shape}; expected {visual_ck.shape[:3]}."
                )
            query_mask = query_mask.to(device=visual_ck.device, dtype=torch.bool)
        queries_per_class = int(visual_ck.shape[2])
        p_weak_c = torch.sigmoid(weak_logits)
        p_weak = p_weak_c.unsqueeze(-1).unsqueeze(-1)
        p_weak = p_weak.expand(-1, -1, queries_per_class, -1)
        weak_visual_support_c = torch.ones_like(p_weak_c)
        weak_visual_support = weak_visual_support_c.unsqueeze(-1).unsqueeze(-1)
        poscov_logit_ck = self.poscov_gate(text_ck).squeeze(-1)
        poscov_gate = torch.sigmoid(poscov_logit_ck).unsqueeze(-1)
        poscov_gate_debug = poscov_gate.squeeze(-1)
        missing_ck = visual_ck * (1.0 - poscov_gate) * weak_visual_support
        omit_input_parts = [
            visual_ck,
            text_ck,
            missing_ck,
            visual_ck - text_ck,
            visual_ck * text_ck,
            p_weak,
        ]
        pool_input_ck = torch.cat([visual_ck, text_ck, missing_ck], dim=-1)
        omit_input_ck = torch.cat(omit_input_parts, dim=-1)
        omit_logits_ck = self.omit_mlp(omit_input_ck).squeeze(-1)
        pool_logits = self.prototype_pool_mlp(pool_input_ck).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~query_mask, -10000.0)
        prototype_pool_weight = torch.softmax(pool_logits, dim=2)
        prototype_pool_weight = prototype_pool_weight * query_mask.to(
            dtype=prototype_pool_weight.dtype
        )
        prototype_pool_weight = prototype_pool_weight / prototype_pool_weight.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-06)
        omit_logits = (prototype_pool_weight * omit_logits_ck).sum(dim=2)
        coverage_logit_c = (prototype_pool_weight * poscov_logit_ck).sum(dim=2)
        omit_logits_ck = omit_logits_ck.masked_fill(~query_mask, 0.0)
        poscov_logit_ck = poscov_logit_ck.masked_fill(~query_mask, 0.0)
        return (
            omit_logits,
            {
                "omit_logits_ck": omit_logits_ck,
                "poscov_logit_ck": poscov_logit_ck,
                "poscov_gate": poscov_gate_debug.masked_fill(~query_mask, 0.0),
                "coverage_logit_c": coverage_logit_c,
                "prototype_pool_weight": prototype_pool_weight,
                "weak_visual_support": weak_visual_support_c,
            },
        )
