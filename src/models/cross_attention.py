from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = None
        if context_attention_mask is not None:
            key_padding_mask = ~context_attention_mask.bool()
        q_norm = self.query_norm(query)
        c_norm = self.context_norm(context)
        attn_out, _ = self.attn(
            q_norm,
            c_norm,
            c_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + self.dropout(attn_out)
        query = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return query


class CategoryCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_classes, queries_per_class, hidden_dim = queries.shape
        x = queries.reshape(batch_size, num_classes * queries_per_class, hidden_dim)
        for layer in self.layers:
            x = layer(x, context, context_attention_mask=context_attention_mask)
        return x.reshape(batch_size, num_classes, queries_per_class, hidden_dim)
