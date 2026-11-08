from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.models.cross_attention import CategoryCrossAttention
from src.models.heads import PrototypeCoverageGatedOmissionHead, WeakVisualHead
from src.models.image_encoder import CXRRepairCLIPVisualEncoder, SyntheticImageEncoder
from src.models.pooling import QueryAttentionPooling
from src.models.query_bank import (
    CategoryQueryBank,
    build_query_bank_from_text_encoder,
    build_synthetic_query_bank,
)
from src.models.text_encoder import CXRTextEncoder, SyntheticTextEncoder


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _get_query_init_cfg(model_cfg: dict[str, Any]) -> tuple[str, bool, bool, bool]:
    nested = model_cfg.get("query_init", {})
    if nested is None:
        nested = {}
    if nested and (not isinstance(nested, dict)):
        raise ValueError(
            "model.query_init must be a mapping with path/trainable/use_projection keys."
        )
    nested_cfg = nested if isinstance(nested, dict) else {}
    query_init_path = nested_cfg.get("path", model_cfg.get("query_init_path", ""))
    query_init_trainable = nested_cfg.get(
        "trainable", model_cfg.get("query_init_trainable", True)
    )
    use_projection = nested_cfg.get(
        "use_projection", model_cfg.get("query_init_use_projection", False)
    )
    required = nested_cfg.get("required", model_cfg.get("query_init_required", False))
    return (
        str(query_init_path or ""),
        _as_bool(query_init_trainable, default=True),
        _as_bool(use_projection, default=False),
        _as_bool(required, default=False),
    )


def _get_query_init_random_reference_max_queries(
    model_cfg: dict[str, Any]
) -> int | None:
    nested = model_cfg.get("query_init", {})
    if nested is None:
        return None
    if not isinstance(nested, dict):
        raise ValueError("model.query_init must be a mapping.")
    value = nested.get("random_reference_max_queries")
    return None if value is None else int(value)


def _get_query_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    query_cfg = model_cfg.get("query", {})
    if query_cfg is None:
        query_cfg = {}
    if query_cfg and (not isinstance(query_cfg, dict)):
        raise ValueError("model.query must be a mapping.")
    return query_cfg if isinstance(query_cfg, dict) else {}


def _get_query_bank_type(model_cfg: dict[str, Any]) -> str:
    query_cfg = _get_query_cfg(model_cfg)
    return str(
        model_cfg.get("query_bank_type", query_cfg.get("type", "class_queries"))
    ).lower()


def _get_image_residual_cfg(
    model_cfg: dict[str, Any]
) -> tuple[bool, float, float, bool]:
    query_cfg = _get_query_cfg(model_cfg)
    residual_cfg = query_cfg.get("image_residual", {}) or {}
    if not isinstance(residual_cfg, dict):
        raise ValueError("model.query.image_residual must be a mapping when provided.")
    return (
        _as_bool(residual_cfg.get("enable", False), default=False),
        float(residual_cfg.get("init_std", 0.02)),
        float(residual_cfg.get("scale", 1.0)),
        _as_bool(residual_cfg.get("trainable", True), default=True),
    )


def _get_queries_per_class_by_class(model_cfg: dict[str, Any]) -> dict[str, int]:
    value = model_cfg.get("queries_per_class_by_class", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            "model.queries_per_class_by_class must be a mapping from class slug to integer count."
        )
    return {str(key): int(count) for key, count in value.items()}


def _masked_query_mean(x: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    mask = query_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    return (x * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)


class OmissionDetectionModel(nn.Module):

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        query_bank: CategoryQueryBank,
        cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        hidden_dim = int(model_cfg.get("hidden_dim", 768))
        cross_cfg = model_cfg.get("cross_attention", {})
        dropout = float(cross_cfg.get("dropout", 0.1))
        self.omission_head_type = str(
            model_cfg.get("omission_head_type", "prototype_coverage_gated")
        ).lower()
        prototype_pooling = str(model_cfg.get("prototype_pooling", "attention")).lower()
        coverage_pooling = str(model_cfg.get("coverage_pooling", "attention")).lower()
        coverage_gate_cfg = model_cfg.get("coverage_gate", {}) or {}
        if not isinstance(coverage_gate_cfg, dict):
            raise ValueError("model.coverage_gate must be a mapping when provided.")
        use_coverage_gate = _as_bool(
            coverage_gate_cfg.get("enable", True), default=True
        )
        weak_visual_support_cfg = coverage_gate_cfg.get("weak_visual_support", {}) or {}
        if not isinstance(weak_visual_support_cfg, dict):
            raise ValueError(
                "model.coverage_gate.weak_visual_support must be a mapping when provided."
            )
        use_weak_visual_support = _as_bool(
            weak_visual_support_cfg.get("enable", False), default=False
        )
        weak_visual_support_floor = float(weak_visual_support_cfg.get("floor", 0.1))
        detach_weak_visual_support = _as_bool(
            weak_visual_support_cfg.get("detach", True), default=True
        )
        weak_head_cfg = model_cfg.get("weak_head", {}) or {}
        if not isinstance(weak_head_cfg, dict):
            raise ValueError("model.weak_head must be a mapping when provided.")
        weak_head_pooling = str(weak_head_cfg.get("pooling", "feature_mean")).lower()
        weak_head_lse_sharpness = float(weak_head_cfg.get("lse_sharpness", 1.0))
        weak_head_normalize = _as_bool(
            weak_head_cfg.get("normalize_by_valid_queries", True), default=True
        )
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.query_bank = query_bank
        self.hidden_dim = hidden_dim
        self.num_classes = int(model_cfg.get("num_classes", 9))
        self.visual_proj = nn.Linear(int(image_encoder.output_dim), hidden_dim)
        self.text_proj = nn.Linear(int(text_encoder.output_dim), hidden_dim)
        self.visual_cross_attention = CategoryCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=int(cross_cfg.get("num_heads", 12)),
            ffn_dim=int(cross_cfg.get("ffn_dim", 3072)),
            dropout=dropout,
            num_layers=int(cross_cfg.get("num_layers", 3)),
        )
        self.text_cross_attention = CategoryCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=int(cross_cfg.get("num_heads", 12)),
            ffn_dim=int(cross_cfg.get("ffn_dim", 3072)),
            dropout=dropout,
            num_layers=int(cross_cfg.get("num_layers", 3)),
        )
        self.visual_pool = QueryAttentionPooling(hidden_dim, dropout=dropout)
        self.text_pool = QueryAttentionPooling(hidden_dim, dropout=dropout)
        self.weak_head = WeakVisualHead(
            hidden_dim,
            dropout=dropout,
            pooling=weak_head_pooling,
            lse_sharpness=weak_head_lse_sharpness,
            normalize_by_valid_queries=weak_head_normalize,
        )
        self.omit_head = PrototypeCoverageGatedOmissionHead(
            hidden_dim,
            dropout=dropout,
            prototype_pooling=prototype_pooling,
            coverage_pooling=coverage_pooling,
            use_coverage_gate=use_coverage_gate,
            use_weak_visual_support=use_weak_visual_support,
            weak_visual_support_floor=weak_visual_support_floor,
            detach_weak_visual_support=detach_weak_visual_support,
        )

    def encode_query_features(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run both backbones and cross-attention branches once.

        Retrieve the per-category visual and textual representations.
        """
        images = batch["images"]
        image_out = self.image_encoder(images)
        visual_tokens = (
            image_out["patch_tokens"] if isinstance(image_out, dict) else image_out
        )
        visual_tokens = self.visual_proj(visual_tokens)
        text_tokens, attention_mask = self.text_encoder(
            texts=batch.get("texts"),
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
        )
        text_tokens = self.text_proj(text_tokens)
        q_img, q_txt, query_mask = self.query_bank(
            batch_size=images.shape[0], device=visual_tokens.device
        )
        visual_ck = self.visual_cross_attention(
            q_img, visual_tokens, context_attention_mask=None
        )
        text_ck = self.text_cross_attention(
            q_txt, text_tokens, context_attention_mask=attention_mask
        )
        return {"visual_ck": visual_ck, "text_ck": text_ck, "query_mask": query_mask}

    def forward_from_query_features(
        self, visual_ck: torch.Tensor, text_ck: torch.Tensor, query_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Replay prediction heads from cached per-query representations."""
        if visual_ck.shape != text_ck.shape:
            raise ValueError(
                f"visual_ck and text_ck must have the same shape, got {tuple(visual_ck.shape)} and {tuple(text_ck.shape)}."
            )
        if query_mask.shape != visual_ck.shape[:3]:
            raise ValueError(
                f"query_mask must match the first three query-feature dimensions, got {tuple(query_mask.shape)} and {tuple(visual_ck.shape[:3])}."
            )
        query_mask = query_mask.to(device=visual_ck.device, dtype=torch.bool)
        if bool((query_mask.sum(dim=2) == 0).any()):
            raise ValueError(
                "Every class must retain at least one valid query during cached replay."
            )
        visual_c_for_weak = _masked_query_mean(visual_ck, query_mask)
        weak_logits = self.weak_head(visual_c_for_weak)
        visual_c = visual_c_for_weak
        text_c = _masked_query_mean(text_ck, query_mask)
        omit_logits, omit_debug = self.omit_head(
            visual_ck, text_ck, weak_logits, query_mask=query_mask
        )
        omit_logits_ck = omit_debug["omit_logits_ck"]
        poscov_logit_ck = omit_debug["poscov_logit_ck"]
        poscov_gate = omit_debug["poscov_gate"]
        coverage_logit_c = omit_debug["coverage_logit_c"]
        prototype_pool_weight = omit_debug["prototype_pool_weight"]
        weak_visual_support = omit_debug["weak_visual_support"]
        return {
            "weak_logits": weak_logits,
            "omit_logits": omit_logits,
            "weak_prob": torch.sigmoid(weak_logits),
            "omit_prob": torch.sigmoid(omit_logits),
            "visual_per_class": visual_c,
            "text_per_class": text_c,
            "omit_logits_ck": omit_logits_ck,
            "poscov_logit_ck": poscov_logit_ck,
            "poscov_gate": poscov_gate,
            "coverage_logit_c": coverage_logit_c,
            "coverage_prob": torch.sigmoid(coverage_logit_c),
            "prototype_pool_weight": prototype_pool_weight,
            "weak_visual_support": weak_visual_support,
            "query_mask": query_mask,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = self.encode_query_features(batch)
        return self.forward_from_query_features(
            visual_ck=features["visual_ck"],
            text_ck=features["text_ck"],
            query_mask=features["query_mask"],
        )


def apply_finetune_policy(model: nn.Module, cfg: dict[str, Any]) -> None:
    finetune = cfg.get("finetune", {})
    if not finetune.get("enable", False):
        return
    image_encoder = getattr(model, "image_encoder", None)
    text_encoder = getattr(model, "text_encoder", None)
    if hasattr(image_encoder, "unfreeze_last_blocks"):
        image_encoder.unfreeze_last_blocks(
            int(finetune.get("unfreeze_image_last_blocks", 0))
        )
    if hasattr(text_encoder, "unfreeze_last_layers"):
        text_encoder.unfreeze_last_layers(
            int(finetune.get("unfreeze_text_last_layers", 0))
        )


def build_model_from_config(
    cfg: dict[str, Any], require_checkpoint: bool = True
) -> nn.Module:
    model_cfg = cfg["model"]
    cross_cfg = model_cfg.get("cross_attention", {})
    query_cfg = _get_query_cfg(model_cfg)
    query_bank_type = _get_query_bank_type(model_cfg)
    (
        image_residual_enable,
        image_residual_init_std,
        image_residual_scale,
        image_residual_trainable,
    ) = _get_image_residual_cfg(model_cfg)
    queries_per_class_by_class = _get_queries_per_class_by_class(model_cfg)
    query_init_path, query_init_trainable, use_projection, query_init_required = (
        _get_query_init_cfg(model_cfg)
    )
    query_init_random_reference_max_queries = (
        _get_query_init_random_reference_max_queries(model_cfg)
    )
    image_encoder = CXRRepairCLIPVisualEncoder(
        model_cfg["image_encoder"], require_checkpoint=require_checkpoint
    )
    text_encoder = CXRTextEncoder(model_cfg["text_encoder"])
    query_bank = build_query_bank_from_text_encoder(
        text_encoder=text_encoder,
        hidden_dim=int(model_cfg.get("hidden_dim", 768)),
        queries_per_class=int(model_cfg.get("queries_per_class", 3)),
        queries_per_class_by_class=queries_per_class_by_class,
        query_init_path=query_init_path,
        query_init_trainable=query_init_trainable,
        query_init_required=query_init_required,
        query_init_random_seed=int(cfg["train"].get("seed", 42)),
        query_init_random_reference_max_queries=query_init_random_reference_max_queries,
        image_residual_enable=image_residual_enable,
        image_residual_init_std=image_residual_init_std,
        image_residual_scale=image_residual_scale,
        image_residual_trainable=image_residual_trainable,
        use_projection=use_projection,
        use_modality_adaptive=_as_bool(
            query_cfg.get("use_modality_adaptive", False), default=False
        ),
        dropout=float(query_cfg.get("dropout", cross_cfg.get("dropout", 0.1))),
        query_bank_type=query_bank_type,
    )
    model = OmissionDetectionModel(image_encoder, text_encoder, query_bank, cfg)
    apply_finetune_policy(model, cfg)
    return model


def build_synthetic_model(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    image_cfg = model_cfg["image_encoder"]
    text_cfg = model_cfg["text_encoder"]
    cross_cfg = model_cfg.get("cross_attention", {})
    query_cfg = _get_query_cfg(model_cfg)
    query_bank_type = _get_query_bank_type(model_cfg)
    (
        image_residual_enable,
        image_residual_init_std,
        image_residual_scale,
        image_residual_trainable,
    ) = _get_image_residual_cfg(model_cfg)
    queries_per_class_by_class = _get_queries_per_class_by_class(model_cfg)
    hidden_dim = int(model_cfg.get("hidden_dim", 768))
    query_init_path, query_init_trainable, use_projection, query_init_required = (
        _get_query_init_cfg(model_cfg)
    )
    query_init_random_reference_max_queries = (
        _get_query_init_random_reference_max_queries(model_cfg)
    )
    image_encoder = SyntheticImageEncoder(
        image_size=int(image_cfg.get("image_size", 512)), patch_size=32, output_dim=768
    )
    text_encoder = SyntheticTextEncoder(
        output_dim=int(text_cfg.get("synthetic_hidden_size", 768)),
        max_length=int(text_cfg.get("max_length", 256)),
    )
    query_bank = build_synthetic_query_bank(
        text_dim=int(text_encoder.output_dim),
        hidden_dim=hidden_dim,
        queries_per_class=int(model_cfg.get("queries_per_class", 3)),
        queries_per_class_by_class=queries_per_class_by_class,
        query_init_path=query_init_path,
        query_init_trainable=query_init_trainable,
        query_init_required=query_init_required,
        query_init_random_seed=int(cfg["train"].get("seed", 42)),
        query_init_random_reference_max_queries=query_init_random_reference_max_queries,
        image_residual_enable=image_residual_enable,
        image_residual_init_std=image_residual_init_std,
        image_residual_scale=image_residual_scale,
        image_residual_trainable=image_residual_trainable,
        use_projection=use_projection,
        use_modality_adaptive=_as_bool(
            query_cfg.get("use_modality_adaptive", False), default=False
        ),
        dropout=float(query_cfg.get("dropout", cross_cfg.get("dropout", 0.1))),
        query_bank_type=query_bank_type,
    )
    return OmissionDetectionModel(image_encoder, text_encoder, query_bank, cfg)
