from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.parquet_dataset import CLASSES, CLASS_SLUGS, infer_is_clean_series
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _format_count(count: int) -> str:
    return f"{count:,}"


def count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    frozen = total - trainable
    return total, trainable, frozen


def log_parameter_summary(model: torch.nn.Module, cfg: dict[str, Any]) -> None:
    total, trainable, frozen = count_parameters(model)
    LOGGER.info(
        "Parameter summary | total=%s | trainable=%s | frozen=%s",
        _format_count(total),
        _format_count(trainable),
        _format_count(frozen),
    )

    top_level_counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            top_level = name.split(".", 1)[0]
            top_level_counts[top_level] = (
                top_level_counts.get(top_level, 0) + param.numel()
            )
    for module_name, count in sorted(top_level_counts.items()):
        LOGGER.info(
            "Trainable top-level module | %s=%s params",
            module_name,
            _format_count(count),
        )

    for module_name, module in model.named_modules():
        direct_count = sum(
            param.numel()
            for param in module.parameters(recurse=False)
            if param.requires_grad
        )
        if direct_count > 0:
            LOGGER.info(
                "Trainable module | %s=%s params",
                module_name or "<root>",
                _format_count(direct_count),
            )

    _log_encoder_freeze_status(model, cfg)


def _log_encoder_freeze_status(model: torch.nn.Module, cfg: dict[str, Any]) -> None:
    image_cfg = cfg["model"]["image_encoder"]
    text_cfg = cfg["model"]["text_encoder"]
    image_trainable = [
        (name, param.numel())
        for name, param in model.named_parameters()
        if name.startswith("image_encoder.") and param.requires_grad
    ]
    text_trainable = [
        (name, param.numel())
        for name, param in model.named_parameters()
        if name.startswith("text_encoder.") and param.requires_grad
    ]

    LOGGER.info(
        "Encoder optimizer participation | image_encoder_trainable=%s | text_encoder_trainable=%s",
        _format_count(sum(count for _, count in image_trainable)),
        _format_count(sum(count for _, count in text_trainable)),
    )

    if image_cfg.get("freeze", True):
        allowed_pos_embed = bool(image_cfg.get("train_pos_embed_when_frozen", False))
        unexpected = [
            name
            for name, _ in image_trainable
            if not (allowed_pos_embed and name.endswith("positional_embedding"))
        ]
        if unexpected:
            LOGGER.warning(
                "model.image_encoder.freeze=true but these image encoder parameters are trainable: %s",
                unexpected,
            )
        elif image_trainable:
            LOGGER.info(
                "model.image_encoder.freeze=true; only train_pos_embed_when_frozen parameters are trainable: %s",
                [name for name, _ in image_trainable],
            )
        else:
            LOGGER.info(
                "model.image_encoder.freeze=true; image encoder has no trainable optimizer parameters."
            )

    if text_cfg.get("freeze", True):
        if text_trainable:
            LOGGER.warning(
                "model.text_encoder.freeze=true but these text encoder parameters are trainable: %s",
                [name for name, _ in text_trainable],
            )
        else:
            LOGGER.info(
                "model.text_encoder.freeze=true; text encoder has no trainable optimizer parameters."
            )


def log_optimizer_group_summary(
    new_params: list[torch.nn.Parameter],
    image_params: list[torch.nn.Parameter],
    text_params: list[torch.nn.Parameter],
) -> None:
    LOGGER.info(
        "Optimizer parameter groups | new_modules=%s | image_encoder=%s | text_encoder=%s",
        _format_count(sum(param.numel() for param in new_params)),
        _format_count(sum(param.numel() for param in image_params)),
        _format_count(sum(param.numel() for param in text_params)),
    )


def log_training_label_statistics(train_df: pd.DataFrame) -> None:
    if "sample_type" in train_df.columns:
        LOGGER.info("Training sample_type distribution:")
        for sample_type, count in (
            train_df["sample_type"].value_counts(dropna=False).items()
        ):
            LOGGER.info("%s=%s", sample_type, _format_count(int(count)))

    has_weak_valid_masks = all(
        f"weak_valid_mask_{slug}" in train_df.columns for slug in CLASS_SLUGS
    )
    LOGGER.info("Training label statistics by class:")
    for class_name, slug in zip(CLASSES, CLASS_SLUGS):
        weak_label = train_df[f"weak_label_{slug}"].astype(int)
        omit_label = train_df[f"omit_label_{slug}"].astype(int)
        omit_mask = train_df[f"omit_mask_{slug}"].astype(int)
        report_label = (
            train_df[f"report_label_{slug}"].astype(int)
            if f"report_label_{slug}" in train_df.columns
            else None
        )
        weak_valid_mask = (
            train_df[f"weak_valid_mask_{slug}"].astype(int)
            if has_weak_valid_masks
            else pd.Series(np.ones(len(train_df), dtype=np.int64), index=train_df.index)
        )
        valid = omit_mask == 1
        weak_valid = weak_valid_mask == 1
        valid_count = int(valid.sum())
        report_pos = int((report_label == 1).sum()) if report_label is not None else -1
        LOGGER.info(
            (
                "%s | weak_pos=%s weak_neg=%s | "
                "weak_valid=%s weak_pos_valid=%s weak_neg_valid=%s | "
                "omit_label_pos_all=%s omit_label_neg_all=%s | "
                "omit_mask_valid=%s omit_pos_valid=%s omit_neg_valid=%s | "
                "report_pos=%s"
            ),
            class_name,
            _format_count(int((weak_label == 1).sum())),
            _format_count(int((weak_label == 0).sum())),
            _format_count(int(weak_valid.sum())),
            _format_count(int(((weak_label == 1) & weak_valid).sum())),
            _format_count(int(((weak_label == 0) & weak_valid).sum())),
            _format_count(int((omit_label == 1).sum())),
            _format_count(int((omit_label == 0).sum())),
            _format_count(valid_count),
            _format_count(int(((omit_label == 1) & valid).sum())),
            _format_count(int(((omit_label == 0) & valid).sum())),
            _format_count(report_pos) if report_label is not None else "n/a",
        )
        if has_weak_valid_masks and int(weak_valid.sum()) == 0:
            LOGGER.warning(
                "%s has zero samples with weak_valid_mask==1; weak loss/metrics ignore this class.",
                class_name,
            )
        if valid_count == 0:
            LOGGER.warning(
                "%s has zero samples with omit_mask==1; omission loss/metrics ignore this class.",
                class_name,
            )


def _pair_weight_override_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("loss", {}).get("pair_weight_override", {}) or {})


def _tail_classes_for_pair_weights(cfg: dict[str, Any]) -> set[str]:
    override_cfg = _pair_weight_override_cfg(cfg)
    override_tail = override_cfg.get("tail_classes")
    sampler_tail = cfg.get("train_sampler", {}).get("tail_classes", [])
    tail_classes = list(override_tail if override_tail is not None else sampler_tail)
    unknown = sorted(set(tail_classes) - set(CLASS_SLUGS))
    if unknown:
        raise ValueError(f"Unknown loss.pair_weight_override.tail_classes: {unknown}")
    return set(tail_classes)


def _pair_weight_value(
    override_cfg: dict[str, Any], group: str, key: str, default: float
) -> float:
    group_cfg = override_cfg.get(group, {})
    if not isinstance(group_cfg, dict):
        raise ValueError(f"loss.pair_weight_override.{group} must be a mapping.")
    return float(group_cfg.get(key, override_cfg.get(key, default)))


def _summarize_weight(class_name: str, pair_type: str, weights: np.ndarray) -> None:
    count = int(weights.size)
    if count == 0:
        LOGGER.info(
            "pair_weight_summary | class=%s | pair_type=%s | count=0 | mean_weight=nan | min_weight=nan | max_weight=nan",
            class_name,
            pair_type,
        )
        return
    LOGGER.info(
        "pair_weight_summary | class=%s | pair_type=%s | count=%s | mean_weight=%.4f | min_weight=%.4f | max_weight=%.4f",
        class_name,
        pair_type,
        _format_count(count),
        float(weights.mean()),
        float(weights.min()),
        float(weights.max()),
    )


def log_pair_weight_summary(train_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Log effective train omission-loss pair weights before training starts."""
    loss_cfg = cfg.get("loss", {})
    override_cfg = _pair_weight_override_cfg(cfg)
    use_override = bool(override_cfg.get("enable", False))
    clean_negative_multiplier = float(
        override_cfg.get("clean_negative_multiplier", 1.0)
    )
    tail_classes = _tail_classes_for_pair_weights(cfg)
    LOGGER.info(
        "Pair weight summary | override_enable=%s | clean_negative_multiplier=%.4f | tail_classes=%s",
        use_override,
        clean_negative_multiplier,
        sorted(tail_classes),
    )
    LOGGER.info(
        "pair_weight_summary | class | pair_type | count | mean_weight | min_weight | max_weight"
    )

    for class_name, slug in zip(CLASSES, CLASS_SLUGS):
        is_tail = slug in tail_classes
        group = "tail" if is_tail else "non_tail"
        omit_label = train_df[f"omit_label_{slug}"].astype(int)
        weak_label = train_df[f"weak_label_{slug}"].astype(int)
        omit_mask = train_df[f"omit_mask_{slug}"].astype(int) == 1
        is_clean = infer_is_clean_series(train_df).astype(int) == 1

        if use_override:
            weights = np.ones(len(train_df), dtype=np.float32)
            weights[(omit_label == 1).to_numpy()] = _pair_weight_value(
                override_cfg, group, "omit_positive", 3.0
            )
            weights[((omit_label == 0) & (weak_label == 1)).to_numpy()] = (
                _pair_weight_value(
                    override_cfg,
                    group,
                    "hard_negative",
                    2.0,
                )
            )
            weights[((omit_label == 0) & (weak_label == 0)).to_numpy()] = (
                _pair_weight_value(
                    override_cfg,
                    group,
                    "weak_negative",
                    1.0,
                )
            )
            weights[
                ((omit_label == 0) & is_clean).to_numpy()
            ] *= clean_negative_multiplier
        else:
            weights = np.ones(len(train_df), dtype=np.float32)
            if bool(loss_cfg.get("use_omit_pair_weight", False)):
                column = f"omit_pair_weight_{slug}"
                if column in train_df.columns:
                    weights *= train_df[column].astype(float).to_numpy(dtype=np.float32)
            if bool(loss_cfg.get("use_tail_class_weight", False)):
                column = f"tail_class_weight_{slug}"
                if column in train_df.columns:
                    weights *= train_df[column].astype(float).to_numpy(dtype=np.float32)

        effective_weights = weights * omit_mask.astype(float).to_numpy(dtype=np.float32)
        pair_type_column = f"pair_type_{slug}"
        if pair_type_column in train_df.columns:
            pair_type = train_df[pair_type_column].fillna("NA").astype(str)
            ordered_pair_types = ["O", "S", "L", "W", "M"]
            extras = sorted(set(pair_type.unique()) - set(ordered_pair_types))
            masks = {
                value: pair_type == value for value in [*ordered_pair_types, *extras]
            }
        else:
            masks = {
                "O": omit_mask & (omit_label == 1),
                "L": omit_mask & (omit_label == 0) & (weak_label == 1) & ~is_clean,
                "L_clean": omit_mask & (omit_label == 0) & (weak_label == 1) & is_clean,
                "W": omit_mask & (omit_label == 0) & (weak_label == 0) & ~is_clean,
                "W_clean": omit_mask & (omit_label == 0) & (weak_label == 0) & is_clean,
            }
        for pair_type, mask in masks.items():
            _summarize_weight(class_name, pair_type, effective_weights[mask.to_numpy()])


def log_pos_weights(
    omit_pos_weight: torch.Tensor, weak_pos_weight: torch.Tensor
) -> None:
    LOGGER.info("Omission pos_weight by class:")
    for class_name, value in zip(CLASSES, omit_pos_weight.detach().cpu().tolist()):
        LOGGER.info("%s omit_pos_weight=%.6f", class_name, float(value))
    LOGGER.info("Weak visual pos_weight by class:")
    for class_name, value in zip(CLASSES, weak_pos_weight.detach().cpu().tolist()):
        LOGGER.info("%s weak_pos_weight=%.6f", class_name, float(value))
