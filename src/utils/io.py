from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        return re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}",
            lambda m: os.environ.get(m[1], m[2] if m[2] is not None else m[0]),
            value,
        )
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return _expand(data)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_config_structure(cfg: dict[str, Any]) -> list[str]:
    required = {
        "model",
        "loss",
        "data",
        "train",
        "threshold",
        "finetune",
        "image_preprocess",
        "train_augmentation",
        "val_test_preprocess",
    }
    errors = [
        f"Missing top-level config key: {key}" for key in sorted(required - cfg.keys())
    ]
    if errors:
        return errors
    model = cfg["model"]
    checks = {
        "model.type": model.get("type", "full") in {"full", "ours", "omission"},
        "model.num_classes": model.get("num_classes") == 9,
        "model.queries_per_class": model.get("queries_per_class") == 1,
        "model.queries_per_class_by_class": all(
            v == 1 for v in model.get("queries_per_class_by_class", {}).values()
        ),
        "model.omission_head_type": model.get("omission_head_type")
        == "prototype_coverage_gated",
        "model.prototype_pooling": model.get("prototype_pooling", "attention")
        == "attention",
        "model.coverage_pooling": model.get("coverage_pooling", "attention")
        == "attention",
        "model.coverage_gate.enable": model.get("coverage_gate", {}).get("enable", True)
        is True,
        "model.coverage_gate.weak_visual_support": not model.get("coverage_gate", {})
        .get("weak_visual_support", {})
        .get("enable", False),
        "model.cross_attention.share_visual_text_params": not model.get(
            "cross_attention", {}
        ).get("share_visual_text_params", False),
        "model.weak_head.pooling": model.get("weak_head", {}).get(
            "pooling", "feature_mean"
        )
        == "feature_mean",
        "model.query_bank_type": model.get("query_bank_type", "class_queries")
        == "class_queries",
        "model.query.image_residual": not model.get("query", {})
        .get("image_residual", {})
        .get("enable", False),
        "model.query_init.path": not model.get("query_init", {}).get("path"),
        "model.query_init.required": not model.get("query_init", {}).get(
            "required", False
        ),
        "train_sampler.enable": not cfg.get("train_sampler", {}).get("enable", False),
    }
    errors.extend(
        f"{key} does not match the released CGQ-Net architecture."
        for key, valid in checks.items()
        if not valid
    )
    for key in [
        "do_not_modify_dataset",
        "do_not_filter_samples",
        "do_not_resample",
        "use_is_clean_only_for_metrics",
    ]:
        if cfg["data"].get(key) is not True:
            errors.append(f"data.{key} must be true.")
    return errors


def _require_resolved(value: Any, name: str) -> None:
    if not value or "${" in str(value):
        raise ValueError(
            f"Set {name} in the YAML or set its referenced environment variable."
        )


def assert_training_paths_filled(cfg: dict[str, Any]) -> None:
    for key in ["train_parquet", "val_parquet", "test_parquet", "image_root"]:
        _require_resolved(cfg["data"].get(key), "data." + key)
    _require_resolved(
        cfg["model"]["image_encoder"].get("checkpoint_path"),
        "model.image_encoder.checkpoint_path",
    )


def assert_eval_paths_filled(cfg: dict[str, Any], split: str) -> None:
    for key in [f"{split}_parquet", "image_root"]:
        _require_resolved(cfg["data"].get(key), "data." + key)


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("train", {}).get("output_dir", "outputs"))
