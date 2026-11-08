"""Keep aligned main-model scores from the exact final FP32 evaluation pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_eval_cache(path, predictions, sample_ids, cfg, args, threshold_path):
    from src.data.parquet_dataset import CLASS_SLUGS

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prediction cache: {path}")
    ids = np.asarray(sample_ids, dtype=str)
    arrays = {"sample_id": ids, "class_slugs": np.asarray(CLASS_SLUGS, dtype=str)}
    for key in [
        "omit_prob",
        "weak_prob",
        "coverage_prob",
        "omit_labels",
        "omit_masks",
        "weak_labels",
        "weak_masks",
        "is_clean",
    ]:
        if key not in predictions:
            if key == "weak_masks":
                continue
            raise ValueError(f"Missing final-evaluation array: {key}")
        value = np.asarray(predictions[key])
        if len(value) != len(ids):
            raise ValueError(f"Prediction/sample-ID misalignment: {key}")
        if key.endswith("prob") and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite final scores: {key}")
        arrays[key] = value.astype(np.float32 if key.endswith("prob") else np.int8)
    metadata = {
        "producer": "optimize_thresholds.py",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "threshold_json": str(threshold_path),
        "batch_size": args.batch_size or cfg["train"]["batch_size"],
        "num_workers": args.num_workers,
        "precision": "FP32/no autocast",
        "resize_mode": cfg["val_test_preprocess"]["resize_mode"],
        "image_size": cfg["val_test_preprocess"]["image_size"],
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "visual_support_key": "weak_prob",
        "coverage_key": "coverage_prob",
        "omission_key": "omit_prob",
        "encoder_layout": "OpenCLIP NLD public interface",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)), **arrays)
    print(f"Wrote same-run FP32 omission, visual-support and coverage scores: {path}")
