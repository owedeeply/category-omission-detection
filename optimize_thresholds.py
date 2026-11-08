from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from src.data.collate import cxr_omission_collate
from src.data.parquet_dataset import CLASS_SLUGS, CLASSES, CxrOmissionDataset
from src.data.transforms import build_val_transform
from src.lightning.lit_omission import LitOmissionModule
from src.metrics.omission_metrics import (
    best_f1_threshold,
    compute_omission_metrics,
    compute_weak_metrics,
    report_scores_and_labels,
    threshold_for_precision,
)
from src.models.omission_model import build_model_from_config
from src.utils.io import (
    assert_eval_paths_filled,
    load_json,
    load_yaml,
    resolve_output_dir,
    save_json,
    validate_config_structure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fit-split", choices=["val"], default="val")
    parser.add_argument("--eval-split", choices=["", "val", "test"], default="test")
    parser.add_argument(
        "--output",
        default="",
        help="Threshold JSON path. Defaults to <output_dir>/thresholds_per_class.json.",
    )
    parser.add_argument(
        "--threshold-json",
        default="",
        help="Use an existing threshold JSON instead of fitting thresholds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override eval batch size. Defaults to config value.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for threshold optimization. Defaults to 0 to reduce RAM use.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable DataLoader pin_memory. Disabled by default to reduce RAM use.",
    )
    parser.add_argument(
        "--eval-prediction-cache",
        default="",
        help="Save aligned same-run omission, visual and coverage scores.",
    )
    return parser.parse_args()


def make_loader(
    cfg: dict[str, Any],
    parquet_path: str,
    batch_size: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    data = cfg["data"]
    resolved_num_workers = int(num_workers)
    print(f"Loading dataset metadata from {parquet_path}...")
    dataset = CxrOmissionDataset(
        parquet_path,
        data["image_root"],
        transform=build_val_transform(cfg),
        require_labels=True,
        image_path_layout=data.get("image_path_layout", "auto"),
    )
    print(f"Loaded {len(dataset)} rows from {parquet_path}.")
    kwargs: dict[str, Any] = {
        "batch_size": (
            int(batch_size)
            if batch_size > 0
            else int(cfg["train"].get("batch_size", 24))
        ),
        "shuffle": False,
        "num_workers": resolved_num_workers,
        "pin_memory": bool(pin_memory),
        "collate_fn": cxr_omission_collate,
        "persistent_workers": False,
    }
    if resolved_num_workers > 0:
        kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def load_checkpoint_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    attempts: list[dict[str, Any]] = [
        {"map_location": "cpu", "weights_only": False, "mmap": True},
        {"map_location": "cpu", "mmap": True},
        {"map_location": "cpu", "weights_only": False},
        {"map_location": "cpu"},
    ]
    last_error: TypeError | None = None
    ckpt: Any = None
    for kwargs in attempts:
        try:
            ckpt = torch.load(checkpoint, **kwargs)
            break
        except TypeError as exc:
            last_error = exc
    if ckpt is None:
        raise RuntimeError(
            "torch.load failed for all supported argument combinations."
        ) from last_error
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt)!r}.")
    state_dict = ckpt.get("state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")
    return state_dict


def load_lit_module(cfg: dict[str, Any], checkpoint: str) -> LitOmissionModule:
    model_cfg = copy.deepcopy(cfg)
    # The Lightning checkpoint already contains the trained visual tower state.
    # Avoid loading the original CXR-RePaiR checkpoint first; that doubles peak CPU
    # memory during threshold optimization on smaller machines.
    model_cfg["model"]["image_encoder"]["checkpoint_path"] = ""
    print("Building model architecture...")
    model = build_model_from_config(model_cfg, require_checkpoint=False)
    ones = torch.ones(len(CLASSES), dtype=torch.float32)
    lit_module = LitOmissionModule(
        model=model,
        cfg=model_cfg,
        omit_pos_weight=ones,
        weak_pos_weight=ones,
    )
    print("Loading checkpoint state_dict...")
    state_dict = load_checkpoint_state_dict(checkpoint)
    load_result = lit_module.load_state_dict(state_dict, strict=True)
    del state_dict
    print(
        "Loaded checkpoint state_dict "
        f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
    )
    return lit_module


@torch.inference_mode()
def collect_predictions(
    lit_module: LitOmissionModule, loader: DataLoader
) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to {device}...")
    model = lit_module.model.to(device).eval()
    outputs: dict[str, list[torch.Tensor]] = {
        "omit_prob": [],
        "weak_prob": [],
        "coverage_prob": [],
        "omit_labels": [],
        "omit_masks": [],
        "weak_labels": [],
        "is_clean": [],
    }
    for batch in tqdm(loader, desc="collect_predictions"):
        batch["images"] = batch["images"].to(device)
        out = model(batch)
        outputs["omit_prob"].append(out["omit_prob"].detach().cpu())
        outputs["weak_prob"].append(out["weak_prob"].detach().cpu())
        outputs["coverage_prob"].append(out["coverage_prob"].detach().cpu())
        outputs["omit_labels"].append(batch["omit_labels"].detach().cpu())
        outputs["omit_masks"].append(batch["omit_masks"].detach().cpu())
        outputs["weak_labels"].append(batch["weak_labels"].detach().cpu())
        outputs["is_clean"].append(batch["is_clean"].detach().cpu())
        del out, batch
    print("Concatenating prediction tensors...")
    return {
        key: torch.cat(value, dim=0).float().numpy() for key, value in outputs.items()
    }


def fit_thresholds(
    preds: dict[str, np.ndarray], fixed_threshold: float
) -> dict[str, Any]:
    probs = preds["omit_prob"]
    labels = preds["omit_labels"].astype(int)
    masks = preds["omit_masks"].astype(int)
    per_class_best_f1: dict[str, float] = {}
    per_class_precision80: dict[str, float] = {}
    per_class_precision90: dict[str, float] = {}

    for idx, slug in enumerate(CLASS_SLUGS):
        valid = masks[:, idx] == 1
        y = labels[valid, idx]
        s = probs[valid, idx]
        per_class_best_f1[slug] = best_f1_threshold(y, s)
        per_class_precision80[slug] = threshold_for_precision(y, s, 0.80)
        per_class_precision90[slug] = threshold_for_precision(y, s, 0.90)

    report_scores, report_labels = report_scores_and_labels(probs, labels, masks)
    return {
        "fixed": float(fixed_threshold),
        "per_class_best_f1": per_class_best_f1,
        "per_class_precision80": per_class_precision80,
        "per_class_precision90": per_class_precision90,
        "report_best_f1": best_f1_threshold(report_labels, report_scores),
        "report_precision80": threshold_for_precision(
            report_labels, report_scores, 0.80
        ),
        "report_precision90": threshold_for_precision(
            report_labels, report_scores, 0.90
        ),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _f1_precision_recall(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = float((y & pred).sum())
    fp = float((~y & pred).sum())
    fn = float((y & ~pred).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return f1, precision, recall


def _threshold_array(thresholds: dict[str, float] | float) -> np.ndarray:
    if isinstance(thresholds, dict):
        return np.asarray(
            [float(thresholds[slug]) for slug in CLASS_SLUGS], dtype=np.float32
        )
    return np.full(len(CLASS_SLUGS), float(thresholds), dtype=np.float32)


def evaluate_threshold_scheme(
    preds: dict[str, np.ndarray],
    class_thresholds: dict[str, float] | float,
    report_threshold: float,
    fixed_threshold: float,
    prefix: str,
) -> dict[str, float]:
    probs = preds["omit_prob"]
    labels = preds["omit_labels"].astype(int)
    masks = preds["omit_masks"].astype(int)
    is_clean = preds["is_clean"].astype(int) == 1
    thresholds = _threshold_array(class_thresholds)
    thresholded = probs >= thresholds[None, :]

    metrics = compute_omission_metrics(
        probs,
        labels,
        masks,
        is_clean=preds["is_clean"],
        threshold=fixed_threshold,
        prefix=prefix,
    )
    metrics.update(
        compute_weak_metrics(preds["weak_prob"], preds["weak_labels"], prefix=prefix)
    )

    per_class_f1 = []
    per_class_precision = []
    per_class_recall = []
    for idx, slug in enumerate(CLASS_SLUGS):
        valid = masks[:, idx] == 1
        f1, precision, recall = _f1_precision_recall(
            labels[valid, idx], thresholded[valid, idx]
        )
        metrics[f"{prefix}/thresholded_omit_{slug}_f1"] = f1
        metrics[f"{prefix}/thresholded_omit_{slug}_precision"] = precision
        metrics[f"{prefix}/thresholded_omit_{slug}_recall"] = recall
        per_class_f1.append(f1)
        per_class_precision.append(precision)
        per_class_recall.append(recall)

    valid_flat = masks.astype(bool)
    micro_f1, micro_precision, micro_recall = _f1_precision_recall(
        labels[valid_flat], thresholded[valid_flat]
    )
    metrics[f"{prefix}/thresholded_omit_macro_f1"] = float(np.mean(per_class_f1))
    metrics[f"{prefix}/thresholded_omit_macro_precision"] = float(
        np.mean(per_class_precision)
    )
    metrics[f"{prefix}/thresholded_omit_macro_recall"] = float(
        np.mean(per_class_recall)
    )
    metrics[f"{prefix}/thresholded_omit_micro_f1"] = micro_f1
    metrics[f"{prefix}/thresholded_omit_micro_precision"] = micro_precision
    metrics[f"{prefix}/thresholded_omit_micro_recall"] = micro_recall

    report_scores, report_labels = report_scores_and_labels(probs, labels, masks)
    report_pred = report_scores >= float(report_threshold)
    report_f1, report_precision, report_recall = _f1_precision_recall(
        report_labels, report_pred
    )
    metrics[f"{prefix}/thresholded_report_f1"] = report_f1
    metrics[f"{prefix}/thresholded_report_precision"] = report_precision
    metrics[f"{prefix}/thresholded_report_recall"] = report_recall

    if is_clean.any():
        metrics[f"{prefix}/thresholded_clean_false_alarm_all_classes"] = float(
            thresholded[is_clean].any(axis=1).mean()
        )
        metrics[f"{prefix}/thresholded_clean_false_alarm_masked"] = float(
            (thresholded & (masks == 1))[is_clean].any(axis=1).mean()
        )
    else:
        metrics[f"{prefix}/thresholded_clean_false_alarm_all_classes"] = float("nan")
        metrics[f"{prefix}/thresholded_clean_false_alarm_masked"] = float("nan")
    return metrics


def compact_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    keys = [
        f"{prefix}/omit_macro_auprc",
        f"{prefix}/omit_micro_auprc",
        f"{prefix}/report_auprc",
        f"{prefix}/thresholded_omit_macro_f1",
        f"{prefix}/thresholded_omit_micro_f1",
        f"{prefix}/thresholded_report_f1",
        f"{prefix}/thresholded_report_precision",
        f"{prefix}/thresholded_report_recall",
        f"{prefix}/thresholded_clean_false_alarm_masked",
    ]
    return {key: float(metrics[key]) for key in keys if key in metrics}


def print_compact(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    errors = validate_config_structure(cfg)
    if errors:
        raise ValueError("\n".join(errors))
    fixed_threshold = float(cfg["threshold"].get("fixed_value", 0.6))
    output_path = (
        Path(args.output)
        if args.output
        else resolve_output_dir(cfg) / "thresholds_per_class.json"
    )

    lit_module = load_lit_module(cfg, args.checkpoint)
    threshold_payload: dict[str, Any]

    if args.threshold_json:
        threshold_payload = load_json(args.threshold_json)
        thresholds = threshold_payload["thresholds"]
    else:
        assert_eval_paths_filled(cfg, args.fit_split)
        fit_loader = make_loader(
            cfg,
            cfg["data"][f"{args.fit_split}_parquet"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        fit_preds = collect_predictions(lit_module, fit_loader)
        thresholds = fit_thresholds(fit_preds, fixed_threshold)
        threshold_payload = {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "fit_split": args.fit_split,
            "thresholds": thresholds,
        }
        val_schemes = {
            "fixed_0.6": evaluate_threshold_scheme(
                fit_preds,
                fixed_threshold,
                fixed_threshold,
                fixed_threshold,
                f"{args.fit_split}/fixed_0.6",
            ),
            "best_f1": evaluate_threshold_scheme(
                fit_preds,
                thresholds["per_class_best_f1"],
                thresholds["report_best_f1"],
                fixed_threshold,
                f"{args.fit_split}/best_f1",
            ),
            "precision80": evaluate_threshold_scheme(
                fit_preds,
                thresholds["per_class_precision80"],
                thresholds["report_precision80"],
                fixed_threshold,
                f"{args.fit_split}/precision80",
            ),
            "precision90": evaluate_threshold_scheme(
                fit_preds,
                thresholds["per_class_precision90"],
                thresholds["report_precision90"],
                fixed_threshold,
                f"{args.fit_split}/precision90",
            ),
        }
        threshold_payload["fit_metrics"] = {
            name: compact_metrics(metrics, f"{args.fit_split}/{name}")
            for name, metrics in val_schemes.items()
        }
        save_json(threshold_payload, output_path)
        print(f"Wrote threshold JSON to {output_path}")
        for name, metrics in threshold_payload["fit_metrics"].items():
            print_compact(f"{args.fit_split} {name}", metrics)

    if args.eval_split:
        assert_eval_paths_filled(cfg, args.eval_split)
        eval_loader = make_loader(
            cfg,
            cfg["data"][f"{args.eval_split}_parquet"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        eval_preds = collect_predictions(lit_module, eval_loader)
        if args.eval_prediction_cache:
            from src.utils.prediction_cache import save_eval_cache

            save_eval_cache(
                args.eval_prediction_cache,
                eval_preds,
                eval_loader.dataset.df["sample_id"].astype(str).to_numpy(dtype=str),
                cfg,
                args,
                output_path,
            )
            threshold_payload["eval_prediction_cache"] = args.eval_prediction_cache
        eval_schemes = {
            "fixed_0.6": evaluate_threshold_scheme(
                eval_preds,
                fixed_threshold,
                fixed_threshold,
                fixed_threshold,
                f"{args.eval_split}/fixed_0.6",
            ),
            "best_f1": evaluate_threshold_scheme(
                eval_preds,
                thresholds["per_class_best_f1"],
                thresholds["report_best_f1"],
                fixed_threshold,
                f"{args.eval_split}/best_f1",
            ),
            "precision80": evaluate_threshold_scheme(
                eval_preds,
                thresholds["per_class_precision80"],
                thresholds["report_precision80"],
                fixed_threshold,
                f"{args.eval_split}/precision80",
            ),
            "precision90": evaluate_threshold_scheme(
                eval_preds,
                thresholds["per_class_precision90"],
                thresholds["report_precision90"],
                fixed_threshold,
                f"{args.eval_split}/precision90",
            ),
        }
        eval_compact = {
            name: compact_metrics(metrics, f"{args.eval_split}/{name}")
            for name, metrics in eval_schemes.items()
        }
        for name, metrics in eval_compact.items():
            print_compact(f"{args.eval_split} {name}", metrics)
        if True:
            threshold_payload["eval_split"] = args.eval_split
            threshold_payload["eval_metrics"] = eval_compact
            save_json(threshold_payload, output_path)


if __name__ == "__main__":
    main()
