"""Evaluate CGQ-Net using fixed validation-selected category thresholds."""

from __future__ import annotations

import argparse

from inference import threshold_vector
from optimize_thresholds import (
    collect_predictions,
    compact_metrics,
    evaluate_threshold_scheme,
    load_lit_module,
    make_loader,
)
from src.utils.io import (
    assert_eval_paths_filled,
    load_json,
    load_yaml,
    save_json,
    validate_config_structure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold-json", default="configs/thresholds_paper.json")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="outputs/evaluation_metrics.json")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    errors = validate_config_structure(cfg)
    if errors:
        raise ValueError("\n".join(errors))
    assert_eval_paths_filled(cfg, args.split)
    payload = load_json(args.threshold_json)
    threshold_vector(payload)
    if payload.get("fit_split") != "val":
        raise ValueError(
            "Evaluation requires thresholds selected on the validation split."
        )
    t = payload["thresholds"]
    module = load_lit_module(cfg, args.checkpoint)
    loader = make_loader(
        cfg, cfg["data"][args.split + "_parquet"], args.batch_size, args.num_workers
    )
    predictions = collect_predictions(module, loader)
    metrics = evaluate_threshold_scheme(
        predictions, t["per_class_best_f1"], t["report_best_f1"], t["fixed"], args.split
    )
    save_json(
        {
            "model": "CGQ-Net",
            "split": args.split,
            "threshold_fit_split": "val",
            "metrics": metrics,
        },
        args.output,
    )
    for key, value in compact_metrics(metrics, args.split).items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
