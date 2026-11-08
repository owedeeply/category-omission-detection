"""Run CGQ-Net on a chest radiograph and a candidate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from optimize_thresholds import load_lit_module
from src.data.parquet_dataset import CLASS_SLUGS
from src.data.transforms import build_val_transform
from src.utils.io import load_json, load_yaml, save_json, validate_config_structure


def threshold_vector(payload: dict) -> list[float]:
    if list(payload.get("class_order", CLASS_SLUGS)) != list(CLASS_SLUGS):
        raise ValueError("Threshold class order does not match CGQ-Net.")
    values = payload["thresholds"]["per_class_best_f1"]
    if set(values) != set(CLASS_SLUGS):
        raise ValueError("Exactly nine category thresholds are required.")
    result = [float(values[c]) for c in CLASS_SLUGS]
    if any(not 0 <= v <= 1 for v in result):
        raise ValueError("Category thresholds must be finite and within [0, 1].")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--report-file", required=True, help="UTF-8 candidate report.")
    parser.add_argument("--threshold-json", default="configs/thresholds_paper.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    errors = validate_config_structure(cfg)
    if errors:
        raise ValueError("\n".join(errors))
    thresholds = threshold_vector(load_json(args.threshold_json))
    report = Path(args.report_file).read_text(encoding="utf-8").strip()
    if not report:
        raise ValueError("The candidate report is empty.")
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model = load_lit_module(cfg, args.checkpoint).model.to(device).eval()
    raw_tokens = model.text_encoder.tokenizer(report, truncation=False)["input_ids"]
    if len(raw_tokens) > model.text_encoder.max_length:
        raise ValueError(
            f"Report exceeds the {model.text_encoder.max_length}-token paper input limit."
        )
    tokenized = model.text_encoder.tokenize([report])
    with Image.open(args.image) as image:
        image_tensor = build_val_transform(cfg)(image.copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        result = model(
            {
                "images": image_tensor,
                "input_ids": tokenized["input_ids"].to(device),
                "attention_mask": tokenized["attention_mask"].to(device),
            }
        )
    scores = {
        key: result[key][0].float().cpu().tolist()
        for key in ["omit_prob", "weak_prob", "coverage_prob"]
    }
    categories = [
        {
            "category": c,
            "omission_probability": scores["omit_prob"][i],
            "support_probability": scores["weak_prob"][i],
            "coverage_probability": scores["coverage_prob"][i],
            "threshold": thresholds[i],
            "predicted_omission": scores["omit_prob"][i] >= thresholds[i],
        }
        for i, c in enumerate(CLASS_SLUGS)
    ]
    payload = {
        "model": "CGQ-Net",
        "categories": categories,
        "report_risk": max(scores["omit_prob"]),
        "predicted_any_omission": any(x["predicted_omission"] for x in categories),
    }
    if args.output:
        save_json(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
