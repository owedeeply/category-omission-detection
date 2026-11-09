import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from inference import threshold_vector
from optimize_thresholds import collect_predictions, evaluate_threshold_scheme
from src.data.parquet_dataset import CLASS_SLUGS
from src.data.transforms import build_val_transform
from src.lightning.lit_omission import LitOmissionModule
from src.models.omission_model import build_synthetic_model
from src.utils.io import load_yaml, validate_config_structure


class _PredictionModel(torch.nn.Module):
    def forward(self, batch):
        return {
            key: batch["images"] for key in ["omit_prob", "weak_prob", "coverage_prob"]
        }


class _EvaluationModule:
    _weak_loss_mask = LitOmissionModule._weak_loss_mask

    def __init__(self, weak_mask_source):
        self.model = _PredictionModel()
        self.weak_mask_source = weak_mask_source


class ReleaseContractTest(unittest.TestCase):
    def test_paper_configuration_and_preprocessing(self):
        train = load_yaml("configs/train.yaml")
        evaluation = load_yaml("configs/eval.yaml")
        self.assertEqual(validate_config_structure(train), [])
        self.assertEqual(validate_config_structure(evaluation), [])
        self.assertEqual(train["train"]["seed"], 42)
        self.assertEqual(train["train"]["batch_size"], 128)
        self.assertEqual(
            build_val_transform(train).resize_mode, "short_edge_center_crop"
        )
        self.assertEqual(build_val_transform(evaluation).resize_mode, "direct_resize")
        bad = copy.deepcopy(train)
        bad["model"]["type"] = "text_only"
        self.assertTrue(validate_config_structure(bad))

    def test_environment_variables_are_configurable(self):
        with patch.dict(
            os.environ,
            {"CXR_OMIT_ROOT": "/synthetic-data", "CXR_IMAGE_ROOT": "/synthetic-images"},
        ):
            cfg = load_yaml("configs/train.yaml")
        self.assertEqual(cfg["data"]["image_root"], "/synthetic-images")
        self.assertTrue(cfg["data"]["train_parquet"].startswith("/synthetic-data/"))

    def test_thresholds_use_category_names(self):
        payload = json.loads(Path("configs/thresholds_paper.json").read_text())
        self.assertEqual(len(threshold_vector(payload)), 9)
        self.assertEqual(threshold_vector(payload)[CLASS_SLUGS.index("edema")], 0.805)
        payload["class_order"] = list(reversed(CLASS_SLUGS))
        with self.assertRaises(ValueError):
            threshold_vector(payload)

    def test_masked_metrics_and_false_alarms(self):
        p = np.full((4, 9), 0.1)
        y = np.zeros((4, 9))
        m = np.ones((4, 9))
        p[0] = 0.9
        y[0] = 1
        p[3, 0] = 0.99
        m[3, 0] = 0
        preds = {
            "omit_prob": p,
            "omit_labels": y,
            "omit_masks": m,
            "weak_prob": p,
            "weak_labels": y,
            "is_clean": np.array([0, 1, 1, 1]),
        }
        metrics = evaluate_threshold_scheme(
            preds, {c: 0.5 for c in CLASS_SLUGS}, 0.5, 0.5, "test"
        )
        self.assertEqual(metrics["test/omit_macro_auprc"], 1.0)
        self.assertEqual(metrics["test/thresholded_omit_macro_f1"], 1.0)
        self.assertEqual(metrics["test/thresholded_clean_false_alarm_masked"], 0.0)

    def test_standalone_weak_metrics_respect_mask(self):
        p = np.tile(np.array([0.9, 0.1, 0.99, 0.98])[:, None], (1, 9))
        y = np.tile(np.array([1, 0, 0, 0])[:, None], (1, 9))
        mask = np.tile(np.array([1, 1, 0, 0])[:, None], (1, 9))
        preds = {
            "omit_prob": p,
            "omit_labels": y,
            "omit_masks": mask,
            "weak_prob": p,
            "weak_labels": y,
            "is_clean": np.array([0, 1, 1, 1]),
        }
        unmasked = evaluate_threshold_scheme(preds, 0.5, 0.5, 0.5, "test")
        masked = evaluate_threshold_scheme(
            {**preds, "weak_masks": mask}, 0.5, 0.5, 0.5, "test"
        )
        self.assertAlmostEqual(unmasked["test/weak_macro_auprc"], 1 / 3)
        self.assertEqual(masked["test/weak_macro_auprc"], 1.0)
        for key, value in masked.items():
            if not key.startswith("test/weak_"):
                self.assertEqual(value, unmasked[key], key)

    def test_prediction_collection_preserves_configured_weak_masks(self):
        labels = torch.arange(36).reshape(4, 9).remainder(2).float()
        valid_masks = torch.ones_like(labels)
        valid_masks[2:] = 0
        omit_masks = 1 - valid_masks
        for source, expected in [
            ("valid", valid_masks),
            ("omit", omit_masks),
            ("all", torch.ones_like(labels)),
        ]:
            with self.subTest(source=source):
                batches = [
                    {
                        "images": labels[start : start + 2],
                        "omit_labels": labels[start : start + 2],
                        "omit_masks": omit_masks[start : start + 2],
                        "weak_labels": labels[start : start + 2],
                        "weak_valid_masks": valid_masks[start : start + 2],
                        "has_weak_valid_masks": True,
                        "is_clean": torch.zeros(2),
                    }
                    for start in [0, 2]
                ]
                preds = collect_predictions(_EvaluationModule(source), batches)
                np.testing.assert_array_equal(preds["weak_masks"], expected.numpy())
                np.testing.assert_array_equal(preds["weak_labels"], labels.numpy())

    def test_prediction_collection_rejects_missing_valid_masks(self):
        batch = {
            "images": torch.zeros(1, 9),
            "omit_labels": torch.zeros(1, 9),
            "omit_masks": torch.ones(1, 9),
            "weak_labels": torch.zeros(1, 9),
            "is_clean": torch.zeros(1),
        }
        with self.assertRaisesRegex(ValueError, "requires complete weak_valid_mask"):
            collect_predictions(_EvaluationModule("valid"), [batch])

    def test_main_model_forward_and_gradient(self):
        cfg = load_yaml("configs/train.yaml")
        cfg["model"]["hidden_dim"] = 24
        cfg["model"]["cross_attention"].update(
            num_heads=4, num_layers=1, ffn_dim=48, dropout=0.0
        )
        cfg["model"]["image_encoder"]["image_size"] = 64
        cfg["model"]["text_encoder"].update(synthetic_hidden_size=24, max_length=8)
        model = build_synthetic_model(cfg)
        out = model(
            {
                "images": torch.randn(2, 3, 64, 64),
                "texts": ["synthetic report", "synthetic control"],
            }
        )
        for key in ["omit_prob", "weak_prob", "coverage_prob"]:
            self.assertEqual(tuple(out[key].shape), (2, 9))
            self.assertTrue(torch.isfinite(out[key]).all())
        (
            out["omit_logits"].sum()
            + out["weak_logits"].sum()
            + out["coverage_logit_c"].sum()
        ).backward()
        self.assertIsNotNone(model.query_bank.class_queries.grad)


if __name__ == "__main__":
    unittest.main()
