import copy
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from inference import threshold_vector
from optimize_thresholds import evaluate_threshold_scheme
from src.data.parquet_dataset import CLASS_SLUGS
from src.data.transforms import build_val_transform
from src.models.omission_model import build_synthetic_model
from src.utils.io import load_yaml, validate_config_structure


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
