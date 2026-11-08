"""Regression tests for OpenCLIP's legacy and modern tensor interfaces."""

import unittest
import torch
from torch import nn
from open_clip.transformer import VisionTransformer
from src.models.image_encoder import CXRRepairCLIPVisualEncoder


class _LegacyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            32, 4, 64, dropout=0.0, batch_first=False
        )

    def forward(self, x):
        return self.layer(x)


class _ModernSequenceFirstAttention(_LegacyTransformer):
    """Modern NLD public interface with sequence-first internal attention."""

    batch_first = False

    def forward(self, x):
        return self.layer(x.transpose(0, 1)).transpose(0, 1)


class CLIPVisualBatchIndependenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(817)
        self.visual = VisionTransformer(
            image_size=64,
            patch_size=32,
            width=32,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            output_dim=16,
        )
        self.encoder = CXRRepairCLIPVisualEncoder.__new__(CXRRepairCLIPVisualEncoder)
        nn.Module.__init__(self.encoder)
        self.encoder.visual = self.visual
        self.encoder.eval()
        self.image = torch.randn(1, 3, 64, 64)
        self.distractor = torch.randn(1, 3, 64, 64) * 2.0

    def assertIndependent(self):
        with torch.inference_mode():
            solo = self.encoder(self.image)["patch_tokens"]
            mixed = self.encoder(torch.cat([self.image, self.distractor]))[
                "patch_tokens"
            ][:1]
            repeated = self.encoder(self.image.repeat(2, 1, 1, 1))["patch_tokens"]
        torch.testing.assert_close(solo, mixed, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(repeated[:1], repeated[1:], atol=1e-7, rtol=1e-7)
        return solo

    def test_modern_matches_native_visual_transformer_and_ignores_other_images(self):
        self.assertIndependent()
        images = torch.cat([self.image, self.distractor])
        with torch.inference_mode():
            native = self.visual.ln_post(
                self.visual.transformer(self.visual._embeds(images))
            )
            actual = self.encoder(images)
        torch.testing.assert_close(
            actual["patch_tokens"], native[:, 1:], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            actual["cls_token"], native[:, 0], atol=1e-6, rtol=1e-6
        )

    def test_legacy_lnd_interface_ignores_other_images(self):
        self.visual.transformer = _LegacyTransformer().eval()
        self.assertIndependent()

    def test_modern_nld_interface_with_sequence_first_attention(self):
        self.visual.transformer = _ModernSequenceFirstAttention().eval()
        self.assertIndependent()

    def test_training_gradient_does_not_cross_images(self):
        self.encoder.train()
        images = torch.cat([self.image, self.distractor]).requires_grad_(True)
        value = self.encoder(images)["patch_tokens"][0, 0, 0]
        (gradient,) = torch.autograd.grad(value, images)
        self.assertGreater(float(gradient[0].abs().max()), 0.0)
        self.assertEqual(float(gradient[1].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
