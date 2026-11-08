from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from src.utils.checkpoint import extract_state_dict, load_visual_checkpoint
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def infer_visual_patch_size(visual: nn.Module) -> int:
    conv1 = getattr(visual, "conv1", None)
    if conv1 is None:
        raise ValueError(
            "CLIP visual tower does not expose conv1; cannot infer patch size."
        )
    stride = conv1.stride
    kernel_size = conv1.kernel_size
    patch_size = stride[0] if isinstance(stride, tuple) else stride
    kernel = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
    if patch_size != kernel:
        LOGGER.warning(
            "visual.conv1 stride (%s) differs from kernel size (%s). Using stride.",
            patch_size,
            kernel,
        )
    return int(patch_size)


def resize_clip_pos_embed(visual: nn.Module, image_size: int) -> None:
    if not hasattr(visual, "positional_embedding"):
        raise ValueError(
            "CLIP visual tower does not expose visual.positional_embedding."
        )
    patch_size = infer_visual_patch_size(visual)
    if image_size % patch_size != 0:
        raise AssertionError(
            f"image_size={image_size} must be divisible by patch_size={patch_size}."
        )
    pos_embed = visual.positional_embedding
    if pos_embed.ndim != 2:
        raise ValueError(
            f"Expected positional_embedding [tokens, dim], got {tuple(pos_embed.shape)}."
        )
    num_tokens, dim = pos_embed.shape
    num_patch_tokens = num_tokens - 1
    old_grid = int(math.sqrt(num_patch_tokens))
    if old_grid * old_grid != num_patch_tokens:
        raise ValueError(
            f"Cannot infer square old grid from {num_patch_tokens} patch positional tokens."
        )
    new_grid = image_size // patch_size
    if new_grid == old_grid:
        return
    cls_pos = pos_embed[:1]
    patch_pos = pos_embed[1:]
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(),
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=pos_embed.dtype, device=pos_embed.device)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_grid * new_grid, dim)
    new_pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
    visual.positional_embedding = nn.Parameter(
        new_pos_embed, requires_grad=pos_embed.requires_grad
    )
    for attr in ("input_resolution", "image_size"):
        if hasattr(visual, attr):
            setattr(visual, attr, image_size)
    if hasattr(visual, "grid_size"):
        setattr(visual, "grid_size", (new_grid, new_grid))
    LOGGER.info(
        "Resized CLIP visual positional embedding from %dx%d to %dx%d.",
        old_grid,
        old_grid,
        new_grid,
        new_grid,
    )


class CXRRepairCLIPVisualEncoder(nn.Module):
    """CLIP ViT-B/32 visual tower that returns patch tokens before final global projection."""

    def __init__(self, cfg: dict[str, Any], require_checkpoint: bool = True) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Install open_clip_torch to use the CXR-RePaiR CLIP visual encoder."
            ) from exc
        self.cfg = cfg
        self.image_size = int(cfg.get("image_size", 512))
        self.pretrained_image_size = int(cfg.get("pretrained_image_size", 224))
        self.freeze_backbone = bool(cfg.get("freeze", True))
        checkpoint_path = str(cfg.get("checkpoint_path", "")).strip()
        try:
            clip_model = open_clip.create_model(
                "ViT-B-32", pretrained=None, force_image_size=self.pretrained_image_size
            )
        except TypeError:
            clip_model = open_clip.create_model("ViT-B-32", pretrained=None)
        self.visual = clip_model.visual
        self.output_dim = int(
            getattr(
                self.visual,
                "output_dim",
                getattr(self.visual.conv1, "out_channels", 768),
            )
        )
        self.output_dim = int(
            getattr(self.visual.conv1, "out_channels", self.output_dim)
        )
        if not checkpoint_path:
            if require_checkpoint:
                raise ValueError(
                    "Please download the CXR-RePaiR MIMIC-CXR pretrained CLIP checkpoint and set model.image_encoder.checkpoint_path."
                )
        else:
            load_visual_checkpoint(
                self.visual,
                Path(checkpoint_path),
                strict=bool(cfg.get("strict_load", False)),
            )
        if self.image_size != self.pretrained_image_size:
            if not cfg.get("interpolate_pos_embed", True):
                raise ValueError(
                    "image_size differs from pretrained_image_size; enable interpolate_pos_embed."
                )
            try:
                resize_clip_pos_embed(self.visual, self.image_size)
            except Exception as exc:
                raise RuntimeError(
                    "Set image_size=224 or provide a compatible checkpoint."
                ) from exc
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        if not self.freeze_backbone:
            return
        for param in self.visual.parameters():
            param.requires_grad = False
        if self.cfg.get("train_pos_embed_when_frozen", True) and hasattr(
            self.visual, "positional_embedding"
        ):
            self.visual.positional_embedding.requires_grad = True

    def unfreeze_last_blocks(self, n_blocks: int) -> None:
        if n_blocks <= 0:
            return
        blocks = getattr(getattr(self.visual, "transformer", None), "resblocks", None)
        if blocks is None:
            LOGGER.warning(
                "Could not find visual.transformer.resblocks for finetuning."
            )
            return
        for block in list(blocks)[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        visual = self.visual
        x = visual.conv1(images)
        batch_size, width, grid_h, grid_w = x.shape
        x = x.reshape(batch_size, width, grid_h * grid_w).permute(0, 2, 1)
        cls = visual.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls + torch.zeros(batch_size, 1, width, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        legacy_sequence_first = not hasattr(visual.transformer, "batch_first")
        if legacy_sequence_first:
            x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        if isinstance(x, tuple):
            x = x[0]
        if legacy_sequence_first:
            x = x.permute(1, 0, 2)
        tokens = visual.ln_post(x)
        return {"cls_token": tokens[:, 0], "patch_tokens": tokens[:, 1:]}


class SyntheticImageEncoder(nn.Module):

    def __init__(
        self, image_size: int = 512, patch_size: int = 32, output_dim: int = 768
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.output_dim = output_dim
        self.proj = nn.Linear(3, output_dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(images, (self.grid, self.grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        patch_tokens = self.proj(tokens)
        return {"cls_token": patch_tokens.mean(dim=1), "patch_tokens": patch_tokens}
