from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

COMMON_PREFIXES = (
    "module.",
    "model.",
    "clip.",
    "clip_model.",
    "visual.",
    "image_encoder.",
)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Checkpoint must be a raw state_dict or contain checkpoint['state_dict'] / checkpoint['model']."
    )


def strip_common_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in COMMON_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def load_visual_checkpoint(
    visual: torch.nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    target_state = visual.state_dict()
    target_keys = set(target_state)
    filtered: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    skipped_shape: list[str] = []

    for raw_key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        key = strip_common_prefixes(raw_key)
        if key not in target_keys:
            unexpected.append(raw_key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_shape.append(
                f"{raw_key} -> {key}: {tuple(value.shape)} != {tuple(target_state[key].shape)}"
            )
            continue
        filtered[key] = value

    if not filtered:
        raise RuntimeError(
            "No visual tower weights were matched from the checkpoint. "
            "Please verify this is a CXR-RePaiR / CLIP visual checkpoint."
        )

    load_result = visual.load_state_dict(filtered, strict=strict)
    loaded_keys = set(filtered)
    loaded_tensor_count = len(loaded_keys)
    has_core_conv = any(key.startswith("conv1.") for key in loaded_keys)
    has_transformer = any(key.startswith("transformer.") for key in loaded_keys)
    if loaded_tensor_count < 5 or not (has_core_conv and has_transformer):
        raise RuntimeError(
            "The checkpoint loaded too few visual backbone tensors. "
            "Please verify checkpoint_path points to the CXR-RePaiR MIMIC-CXR pretrained CLIP checkpoint."
        )

    LOGGER.info(
        "Loaded %d visual tensors from %s", loaded_tensor_count, checkpoint_path
    )
    LOGGER.info("Missing visual keys: %s", list(load_result.missing_keys))
    LOGGER.info("Unexpected checkpoint keys ignored: %s", unexpected[:50])
    if skipped_shape:
        LOGGER.info("Shape-mismatched visual keys skipped: %s", skipped_shape[:50])
