from __future__ import annotations
import random
from typing import Any
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as F

try:
    import cv2
except ImportError:
    cv2 = None
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
UINT16_IMAGE_MODES = {"I;16", "I;16L", "I;16B"}


def _interpolation(name: str) -> InterpolationMode:
    mapping = {
        "bicubic": InterpolationMode.BICUBIC,
        "bilinear": InterpolationMode.BILINEAR,
        "nearest": InterpolationMode.NEAREST,
    }
    return mapping.get(str(name).lower(), InterpolationMode.BICUBIC)


def _apply_clahe(image: Image.Image, clip_limit: float = 1.5) -> Image.Image:
    gray = image.convert("L")
    array = np.asarray(gray)
    if cv2 is None:
        return gray
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    out = clahe.apply(array)
    return Image.fromarray(out, mode="L")


def _to_8bit_grayscale(image: Image.Image) -> Image.Image:
    """Convert mixed 8/16-bit images to deterministic 8-bit grayscale."""
    if image.mode in UINT16_IMAGE_MODES:
        array = np.asarray(image).astype(np.float32)
        array = np.clip(np.rint(array * (255.0 / 65535.0)), 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(array)
    return image.convert("L")


def _prepare_cxr_channels(image: Image.Image, force_gray: bool) -> Image.Image:
    """Preserve the configured image-input contract before tensor conversion."""
    if force_gray:
        return _to_8bit_grayscale(image)
    return image.convert("RGB")


def _to_three_channel_tensor(image: Image.Image, to_3ch: bool = True) -> torch.Tensor:
    tensor = F.to_tensor(image).float()
    if to_3ch and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return tensor


def _normalize_clip(tensor: torch.Tensor, normalize: str) -> torch.Tensor:
    if normalize == "clip_mean_std":
        return F.normalize(tensor, mean=CLIP_MEAN, std=CLIP_STD)
    if normalize == "imagenet_mean_std":
        return F.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if normalize == "xrv_1024":
        return tensor * 2048.0 - 1024.0
    return tensor


class CXRTrainTransform:
    """Mild CXR augmentation, applied in memory only during training."""

    def __init__(
        self, preprocess_cfg: dict[str, Any], augmentation_cfg: dict[str, Any]
    ) -> None:
        self.preprocess_cfg = preprocess_cfg
        self.augmentation_cfg = augmentation_cfg
        self.image_size = int(preprocess_cfg.get("image_size", 512))
        self.interpolation = _interpolation(
            preprocess_cfg.get("interpolation", "bicubic")
        )
        self.normalize = preprocess_cfg.get("normalize", "clip_mean_std")
        self.to_3ch = bool(preprocess_cfg.get("to_3ch", True))
        self.force_gray = bool(preprocess_cfg.get("force_gray", True))
        jitter_cfg = augmentation_cfg.get("color_jitter", {})
        self.color_jitter = ColorJitter(
            brightness=float(jitter_cfg.get("brightness", 0.1)),
            contrast=float(jitter_cfg.get("contrast", 0.1)),
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = _prepare_cxr_channels(image, self.force_gray)
        aug = self.augmentation_cfg
        if aug.get("enable", True) and aug.get("random_resized_crop", {}).get(
            "enable", True
        ):
            crop_cfg = aug["random_resized_crop"]
            i, j, h, w = RandomResizedCrop.get_params(
                image,
                scale=tuple(crop_cfg.get("scale", [0.9, 1.0])),
                ratio=tuple(crop_cfg.get("ratio", [0.95, 1.05])),
            )
            image = F.resized_crop(
                image,
                i,
                j,
                h,
                w,
                [self.image_size, self.image_size],
                interpolation=self.interpolation,
            )
        else:
            image = F.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=self.interpolation,
            )
        if aug.get("random_affine", {}).get("enable", False):
            affine_cfg = aug["random_affine"]
            translate = float(affine_cfg.get("translate", 0.02))
            max_dx = int(round(translate * image.size[0]))
            max_dy = int(round(translate * image.size[1]))
            tx = random.randint(-max_dx, max_dx)
            ty = random.randint(-max_dy, max_dy)
            scale_min, scale_max = affine_cfg.get("scale", [0.95, 1.05])
            image = F.affine(
                image,
                angle=0.0,
                translate=[tx, ty],
                scale=random.uniform(float(scale_min), float(scale_max)),
                shear=[0.0, 0.0],
                interpolation=self.interpolation,
                fill=0,
            )
        rotate_cfg = aug.get("random_rotate", {})
        if rotate_cfg.get("enable", True) and random.random() < float(
            rotate_cfg.get("prob", 0.3)
        ):
            degrees = float(rotate_cfg.get("degrees", 5))
            image = F.rotate(
                image,
                angle=random.uniform(-degrees, degrees),
                interpolation=self.interpolation,
                fill=0,
            )
        flip_cfg = aug.get("horizontal_flip", {})
        if flip_cfg.get("enable", False) and random.random() < float(
            flip_cfg.get("prob", 0.0)
        ):
            image = F.hflip(image)
        clahe_cfg = aug.get("clahe", {})
        if clahe_cfg.get("enable", True) and random.random() < float(
            clahe_cfg.get("prob", 0.3)
        ):
            image = _apply_clahe(
                image, clip_limit=float(clahe_cfg.get("clip_limit", 1.5))
            )
        tensor = _to_three_channel_tensor(image, to_3ch=self.to_3ch)
        jitter_cfg = aug.get("color_jitter", {})
        if jitter_cfg.get("enable", False):
            tensor = self.color_jitter(tensor)
        noise_cfg = aug.get("gaussian_noise", {})
        if noise_cfg.get("enable", False):
            std = float(noise_cfg.get("std", 0.01))
            tensor = (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)
        return _normalize_clip(tensor, self.normalize)


class CXRValTestTransform:
    """Deterministic CXR preprocessing for validation, testing, and inference."""

    SUPPORTED_RESIZE_MODES = {"short_edge_center_crop", "direct_resize"}

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.resize_mode = str(cfg.get("resize_mode", "short_edge_center_crop")).lower()
        if self.resize_mode not in self.SUPPORTED_RESIZE_MODES:
            raise ValueError(
                f"val_test_preprocess.resize_mode must be one of {sorted(self.SUPPORTED_RESIZE_MODES)}, got {self.resize_mode!r}."
            )
        self.image_size = int(cfg.get("image_size", cfg.get("center_crop", 512)))
        if self.resize_mode == "direct_resize" and self.image_size <= 0:
            raise ValueError(
                "val_test_preprocess.image_size must be positive for direct_resize."
            )
        self.resize_short_edge = int(cfg.get("resize_short_edge", 544))
        self.center_crop = int(cfg.get("center_crop", 512))
        self.interpolation = _interpolation(cfg.get("interpolation", "bicubic"))
        self.normalize = cfg.get("normalize", "clip_mean_std")
        self.to_3ch = bool(cfg.get("to_3ch", True))
        self.force_gray = bool(cfg.get("force_gray", True))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = _prepare_cxr_channels(image, self.force_gray)
        if self.cfg.get("apply_clahe", False):
            image = _apply_clahe(image)
        if self.resize_mode == "direct_resize":
            image = F.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=self.interpolation,
            )
        else:
            image = F.resize(
                image, self.resize_short_edge, interpolation=self.interpolation
            )
            image = F.center_crop(image, [self.center_crop, self.center_crop])
        tensor = _to_three_channel_tensor(image, to_3ch=self.to_3ch)
        return _normalize_clip(tensor, self.normalize)


def build_train_transform(cfg: dict[str, Any]) -> CXRTrainTransform:
    return CXRTrainTransform(cfg["image_preprocess"], cfg["train_augmentation"])


def build_val_transform(cfg: dict[str, Any]) -> CXRValTestTransform:
    return CXRValTestTransform(cfg["val_test_preprocess"])
