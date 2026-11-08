from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
from src.data.collate import cxr_omission_collate
from src.data.parquet_dataset import CLASS_SLUGS, CxrOmissionDataset
from src.data.transforms import build_train_transform, build_val_transform
from src.losses.asymmetric_loss import asymmetric_loss_with_logits
from src.losses.masked_weighted_bce import (
    masked_weighted_bce_with_logits,
    weighted_bce_with_logits,
)
from src.utils.diagnostics import log_optimizer_group_summary
from src.utils.logging import get_logger
from src.metrics.omission_metrics import (
    compute_omission_metrics,
    compute_text_coverage_metrics,
    compute_val_thresholds,
    compute_weak_metrics,
)

LOGGER = get_logger(__name__)


class CxrOmissionDataModule(pl.LightningDataModule):

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: CxrOmissionDataset | None = None
        self.val_dataset: CxrOmissionDataset | None = None
        self.test_dataset: CxrOmissionDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        data = self.cfg["data"]
        image_root = data["image_root"]
        if stage in (None, "fit"):
            self.train_dataset = CxrOmissionDataset(
                data["train_parquet"],
                image_root,
                transform=build_train_transform(self.cfg),
                require_labels=True,
                image_path_layout=data.get("image_path_layout", "auto"),
            )
            self.val_dataset = CxrOmissionDataset(
                data["val_parquet"],
                image_root,
                transform=build_val_transform(self.cfg),
                require_labels=True,
                image_path_layout=data.get("image_path_layout", "auto"),
            )
        if stage in (None, "test"):
            self.test_dataset = CxrOmissionDataset(
                data["test_parquet"],
                image_root,
                transform=build_val_transform(self.cfg),
                require_labels=True,
                image_path_layout=data.get("image_path_layout", "auto"),
            )

    def _loader(self, dataset: CxrOmissionDataset, shuffle: bool) -> DataLoader:
        data = self.cfg["data"]
        num_workers = int(data.get("num_workers", 8))
        sampler = None
        kwargs: dict[str, Any] = {
            "batch_size": int(self.cfg["train"].get("batch_size", 24)),
            "shuffle": shuffle and sampler is None,
            "num_workers": num_workers,
            "pin_memory": bool(data.get("pin_memory", True)),
            "collate_fn": cxr_omission_collate,
            "persistent_workers": bool(data.get("persistent_workers", True))
            and num_workers > 0,
        }
        if sampler is not None:
            kwargs["sampler"] = sampler
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup('fit') has not been called.")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup('fit') has not been called.")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule.setup('test') has not been called.")
        return self._loader(self.test_dataset, shuffle=False)


class LitOmissionModule(pl.LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: dict[str, Any],
        omit_pos_weight: torch.Tensor,
        weak_pos_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters({"cfg": cfg}, ignore=["model"])
        self.register_buffer("omit_pos_weight", omit_pos_weight.float())
        self.register_buffer("weak_pos_weight", weak_pos_weight.float())
        self.lambda_weak = float(cfg["loss"].get("lambda_weak", 0.5))
        self.lambda_text_coverage = float(cfg["loss"].get("lambda_text_coverage", 0.0))
        self.use_omit_pair_weight = bool(cfg["loss"].get("use_omit_pair_weight", False))
        self.use_tail_class_weight = bool(
            cfg["loss"].get("use_tail_class_weight", False)
        )
        self.pair_weight_override_cfg = dict(
            cfg["loss"].get("pair_weight_override", {}) or {}
        )
        self.use_pair_weight_override = bool(
            self.pair_weight_override_cfg.get("enable", False)
        )
        self.omit_loss_name = str(cfg["loss"].get("omit_loss", "masked_weighted_bce"))
        self.omit_asl_gamma_pos = float(cfg["loss"].get("omit_asl_gamma_pos", 0.0))
        self.omit_asl_gamma_neg = float(cfg["loss"].get("omit_asl_gamma_neg", 2.0))
        self.omit_asl_eps = float(cfg["loss"].get("omit_asl_eps", 1e-08))
        self.register_buffer(
            "omit_asl_gamma_pos_vec",
            self._gamma_vector_from_config(
                "omit_asl_gamma_pos_by_class", self.omit_asl_gamma_pos
            ),
        )
        self.register_buffer(
            "omit_asl_gamma_neg_vec",
            self._gamma_vector_from_config(
                "omit_asl_gamma_neg_by_class", self.omit_asl_gamma_neg
            ),
        )
        self.register_buffer("tail_class_mask", self._tail_class_mask_from_config())
        self.weak_use_omit_mask = bool(cfg["loss"].get("weak_use_omit_mask", False))
        self.weak_mask_source = str(
            cfg["loss"].get(
                "weak_mask_source", "omit" if self.weak_use_omit_mask else "all"
            )
        ).lower()
        if self.weak_mask_source not in {"all", "omit", "valid"}:
            raise ValueError("loss.weak_mask_source must be one of: all, omit, valid.")
        query_warmup_cfg = cfg["train"].get("query_warmup", {}) or {}
        if not isinstance(query_warmup_cfg, dict):
            raise ValueError("train.query_warmup must be a mapping.")
        self.freeze_class_queries_epochs = int(
            query_warmup_cfg.get("freeze_class_queries_epochs", 0)
        )
        if self.freeze_class_queries_epochs < 0:
            raise ValueError(
                "train.query_warmup.freeze_class_queries_epochs must be non-negative."
            )
        self._last_class_queries_trainable: bool | None = None
        self.threshold = float(cfg["threshold"].get("fixed_value", 0.6))
        self.val_outputs: list[dict[str, torch.Tensor]] = []
        self.test_outputs: list[dict[str, torch.Tensor]] = []
        self.latest_val_thresholds: dict[str, float] = {}
        self.train_loss_metric = MeanMetric()
        self.train_loss_omit_metric = MeanMetric()
        self.train_loss_weak_metric = MeanMetric()
        self.train_loss_text_coverage_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()
        self.val_loss_omit_metric = MeanMetric()
        self.val_loss_weak_metric = MeanMetric()
        self.val_loss_text_coverage_metric = MeanMetric()
        self.test_loss_metric = MeanMetric()
        self.test_loss_omit_metric = MeanMetric()
        self.test_loss_weak_metric = MeanMetric()
        self.test_loss_text_coverage_metric = MeanMetric()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.model(batch)

    def _gamma_vector_from_config(self, key: str, default: float) -> torch.Tensor:
        value = self.cfg["loss"].get(key)
        if value is None:
            return torch.full((len(CLASS_SLUGS),), float(default), dtype=torch.float32)
        if not isinstance(value, dict):
            raise ValueError(f"loss.{key} must be a mapping from class slug to gamma.")
        return torch.tensor(
            [float(value.get(slug, default)) for slug in CLASS_SLUGS],
            dtype=torch.float32,
        )

    def _tail_class_mask_from_config(self) -> torch.Tensor:
        override_tail = self.pair_weight_override_cfg.get("tail_classes")
        sampler_tail = self.cfg.get("train_sampler", {}).get("tail_classes", [])
        tail_classes = list(
            override_tail if override_tail is not None else sampler_tail
        )
        unknown = sorted(set(tail_classes) - set(CLASS_SLUGS))
        if unknown:
            raise ValueError(
                f"Unknown loss.pair_weight_override.tail_classes: {unknown}"
            )
        return torch.tensor(
            [slug in tail_classes for slug in CLASS_SLUGS], dtype=torch.bool
        )

    def _pair_weight_value(self, group: str, key: str, default: float) -> float:
        group_cfg = self.pair_weight_override_cfg.get(group, {})
        if not isinstance(group_cfg, dict):
            raise ValueError(f"loss.pair_weight_override.{group} must be a mapping.")
        return float(
            group_cfg.get(key, self.pair_weight_override_cfg.get(key, default))
        )

    def _assert_finite(self, tensor: torch.Tensor, name: str) -> None:
        if torch.isfinite(tensor).all():
            return
        bad_count = int((~torch.isfinite(tensor)).sum().detach().cpu().item())
        raise FloatingPointError(
            f"{name} contains NaN or Inf at global_step={self.global_step}; bad_values={bad_count}."
        )

    def _weak_loss_mask(self, batch: dict[str, Any]) -> torch.Tensor | None:
        if self.weak_mask_source == "all":
            return None
        if self.weak_mask_source == "omit":
            return batch["omit_masks"]
        if not bool(batch.get("has_weak_valid_masks", False)):
            raise ValueError(
                "loss.weak_mask_source='valid' requires complete weak_valid_mask_* columns in the parquet files."
            )
        return batch["weak_valid_masks"]

    def _coverage_loss_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        if not bool(batch.get("has_weak_valid_masks", False)):
            raise ValueError(
                "loss.lambda_text_coverage > 0 requires complete weak_valid_mask_* columns in the parquet files."
            )
        return batch["weak_valid_masks"]

    def _text_coverage_loss(
        self, out: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> torch.Tensor:
        if self.lambda_text_coverage <= 0.0:
            return out["omit_logits"].new_zeros(())
        if "coverage_logit_c" not in out:
            raise ValueError(
                "loss.lambda_text_coverage > 0 requires model output 'coverage_logit_c'."
            )
        self._assert_finite(out["coverage_logit_c"], "coverage_logit_c")
        coverage_mask = self._coverage_loss_mask(batch).float()
        raw_loss = F.binary_cross_entropy_with_logits(
            out["coverage_logit_c"], batch["report_labels"].float(), reduction="none"
        )
        denom = coverage_mask.sum().clamp_min(1.0)
        return (raw_loss * coverage_mask).sum() / denom

    def _omit_valid_weight(
        self, batch: dict[str, Any], logits: torch.Tensor
    ) -> torch.Tensor:
        valid_weight = batch["omit_masks"].to(device=logits.device, dtype=logits.dtype)
        if self.use_pair_weight_override:
            return valid_weight * self._configured_pair_weight(batch, logits)
        if self.use_omit_pair_weight:
            valid_weight = valid_weight * batch["omit_pair_weight"].to(
                device=logits.device, dtype=logits.dtype
            )
        if self.use_tail_class_weight:
            valid_weight = valid_weight * batch["tail_class_weight"].to(
                device=logits.device, dtype=logits.dtype
            )
        return valid_weight

    def _configured_pair_weight(
        self, batch: dict[str, Any], logits: torch.Tensor
    ) -> torch.Tensor:
        targets = batch["omit_labels"].to(device=logits.device, dtype=logits.dtype)
        weak_labels = batch["weak_labels"].to(device=logits.device, dtype=logits.dtype)
        tail_mask = self.tail_class_mask.to(device=logits.device).unsqueeze(0)
        weights = torch.ones_like(targets)
        omit_positive = targets > 0.5
        hard_negative = (targets <= 0.5) & (weak_labels > 0.5)
        weak_negative = (targets <= 0.5) & (weak_labels <= 0.5)
        for group, group_mask in (("non_tail", ~tail_mask), ("tail", tail_mask)):
            weights = torch.where(
                omit_positive & group_mask,
                torch.full_like(
                    weights, self._pair_weight_value(group, "omit_positive", 3.0)
                ),
                weights,
            )
            weights = torch.where(
                hard_negative & group_mask,
                torch.full_like(
                    weights, self._pair_weight_value(group, "hard_negative", 2.0)
                ),
                weights,
            )
            weights = torch.where(
                weak_negative & group_mask,
                torch.full_like(
                    weights, self._pair_weight_value(group, "weak_negative", 1.0)
                ),
                weights,
            )
        clean_negative_multiplier = float(
            self.pair_weight_override_cfg.get("clean_negative_multiplier", 1.0)
        )
        if clean_negative_multiplier != 1.0:
            clean_rows = (
                batch["is_clean"]
                .to(device=logits.device, dtype=torch.bool)
                .unsqueeze(1)
            )
            weights = torch.where(
                clean_rows & (targets <= 0.5),
                weights * clean_negative_multiplier,
                weights,
            )
        return weights

    def _omit_loss(
        self, out: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> torch.Tensor:
        logits = out["omit_logits"]
        targets = batch["omit_labels"].to(device=logits.device, dtype=logits.dtype)
        valid_weight = self._omit_valid_weight(batch, logits)
        if self.omit_loss_name == "masked_weighted_bce":
            raw_loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
                pos_weight=self.omit_pos_weight.to(
                    device=logits.device, dtype=logits.dtype
                ),
            )
        elif self.omit_loss_name == "masked_asymmetric":
            raw_loss = asymmetric_loss_with_logits(
                logits,
                targets,
                pos_weight=self.omit_pos_weight,
                gamma_pos=self.omit_asl_gamma_pos_vec.to(
                    device=logits.device, dtype=logits.dtype
                ),
                gamma_neg=self.omit_asl_gamma_neg_vec.to(
                    device=logits.device, dtype=logits.dtype
                ),
                eps=self.omit_asl_eps,
            )
        else:
            raise ValueError(f"Unsupported loss.omit_loss={self.omit_loss_name!r}.")
        return (raw_loss * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

    def _loss(
        self, out: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._assert_finite(out["omit_logits"], "omit_logits")
        self._assert_finite(out["weak_logits"], "weak_logits")
        loss_omit = self._omit_loss(out, batch)
        weak_loss_mask = self._weak_loss_mask(batch)
        if weak_loss_mask is not None:
            loss_weak = masked_weighted_bce_with_logits(
                out["weak_logits"],
                batch["weak_labels"],
                weak_loss_mask,
                self.weak_pos_weight,
            )
        else:
            loss_weak = weighted_bce_with_logits(
                out["weak_logits"], batch["weak_labels"], self.weak_pos_weight
            )
        loss_text_coverage = self._text_coverage_loss(out, batch)
        total = (
            loss_omit
            + self.lambda_weak * loss_weak
            + self.lambda_text_coverage * loss_text_coverage
        )
        self._assert_finite(loss_omit, "loss_omit")
        self._assert_finite(loss_weak, "loss_weak")
        self._assert_finite(loss_text_coverage, "loss_text_coverage")
        self._assert_finite(total, "loss")
        return (total, loss_omit, loss_weak, loss_text_coverage)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self(batch)
        total, loss_omit, loss_weak, loss_text_coverage = self._loss(out, batch)
        batch_size = int(batch["images"].shape[0])
        self.train_loss_metric.update(total.detach())
        self.train_loss_omit_metric.update(loss_omit.detach())
        self.train_loss_weak_metric.update(loss_weak.detach())
        self.train_loss_text_coverage_metric.update(loss_text_coverage.detach())
        self.log(
            "train/loss",
            total,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/loss_omit",
            loss_omit,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/loss_weak",
            loss_weak,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/loss_text_coverage",
            loss_text_coverage,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return total

    def _shared_eval_step(
        self, batch: dict[str, Any], storage: list[dict[str, torch.Tensor]], prefix: str
    ) -> torch.Tensor:
        out = self(batch)
        total, loss_omit, loss_weak, loss_text_coverage = self._loss(out, batch)
        batch_size = int(batch["images"].shape[0])
        weak_loss_mask = self._weak_loss_mask(batch)
        if prefix == "val":
            self.val_loss_metric.update(total.detach())
            self.val_loss_omit_metric.update(loss_omit.detach())
            self.val_loss_weak_metric.update(loss_weak.detach())
            self.val_loss_text_coverage_metric.update(loss_text_coverage.detach())
        elif prefix == "test":
            self.test_loss_metric.update(total.detach())
            self.test_loss_omit_metric.update(loss_omit.detach())
            self.test_loss_weak_metric.update(loss_weak.detach())
            self.test_loss_text_coverage_metric.update(loss_text_coverage.detach())
        self.log(
            f"{prefix}/loss",
            total,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}/loss_omit",
            loss_omit,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}/loss_weak",
            loss_weak,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}/loss_text_coverage",
            loss_text_coverage,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        coverage_prob = out.get("coverage_prob")
        if coverage_prob is None:
            coverage_prob = torch.zeros_like(out["omit_prob"])
        storage.append(
            {
                "omit_prob": out["omit_prob"].detach().cpu(),
                "weak_prob": out["weak_prob"].detach().cpu(),
                "coverage_prob": coverage_prob.detach().cpu(),
                "omit_labels": batch["omit_labels"].detach().cpu(),
                "omit_masks": batch["omit_masks"].detach().cpu(),
                "weak_labels": batch["weak_labels"].detach().cpu(),
                "report_labels": batch["report_labels"].detach().cpu(),
                "weak_masks": (
                    weak_loss_mask.detach().cpu()
                    if weak_loss_mask is not None
                    else torch.ones_like(batch["weak_labels"]).detach().cpu()
                ),
                "coverage_masks": (
                    batch["weak_valid_masks"].detach().cpu()
                    if bool(batch.get("has_weak_valid_masks", False))
                    else torch.ones_like(batch["report_labels"]).detach().cpu()
                ),
                "is_clean": batch["is_clean"].detach().cpu(),
            }
        )
        return total

    @staticmethod
    def _fmt_metric(value: float) -> str:
        if value != value:
            return "nan"
        return f"{float(value):.4f}"

    def _metric_value(self, metric: MeanMetric) -> float:
        return float(metric.compute().detach().cpu().item())

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_eval_step(batch, self.val_outputs, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_eval_step(batch, self.test_outputs, "test")

    def on_validation_epoch_start(self) -> None:
        self.val_outputs.clear()
        self.val_loss_metric.reset()
        self.val_loss_omit_metric.reset()
        self.val_loss_weak_metric.reset()
        self.val_loss_text_coverage_metric.reset()

    def on_test_epoch_start(self) -> None:
        self.test_outputs.clear()
        self.test_loss_metric.reset()
        self.test_loss_omit_metric.reset()
        self.test_loss_weak_metric.reset()
        self.test_loss_text_coverage_metric.reset()

    def _set_class_queries_trainable_for_epoch(self) -> None:
        query_bank = getattr(self.model, "query_bank", None)
        class_queries = getattr(query_bank, "class_queries", None)
        if class_queries is None:
            return
        trainable = int(self.current_epoch) >= self.freeze_class_queries_epochs
        class_queries.requires_grad = trainable
        image_query_residual = getattr(query_bank, "image_query_residual", None)
        residual_trainable = False
        if image_query_residual is not None:
            residual_trainable = trainable and bool(
                getattr(query_bank, "image_residual_trainable", True)
            )
            image_query_residual.requires_grad = residual_trainable
        if self._last_class_queries_trainable == trainable:
            return
        self._last_class_queries_trainable = trainable
        if self.trainer is None or self.trainer.is_global_zero:
            LOGGER.info(
                "Query warm-up | epoch=%d freeze_class_queries_epochs=%d class_queries_trainable=%s image_query_residual_trainable=%s",
                int(self.current_epoch),
                self.freeze_class_queries_epochs,
                trainable,
                residual_trainable,
            )

    def on_train_epoch_start(self) -> None:
        self._set_class_queries_trainable_for_epoch()

    def on_train_epoch_end(self) -> None:
        if self.trainer is None or self.trainer.is_global_zero:
            LOGGER.info(
                "TRAIN summary | epoch=%d loss=%s loss_omit=%s loss_weak=%s loss_text_coverage=%s",
                int(self.current_epoch),
                self._fmt_metric(self._metric_value(self.train_loss_metric)),
                self._fmt_metric(self._metric_value(self.train_loss_omit_metric)),
                self._fmt_metric(self._metric_value(self.train_loss_weak_metric)),
                self._fmt_metric(
                    self._metric_value(self.train_loss_text_coverage_metric)
                ),
            )
        self.train_loss_metric.reset()
        self.train_loss_omit_metric.reset()
        self.train_loss_weak_metric.reset()
        self.train_loss_text_coverage_metric.reset()

    def _log_epoch_metrics(
        self, outputs: list[dict[str, torch.Tensor]], prefix: str
    ) -> None:
        if not outputs:
            return
        omit_prob = torch.cat([x["omit_prob"] for x in outputs], dim=0)
        weak_prob = torch.cat([x["weak_prob"] for x in outputs], dim=0)
        coverage_prob = torch.cat([x["coverage_prob"] for x in outputs], dim=0)
        omit_labels = torch.cat([x["omit_labels"] for x in outputs], dim=0)
        omit_masks = torch.cat([x["omit_masks"] for x in outputs], dim=0)
        weak_labels = torch.cat([x["weak_labels"] for x in outputs], dim=0)
        report_labels = torch.cat([x["report_labels"] for x in outputs], dim=0)
        weak_masks = (
            torch.cat([x["weak_masks"] for x in outputs], dim=0)
            if self.weak_mask_source != "all"
            else None
        )
        coverage_masks = torch.cat([x["coverage_masks"] for x in outputs], dim=0)
        is_clean = torch.cat([x["is_clean"] for x in outputs], dim=0)
        metrics = compute_omission_metrics(
            omit_prob,
            omit_labels,
            omit_masks,
            is_clean=is_clean,
            threshold=self.threshold,
            prefix=prefix,
        )
        metrics.update(
            compute_weak_metrics(
                weak_prob, weak_labels, weak_masks=weak_masks, prefix=prefix
            )
        )
        metrics.update(
            compute_text_coverage_metrics(
                coverage_prob,
                report_labels,
                coverage_masks=coverage_masks,
                prefix=prefix,
            )
        )
        self.log_dict(metrics, prog_bar=False, sync_dist=True)
        if self.trainer is not None and self.trainer.sanity_checking:
            return
        loss_metric = self.val_loss_metric if prefix == "val" else self.test_loss_metric
        loss_omit_metric = (
            self.val_loss_omit_metric if prefix == "val" else self.test_loss_omit_metric
        )
        loss_weak_metric = (
            self.val_loss_weak_metric if prefix == "val" else self.test_loss_weak_metric
        )
        loss_text_coverage_metric = (
            self.val_loss_text_coverage_metric
            if prefix == "val"
            else self.test_loss_text_coverage_metric
        )
        if self.trainer is None or self.trainer.is_global_zero:
            LOGGER.info(
                "%s summary | epoch=%d loss=%s loss_omit=%s loss_weak=%s loss_text_coverage=%s omit_macro_auprc=%s omit_macro_auroc=%s omit_macro_f1=%s omit_micro_auprc=%s report_auprc=%s report_auroc=%s report_f1=%s weak_macro_auprc=%s weak_macro_auroc=%s text_coverage_macro_auprc=%s text_coverage_macro_auroc=%s clean_false_alarm_masked=%s",
                prefix.upper(),
                int(self.current_epoch),
                self._fmt_metric(self._metric_value(loss_metric)),
                self._fmt_metric(self._metric_value(loss_omit_metric)),
                self._fmt_metric(self._metric_value(loss_weak_metric)),
                self._fmt_metric(self._metric_value(loss_text_coverage_metric)),
                self._fmt_metric(
                    metrics.get(f"{prefix}/omit_macro_auprc", float("nan"))
                ),
                self._fmt_metric(
                    metrics.get(f"{prefix}/omit_macro_auroc", float("nan"))
                ),
                self._fmt_metric(metrics.get(f"{prefix}/omit_macro_f1", float("nan"))),
                self._fmt_metric(
                    metrics.get(f"{prefix}/omit_micro_auprc", float("nan"))
                ),
                self._fmt_metric(metrics.get(f"{prefix}/report_auprc", float("nan"))),
                self._fmt_metric(metrics.get(f"{prefix}/report_auroc", float("nan"))),
                self._fmt_metric(metrics.get(f"{prefix}/report_f1", float("nan"))),
                self._fmt_metric(
                    metrics.get(f"{prefix}/weak_macro_auprc", float("nan"))
                ),
                self._fmt_metric(
                    metrics.get(f"{prefix}/weak_macro_auroc", float("nan"))
                ),
                self._fmt_metric(
                    metrics.get(f"{prefix}/text_coverage_macro_auprc", float("nan"))
                ),
                self._fmt_metric(
                    metrics.get(f"{prefix}/text_coverage_macro_auroc", float("nan"))
                ),
                self._fmt_metric(
                    metrics.get(
                        f"{prefix}/clean_false_alarm_masked@{self.threshold:g}",
                        float("nan"),
                    )
                ),
            )
        if prefix == "val":
            self.latest_val_thresholds = compute_val_thresholds(
                omit_prob, omit_labels, omit_masks
            )
            self.val_loss_metric.reset()
            self.val_loss_omit_metric.reset()
            self.val_loss_weak_metric.reset()
            self.val_loss_text_coverage_metric.reset()
        elif prefix == "test":
            self.test_loss_metric.reset()
            self.test_loss_omit_metric.reset()
            self.test_loss_weak_metric.reset()
            self.test_loss_text_coverage_metric.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics(self.val_outputs, "val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_outputs, "test")

    def configure_optimizers(self) -> Any:
        train_cfg = self.cfg["train"]
        weight_decay = float(train_cfg.get("weight_decay", 0.0001))
        new_params = []
        image_params = []
        text_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("image_encoder."):
                image_params.append(param)
            elif name.startswith("text_encoder."):
                text_params.append(param)
            else:
                new_params.append(param)
        param_groups = []
        if new_params:
            param_groups.append(
                {
                    "params": new_params,
                    "lr": float(train_cfg.get("lr_new_modules", 0.0001)),
                }
            )
        if image_params:
            param_groups.append(
                {
                    "params": image_params,
                    "lr": float(train_cfg.get("lr_image_encoder", 1e-05)),
                }
            )
        if text_params:
            param_groups.append(
                {
                    "params": text_params,
                    "lr": float(train_cfg.get("lr_text_encoder", 5e-06)),
                }
            )
        log_optimizer_group_summary(new_params, image_params, text_params)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        if train_cfg.get("scheduler", "cosine") != "cosine":
            return optimizer
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.05)))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def save_latest_thresholds(module: LitOmissionModule, output_dir: str | Path) -> None:
    if not module.latest_val_thresholds:
        return
    from src.utils.io import save_json

    save_json(module.latest_val_thresholds, Path(output_dir) / "thresholds.json")
