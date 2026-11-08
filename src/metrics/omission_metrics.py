from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.data.parquet_dataset import CLASSES, CLASS_SLUGS


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _has_pos_and_neg(labels: np.ndarray) -> bool:
    labels = labels.astype(int)
    return bool((labels == 1).any() and (labels == 0).any())


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or not _has_pos_and_neg(labels):
        return float("nan")
    return float(average_precision_score(labels, scores))


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or not _has_pos_and_neg(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_f1(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    if labels.size == 0 or not _has_pos_and_neg(labels):
        return float("nan")
    preds = (scores >= threshold).astype(int)
    return float(f1_score(labels, preds, zero_division=0))


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def recall_at_precision(
    labels: np.ndarray, scores: np.ndarray, target_precision: float
) -> float:
    if labels.size == 0 or not _has_pos_and_neg(labels):
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    valid = precision >= target_precision
    if not valid.any():
        return 0.0
    return float(np.max(recall[valid]))


def report_scores_and_labels(
    omit_probs: np.ndarray,
    omit_labels: np.ndarray,
    omit_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    has_valid = omit_masks.astype(bool).any(axis=1)
    if not has_valid.any():
        return np.asarray([]), np.asarray([])
    masked_scores = np.where(omit_masks.astype(bool), omit_probs, -np.inf)
    scores = np.max(masked_scores[has_valid], axis=1)
    labels = (
        ((omit_labels[has_valid] == 1) & (omit_masks[has_valid] == 1))
        .any(axis=1)
        .astype(int)
    )
    return scores, labels


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or not _has_pos_and_neg(labels):
        return 0.6
    thresholds = np.linspace(0.0, 1.0, 1001)
    f1s = [
        f1_score(labels, scores >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(f1s))])


def threshold_for_precision(
    labels: np.ndarray, scores: np.ndarray, target_precision: float
) -> float:
    if labels.size == 0 or not _has_pos_and_neg(labels):
        return 0.6
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    best_threshold = 1.0
    best_recall = -1.0
    for idx, threshold in enumerate(thresholds):
        if precision[idx] >= target_precision and recall[idx] > best_recall:
            best_recall = float(recall[idx])
            best_threshold = float(threshold)
    return best_threshold


def compute_val_thresholds(
    omit_probs: torch.Tensor | np.ndarray,
    omit_labels: torch.Tensor | np.ndarray,
    omit_masks: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    probs = _to_numpy(omit_probs)
    labels = _to_numpy(omit_labels)
    masks = _to_numpy(omit_masks)
    report_scores, report_labels = report_scores_and_labels(probs, labels, masks)
    return {
        "val_best_f1_threshold": best_f1_threshold(report_labels, report_scores),
        "val_precision80_threshold": threshold_for_precision(
            report_labels, report_scores, 0.80
        ),
        "val_precision90_threshold": threshold_for_precision(
            report_labels, report_scores, 0.90
        ),
    }


def compute_omission_metrics(
    omit_probs: torch.Tensor | np.ndarray,
    omit_labels: torch.Tensor | np.ndarray,
    omit_masks: torch.Tensor | np.ndarray,
    is_clean: torch.Tensor | np.ndarray | None = None,
    threshold: float = 0.6,
    prefix: str = "val",
) -> dict[str, float]:
    probs = _to_numpy(omit_probs)
    labels = _to_numpy(omit_labels).astype(int)
    masks = _to_numpy(omit_masks).astype(int)
    metrics: dict[str, float] = {}

    per_auprc: list[float] = []
    per_auroc: list[float] = []
    per_f1: list[float] = []
    per_recall80: list[float] = []
    per_recall90: list[float] = []

    for idx, slug in enumerate(CLASS_SLUGS):
        valid = masks[:, idx] == 1
        y = labels[valid, idx]
        s = probs[valid, idx]
        auprc = _safe_auprc(y, s)
        auroc = _safe_auroc(y, s)
        f1 = _safe_f1(y, s, threshold)
        r80 = recall_at_precision(y, s, 0.80)
        r90 = recall_at_precision(y, s, 0.90)
        metrics[f"{prefix}/omit_{slug}_auprc"] = auprc
        metrics[f"{prefix}/omit_{slug}_auroc"] = auroc
        metrics[f"{prefix}/omit_{slug}_f1"] = f1
        metrics[f"{prefix}/omit_{slug}_recall@precision80"] = r80
        metrics[f"{prefix}/omit_{slug}_recall@precision90"] = r90
        per_auprc.append(auprc)
        per_auroc.append(auroc)
        per_f1.append(f1)
        per_recall80.append(r80)
        per_recall90.append(r90)

    flat_valid = masks.astype(bool)
    flat_y = labels[flat_valid]
    flat_s = probs[flat_valid]
    metrics[f"{prefix}/omit_macro_auprc"] = _nanmean(per_auprc)
    metrics[f"{prefix}/omit_macro_auroc"] = _nanmean(per_auroc)
    metrics[f"{prefix}/omit_macro_f1"] = _nanmean(per_f1)
    metrics[f"{prefix}/omit_micro_auprc"] = _safe_auprc(flat_y, flat_s)
    metrics[f"{prefix}/omit_micro_f1"] = _safe_f1(flat_y, flat_s, threshold)
    metrics[f"{prefix}/omit_macro_recall@precision80"] = _nanmean(per_recall80)
    metrics[f"{prefix}/omit_macro_recall@precision90"] = _nanmean(per_recall90)

    has_positive = ((labels == 1) & (masks == 1)).any(axis=1)
    if has_positive.any():
        masked_scores = np.where(masks == 1, probs, -np.inf)
        order = np.argsort(-masked_scores, axis=1)
        positives = (labels == 1) & (masks == 1)
        top1_hit = [positives[i, order[i, :1]].any() for i in np.where(has_positive)[0]]
        top3_hit = [positives[i, order[i, :3]].any() for i in np.where(has_positive)[0]]
        metrics[f"{prefix}/omit_top1_recall"] = float(np.mean(top1_hit))
        metrics[f"{prefix}/omit_top3_recall"] = float(np.mean(top3_hit))
    else:
        metrics[f"{prefix}/omit_top1_recall"] = float("nan")
        metrics[f"{prefix}/omit_top3_recall"] = float("nan")

    report_scores, report_labels = report_scores_and_labels(probs, labels, masks)
    metrics[f"{prefix}/report_auprc"] = _safe_auprc(report_labels, report_scores)
    metrics[f"{prefix}/report_auroc"] = _safe_auroc(report_labels, report_scores)
    metrics[f"{prefix}/report_f1"] = _safe_f1(report_labels, report_scores, threshold)

    if is_clean is not None:
        clean = _to_numpy(is_clean).astype(int) == 1
        if clean.any():
            metrics[f"{prefix}/clean_false_alarm_all_classes@{threshold:g}"] = float(
                (probs[clean] > threshold).any(axis=1).mean()
            )
            masked_alarm = ((probs > threshold) & (masks == 1))[clean].any(axis=1)
            metrics[f"{prefix}/clean_false_alarm_masked@{threshold:g}"] = float(
                masked_alarm.mean()
            )
        else:
            metrics[f"{prefix}/clean_false_alarm_all_classes@{threshold:g}"] = float(
                "nan"
            )
            metrics[f"{prefix}/clean_false_alarm_masked@{threshold:g}"] = float("nan")

    return metrics


def compute_weak_metrics(
    weak_probs: torch.Tensor | np.ndarray,
    weak_labels: torch.Tensor | np.ndarray,
    weak_masks: torch.Tensor | np.ndarray | None = None,
    prefix: str = "val",
) -> dict[str, float]:
    probs = _to_numpy(weak_probs)
    labels = _to_numpy(weak_labels).astype(int)
    masks = _to_numpy(weak_masks).astype(int) if weak_masks is not None else None
    metrics: dict[str, float] = {}
    auprcs: list[float] = []
    aurocs: list[float] = []
    for idx, slug in enumerate(CLASS_SLUGS):
        if masks is not None:
            valid = masks[:, idx] == 1
            y = labels[valid, idx]
            s = probs[valid, idx]
        else:
            y = labels[:, idx]
            s = probs[:, idx]
        auprc = _safe_auprc(y, s)
        auroc = _safe_auroc(y, s)
        metrics[f"{prefix}/weak_{slug}_auprc"] = auprc
        metrics[f"{prefix}/weak_{slug}_auroc"] = auroc
        auprcs.append(auprc)
        aurocs.append(auroc)
    metrics[f"{prefix}/weak_macro_auprc"] = _nanmean(auprcs)
    metrics[f"{prefix}/weak_macro_auroc"] = _nanmean(aurocs)
    return metrics


def compute_text_coverage_metrics(
    coverage_probs: torch.Tensor | np.ndarray,
    coverage_labels: torch.Tensor | np.ndarray,
    coverage_masks: torch.Tensor | np.ndarray | None = None,
    prefix: str = "val",
) -> dict[str, float]:
    probs = _to_numpy(coverage_probs)
    labels = _to_numpy(coverage_labels).astype(int)
    masks = (
        _to_numpy(coverage_masks).astype(int) if coverage_masks is not None else None
    )
    metrics: dict[str, float] = {}
    auprcs: list[float] = []
    aurocs: list[float] = []
    for idx, slug in enumerate(CLASS_SLUGS):
        if masks is not None:
            valid = masks[:, idx] == 1
            y = labels[valid, idx]
            s = probs[valid, idx]
        else:
            y = labels[:, idx]
            s = probs[:, idx]
        auprc = _safe_auprc(y, s)
        auroc = _safe_auroc(y, s)
        metrics[f"{prefix}/text_coverage_{slug}_auprc"] = auprc
        metrics[f"{prefix}/text_coverage_{slug}_auroc"] = auroc
        auprcs.append(auprc)
        aurocs.append(auroc)
    metrics[f"{prefix}/text_coverage_macro_auprc"] = _nanmean(auprcs)
    metrics[f"{prefix}/text_coverage_macro_auroc"] = _nanmean(aurocs)
    return metrics


def topk_omission_strings(probs: np.ndarray, k: int = 3) -> tuple[list[str], list[str]]:
    names: list[str] = []
    scores: list[str] = []
    order = np.argsort(-probs)[:k]
    for idx in order:
        names.append(CLASSES[int(idx)])
        scores.append(f"{float(probs[int(idx)]):.6f}")
    return names, scores
