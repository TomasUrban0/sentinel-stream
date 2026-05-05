"""Evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute precision/recall/F1/AUC for a labeled subset."""
    preds = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
    return metrics
