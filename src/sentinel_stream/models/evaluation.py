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


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Sweep candidate thresholds and return the one that maximises F1.

    Returns (threshold, f1). Candidates are taken from the unique sorted score
    values, so the search is exhaustive without being quadratic in n.
    """
    if len(np.unique(labels)) < 2:
        raise ValueError("Cannot tune a threshold without both classes present.")
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    total_pos = int(sorted_labels.sum())
    cum_pos = np.cumsum(sorted_labels)
    n = len(scores)

    best_thr = float(sorted_scores[0])
    best_f1 = 0.0
    # Threshold = sorted_scores[i] means we predict positive for indices > i.
    # Walk through every unique cut point.
    for i in range(n):
        if i + 1 < n and sorted_scores[i] == sorted_scores[i + 1]:
            continue
        tp = total_pos - cum_pos[i] if i < n else 0
        fp = (n - 1 - i) - tp if i < n - 1 else 0
        fn = total_pos - tp
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        if f1 > best_f1:
            best_f1 = float(f1)
            # Use a midpoint between this score and the next so equal-score
            # ties fall on the positive side consistently.
            if i + 1 < n:
                best_thr = float((sorted_scores[i] + sorted_scores[i + 1]) / 2)
            else:
                best_thr = float(sorted_scores[i])
    return best_thr, best_f1
