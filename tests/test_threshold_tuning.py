"""Tests for the F1-maximising threshold search."""

import numpy as np

from sentinel_stream.models.evaluation import best_f1_threshold, evaluate


def test_best_f1_threshold_perfectly_separable():
    rng = np.random.default_rng(0)
    normal_scores = rng.uniform(0.0, 0.4, size=100)
    anomaly_scores = rng.uniform(0.6, 1.0, size=100)
    scores = np.concatenate([normal_scores, anomaly_scores])
    labels = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)

    thr, f1 = best_f1_threshold(scores, labels)
    assert f1 == 1.0
    assert 0.4 <= thr <= 0.6
    metrics = evaluate(scores, labels, thr)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_best_f1_threshold_beats_default_on_overlapping_classes():
    rng = np.random.default_rng(0)
    n = 1000
    labels = rng.integers(0, 2, size=n)
    scores = rng.normal(loc=labels.astype(float), scale=1.0)

    default_threshold = float(np.percentile(scores[labels == 0], 99))
    default_f1 = evaluate(scores, labels, default_threshold)["f1"]

    thr, f1 = best_f1_threshold(scores, labels)
    assert f1 >= default_f1
    assert evaluate(scores, labels, thr)["f1"] == f1
