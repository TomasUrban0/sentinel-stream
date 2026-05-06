"""Smoke tests for the supervised classifiers."""

import numpy as np
import pytest

from sentinel_stream.models.evaluation import evaluate
from sentinel_stream.models.xgboost_classifier import XGBoostClassifier


def _toy_dataset(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    x_neg = rng.normal(loc=0.0, scale=1.0, size=(int(n * 0.95), 8))
    x_pos = rng.normal(loc=2.0, scale=1.0, size=(n - len(x_neg), 8))
    x = np.vstack([x_neg, x_pos]).astype(np.float32)
    y = np.concatenate([np.zeros(len(x_neg)), np.ones(len(x_pos))]).astype(int)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def test_xgboost_fits_and_separates_classes(tmp_path):
    x, y = _toy_dataset()
    clf = XGBoostClassifier(n_estimators=80, max_depth=4)
    clf.fit(x, y, feature_names=[f"f{i}" for i in range(8)])

    proba = clf.predict_proba(x)
    assert proba.shape == (len(y),)
    metrics = evaluate(proba, y, threshold=0.5)
    assert metrics["roc_auc"] > 0.9

    clf.save(str(tmp_path))
    loaded = XGBoostClassifier.load(str(tmp_path))
    np.testing.assert_allclose(loaded.predict_proba(x), proba, rtol=1e-5)


def test_keras_classifier_smoke(tmp_path):
    pytest.importorskip("tensorflow")
    from sentinel_stream.models.keras_classifier import KerasClassifier

    x, y = _toy_dataset(n=800)
    clf = KerasClassifier(hidden_layers=[16, 8], epochs=40, batch_size=64, dropout=0.0)
    clf.fit(x, y)
    proba = clf.predict_proba(x)
    assert proba.shape == (len(y),)
    # ROC-AUC is symmetric; we only need to confirm the model is not
    # essentially random in either direction.
    auc = evaluate(proba, y, 0.5)["roc_auc"]
    assert max(auc, 1 - auc) > 0.85
