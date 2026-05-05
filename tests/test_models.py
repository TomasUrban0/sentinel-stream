import numpy as np

from sentinel_stream.models.evaluation import evaluate
from sentinel_stream.models.isolation_forest import IsolationForestDetector


def test_isolation_forest_fit_and_score(tmp_path):
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(500, 8)).astype(np.float32)

    detector = IsolationForestDetector(n_estimators=50, contamination=0.05)
    detector.fit(x_train)

    x_eval = np.vstack([rng.normal(size=(50, 8)), rng.normal(loc=10.0, size=(10, 8))])
    scores = detector.score(x_eval)
    assert scores.shape == (60,)
    # The shifted block should score higher on average than the in-distribution block.
    assert scores[50:].mean() > scores[:50].mean()

    detector.save(str(tmp_path))
    loaded = IsolationForestDetector.load(str(tmp_path))
    np.testing.assert_allclose(loaded.score(x_eval), scores, rtol=1e-6)


def test_evaluate_metrics():
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.05])
    labels = np.array([0, 0, 1, 1, 0])
    metrics = evaluate(scores, labels, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
