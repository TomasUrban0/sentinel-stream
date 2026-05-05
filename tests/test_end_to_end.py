"""End-to-end test: synthetic SKAB-shaped data -> streaming features -> detector.

The PySpark batch pipeline is exercised separately by the training script. Here
we feed the same kind of multivariate signal directly into the streaming
transformer to keep the test fast (no Spark, no real download), while still
validating that the feature vector flows through to the detector unchanged.
"""

from datetime import UTC, datetime, timedelta

import numpy as np

from sentinel_stream.features.engineering import BASE_FEATURES
from sentinel_stream.features.transformer import StreamingFeatureTransformer
from sentinel_stream.models.isolation_forest import IsolationForestDetector


def test_full_pipeline_flags_injected_anomalies():
    rng = np.random.default_rng(11)
    n = 2000
    base = rng.normal(loc=0.0, scale=0.3, size=(n, len(BASE_FEATURES)))

    labels = np.zeros(n, dtype=int)
    anomaly_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    base[anomaly_idx] += rng.normal(loc=4.0, scale=0.5, size=base[anomaly_idx].shape)
    labels[anomaly_idx] = 1

    transformer = StreamingFeatureTransformer()
    base_ts = datetime(2025, 1, 1, tzinfo=UTC)

    feature_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    for i, row in enumerate(base):
        record = dict(zip(BASE_FEATURES, row.tolist(), strict=True))
        transformer.push(record, timestamp=base_ts + timedelta(seconds=i))
        vec = transformer.transform()
        if vec is not None:
            feature_rows.append(vec)
            label_rows.append(int(labels[i]))

    x = np.vstack(feature_rows)
    y = np.array(label_rows)
    assert x.shape[0] > 1500
    assert y.sum() > 0

    detector = IsolationForestDetector(n_estimators=80, contamination=0.05, random_state=0)
    detector.fit(x[y == 0])

    scores = detector.score(x)
    assert scores[y == 1].mean() > scores[y == 0].mean()

    flags = (scores >= np.quantile(scores, 0.95)).astype(int)
    true_positives = int(((flags == 1) & (y == 1)).sum())
    assert true_positives > 0
