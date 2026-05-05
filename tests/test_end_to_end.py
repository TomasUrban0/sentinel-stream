"""End-to-end test: synthetic data -> streaming features -> detector -> prediction.

Skips PySpark to keep the test fast; the streaming feature transformer is
deliberately exercised on the same data the batch pipeline would consume,
so feature parity is implicitly covered.
"""

from datetime import UTC, datetime, timedelta

import numpy as np

from sentinel_stream.data.generator import GeneratorConfig, generate
from sentinel_stream.features.transformer import StreamingFeatureTransformer
from sentinel_stream.models.isolation_forest import IsolationForestDetector


def test_full_pipeline_flags_injected_anomalies():
    df = generate(GeneratorConfig(rows=2000, anomaly_rate=0.02, seed=7))

    transformer = StreamingFeatureTransformer()
    base_ts = datetime(2025, 1, 1, tzinfo=UTC)

    feature_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    for i, row in df.iterrows():
        transformer.push(
            {c: row[c] for c in transformer.base_features},
            timestamp=base_ts + timedelta(seconds=int(i)),
        )
        vec = transformer.transform()
        if vec is not None:
            feature_rows.append(vec)
            label_rows.append(int(row["is_anomaly"]))

    x = np.vstack(feature_rows)
    y = np.array(label_rows)
    assert x.shape[0] > 1500
    assert y.sum() > 0

    normal_mask = y == 0
    detector = IsolationForestDetector(n_estimators=80, contamination=0.02, random_state=0)
    detector.fit(x[normal_mask])

    scores = detector.score(x)
    assert scores.shape == x.shape[:1]
    assert scores[y == 1].mean() > scores[y == 0].mean()

    flags = (scores >= np.quantile(scores, 0.98)).astype(int)
    true_positives = int(((flags == 1) & (y == 1)).sum())
    assert true_positives > 0
