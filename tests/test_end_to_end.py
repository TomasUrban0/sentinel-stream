"""End-to-end smoke test: synthetic AI4I-shaped record -> streaming transformer
-> XGBoost classifier -> probability."""

import numpy as np

from sentinel_stream.data.ai4i_loader import BASE_FEATURES
from sentinel_stream.features.engineering import ALL_FEATURE_COLUMNS
from sentinel_stream.features.transformer import StreamingFeatureTransformer
from sentinel_stream.models.xgboost_classifier import XGBoostClassifier


def _build_synthetic(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for _ in range(n):
        is_failure = rng.random() < 0.15
        record = {
            "type": rng.choice(["L", "M", "H"]),
            "air_temperature_k": rng.normal(298.0, 1.5),
            "process_temperature_k": rng.normal(308.0, 1.5)
            + (3.0 if is_failure else 0.0),
            "rotational_speed_rpm": rng.normal(1500, 100)
            - (200 if is_failure else 0.0),
            "torque_nm": rng.normal(40, 6) + (10 if is_failure else 0.0),
            "tool_wear_min": rng.uniform(0, 250) + (50 if is_failure else 0.0),
        }
        rows.append(record)
        labels.append(int(is_failure))
    return rows, np.array(labels)


def test_full_pipeline_predicts_synthetic_failures():
    rows, y = _build_synthetic()
    transformer = StreamingFeatureTransformer()
    x = np.vstack([transformer.transform(r) for r in rows])
    assert x.shape == (len(rows), len(ALL_FEATURE_COLUMNS))

    clf = XGBoostClassifier(n_estimators=80, max_depth=4)
    clf.fit(x, y, feature_names=transformer.feature_columns)
    proba = clf.predict_proba(x)

    assert proba[y == 1].mean() > proba[y == 0].mean() + 0.2

    # Round-trip through a per-record transformer call to confirm the served
    # feature vector still matches a row in the training matrix.
    spot_check = transformer.transform({**rows[0], **{k: rows[0][k] for k in BASE_FEATURES}})
    assert np.allclose(spot_check, x[0])
