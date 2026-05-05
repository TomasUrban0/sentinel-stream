from datetime import datetime, timedelta

from sentinel_stream.features.engineering import BASE_FEATURES, feature_columns
from sentinel_stream.features.transformer import StreamingFeatureTransformer


def test_feature_columns_deterministic_order():
    cols = feature_columns()
    # 8 base features × (4 stats × 2 windows + 2 lags) = 8 × 10 = 80, plus hour & dayofweek.
    assert len(cols) == 82
    assert cols[-2:] == ["hour", "dayofweek"]
    assert all(any(col.startswith(b + "_") for b in BASE_FEATURES) for col in cols[:-2])


def _record(value: float = 1.0) -> dict[str, float]:
    return {feat: value for feat in BASE_FEATURES}


def test_streaming_transformer_warmup():
    t = StreamingFeatureTransformer()
    for _ in range(10):
        t.push(_record())
    assert t.transform() is None


def test_streaming_transformer_returns_vector_after_warmup():
    t = StreamingFeatureTransformer()
    ts = datetime(2025, 1, 1, 12, 0, 0)
    for i in range(40):
        t.push(_record(value=float(i)), timestamp=ts + timedelta(seconds=i))
    vec = t.transform()
    assert vec is not None
    assert vec.shape == (len(t.feature_columns),)
