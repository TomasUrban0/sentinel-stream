from datetime import datetime, timedelta

from sentinel_stream.features.engineering import feature_columns
from sentinel_stream.features.transformer import StreamingFeatureTransformer


def test_feature_columns_deterministic_order():
    cols = feature_columns()
    # 4 base features × (4 stats × 2 windows + 2 lags) = 4 × 10 = 40, plus hour & dayofweek.
    assert len(cols) == 42
    assert cols[-2:] == ["hour", "dayofweek"]


def test_streaming_transformer_warmup():
    t = StreamingFeatureTransformer()
    base_record = {"temperature": 70.0, "pressure": 101.0, "vibration": 0.1, "humidity": 45.0}
    # Below the largest rolling window we should get None.
    for _ in range(10):
        t.push(base_record)
    assert t.transform() is None


def test_streaming_transformer_returns_vector_after_warmup():
    t = StreamingFeatureTransformer()
    record = {"temperature": 70.0, "pressure": 101.0, "vibration": 0.1, "humidity": 45.0}
    ts = datetime(2025, 1, 1, 12, 0, 0)
    for i in range(40):
        t.push(record, timestamp=ts + timedelta(seconds=i))
    vec = t.transform()
    assert vec is not None
    assert vec.shape == (len(t.feature_columns),)
