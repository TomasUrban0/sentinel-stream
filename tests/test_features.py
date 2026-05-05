from datetime import datetime, timedelta

from sentinel_stream.features.engineering import BASE_FEATURES, feature_columns
from sentinel_stream.features.transformer import StreamingFeatureTransformer


def test_feature_columns_deterministic_order_default():
    # Default config disables spectral features after they degraded both
    # detectors on SKAB; with no spectral channels the count returns to 82.
    cols = feature_columns(spectral_channels=())
    assert len(cols) == 82
    assert cols[-2:] == ["hour", "dayofweek"]


def test_feature_columns_with_spectral():
    cols = feature_columns(spectral_channels=("accelerometer_1_rms",), spectral_bands=4)
    assert len(cols) == 88  # 82 + 6 spectral features for one channel
    assert any(c.endswith("_spec_centroid") for c in cols)


def _record(value: float = 1.0) -> dict[str, float]:
    return {feat: value for feat in BASE_FEATURES}


def test_streaming_transformer_warmup():
    t = StreamingFeatureTransformer()
    for _ in range(10):
        t.push(_record())
    assert t.transform() is None


def test_streaming_transformer_returns_vector_after_warmup():
    t = StreamingFeatureTransformer(spectral_channels=())
    ts = datetime(2025, 1, 1, 12, 0, 0)
    for i in range(40):
        t.push(_record(value=float(i)), timestamp=ts + timedelta(seconds=i))
    vec = t.transform()
    assert vec is not None
    assert vec.shape == (len(t.feature_columns),)


def test_streaming_transformer_with_spectral_emits_extra_columns():
    t = StreamingFeatureTransformer(spectral_channels=("accelerometer_1_rms",))
    ts = datetime(2025, 1, 1, 12, 0, 0)
    for i in range(80):
        t.push(_record(value=float(i)), timestamp=ts + timedelta(seconds=i))
    vec = t.transform()
    assert vec is not None
    assert vec.shape == (88,)
