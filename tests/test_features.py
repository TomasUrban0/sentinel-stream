import math

from sentinel_stream.features.engineering import ALL_FEATURE_COLUMNS, feature_columns
from sentinel_stream.features.transformer import StreamingFeatureTransformer


def test_feature_columns_are_deterministic():
    cols = feature_columns()
    assert cols == list(ALL_FEATURE_COLUMNS)
    assert len(cols) == 10


def test_streaming_transformer_emits_engineered_features():
    t = StreamingFeatureTransformer()
    record = {
        "type": "M",
        "air_temperature_k": 298.5,
        "process_temperature_k": 308.5,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 100.0,
    }
    vec = t.transform(record)
    assert vec.shape == (10,)

    by_name = dict(zip(t.feature_columns, vec.tolist(), strict=True))
    assert by_name["type_ordinal"] == 2.0
    assert by_name["temperature_delta_k"] == 10.0
    expected_power = 40.0 * 1500.0 * (2 * math.pi / 60.0)
    assert abs(by_name["mechanical_power_w"] - expected_power) < 1e-3
    assert by_name["wear_torque_proxy"] == 40.0 * 100.0
    assert by_name["wear_speed_proxy"] == 1500.0 * 100.0


def test_streaming_transformer_handles_unknown_type_safely():
    t = StreamingFeatureTransformer()
    record = {
        "type": "Z",
        "air_temperature_k": 298.0,
        "process_temperature_k": 309.0,
        "rotational_speed_rpm": 1400.0,
        "torque_nm": 30.0,
        "tool_wear_min": 50.0,
    }
    vec = t.transform(record)
    assert vec[t.feature_columns.index("type_ordinal")] == 1.0
