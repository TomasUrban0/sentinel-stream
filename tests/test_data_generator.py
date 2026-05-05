from sentinel_stream.data.generator import GeneratorConfig, generate


def test_generator_shape_and_columns():
    df = generate(GeneratorConfig(rows=1000, anomaly_rate=0.02, seed=0))
    assert len(df) == 1000
    assert {"timestamp", "temperature", "pressure", "vibration", "humidity", "is_anomaly"} <= set(
        df.columns
    )


def test_generator_injects_anomalies():
    df = generate(GeneratorConfig(rows=2000, anomaly_rate=0.05, seed=0))
    # Drift anomalies expand the count beyond the seed rate, so we check a lower bound.
    assert df["is_anomaly"].sum() >= int(2000 * 0.05)
    assert df["is_anomaly"].sum() < len(df)


def test_generator_is_deterministic():
    a = generate(GeneratorConfig(rows=500, seed=123))
    b = generate(GeneratorConfig(rows=500, seed=123))
    assert (a["temperature"].to_numpy() == b["temperature"].to_numpy()).all()
