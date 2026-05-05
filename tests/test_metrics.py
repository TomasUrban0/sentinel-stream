from sentinel_stream.monitoring.metrics import MetricsRegistry


def test_metrics_registry_aggregates():
    registry = MetricsRegistry()
    for i in range(100):
        registry.record(latency_ms=float(i), is_anomaly=(i % 10 == 0))
    snap = registry.snapshot()
    assert snap["total_predictions"] == 100
    assert snap["total_anomalies"] == 10
    assert snap["p50_ms"] <= snap["p95_ms"] <= snap["p99_ms"]
