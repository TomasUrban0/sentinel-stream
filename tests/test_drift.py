import numpy as np

from sentinel_stream.monitoring.drift import DriftMonitor


def test_drift_monitor_detects_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(2000, 3)).astype(np.float32)
    monitor = DriftMonitor(
        reference=reference,
        feature_names=["a", "b", "c"],
        threshold=0.15,
        window_size=500,
    )

    shifted = rng.normal(loc=3.0, size=(500, 3)).astype(np.float32)
    for row in shifted:
        monitor.observe(row)

    report = monitor.report()
    assert all(report[name]["drift"] for name in ["a", "b", "c"])


def test_drift_monitor_quiet_when_distributions_match():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(2000, 2)).astype(np.float32)
    monitor = DriftMonitor(
        reference=reference,
        feature_names=["a", "b"],
        threshold=0.2,
        window_size=500,
    )
    live = rng.normal(size=(500, 2)).astype(np.float32)
    for row in live:
        monitor.observe(row)
    report = monitor.report()
    assert not any(report[name]["drift"] for name in ["a", "b"])
