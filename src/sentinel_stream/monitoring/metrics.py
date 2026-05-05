"""In-process metrics aggregator for the API."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class MetricsRegistry:
    max_latencies: int = 5000
    total_predictions: int = 0
    total_anomalies: int = 0
    _latencies_ms: deque[float] = field(default_factory=deque)
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self._latencies_ms = deque(maxlen=self.max_latencies)

    def record(self, latency_ms: float, is_anomaly: bool) -> None:
        with self._lock:
            self._latencies_ms.append(latency_ms)
            self.total_predictions += 1
            if is_anomaly:
                self.total_anomalies += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            latencies = sorted(self._latencies_ms)
        if not latencies:
            return {
                "total_predictions": self.total_predictions,
                "total_anomalies": self.total_anomalies,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        def pct(p: float) -> float:
            idx = min(len(latencies) - 1, int(p * (len(latencies) - 1)))
            return latencies[idx]

        return {
            "total_predictions": self.total_predictions,
            "total_anomalies": self.total_anomalies,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
        }
