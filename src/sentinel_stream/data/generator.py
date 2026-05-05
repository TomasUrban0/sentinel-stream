"""Synthetic multivariate sensor data generator with injected anomalies.

The generator simulates four correlated sensor channels (temperature, pressure,
vibration, humidity) at one-second resolution. Each channel combines a daily
seasonality, a slow trend, and Gaussian noise. A configurable fraction of the
points are perturbed to act as anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    rows: int = 50_000
    anomaly_rate: float = 0.02
    seed: int = 42
    start: str = "2025-01-01"
    freq: str = "s"


def _seasonal(t: np.ndarray, period: float, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * t / period)


def generate(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = config.rows
    t = np.arange(n)

    temperature = (
        70.0
        + _seasonal(t, period=86_400, amplitude=4.0)
        + 0.00002 * t
        + rng.normal(0, 0.7, n)
    )
    pressure = (
        101.3
        + _seasonal(t, period=86_400, amplitude=0.5)
        + rng.normal(0, 0.15, n)
    )
    vibration = 0.10 + np.abs(rng.normal(0, 0.02, n))
    humidity = (
        45.0
        + _seasonal(t, period=86_400, amplitude=8.0)
        + rng.normal(0, 1.2, n)
    )

    n_anomalies = int(n * config.anomaly_rate)
    anomaly_idx = rng.choice(n, size=n_anomalies, replace=False)
    is_anomaly = np.zeros(n, dtype=int)
    is_anomaly[anomaly_idx] = 1

    # Three flavors of anomaly: spikes, drifts, and correlated faults.
    flavors = rng.integers(0, 3, size=n_anomalies)
    for i, idx in enumerate(anomaly_idx):
        flavor = flavors[i]
        if flavor == 0:  # spike
            temperature[idx] += rng.normal(15, 3) * rng.choice([-1, 1])
        elif flavor == 1:  # drift segment
            end = min(idx + rng.integers(20, 80), n)
            pressure[idx:end] += rng.normal(2.5, 0.4)
            is_anomaly[idx:end] = 1
        else:  # correlated fault
            vibration[idx] += rng.uniform(0.5, 1.5)
            temperature[idx] += rng.normal(8, 2)

    timestamps = pd.date_range(start=config.start, periods=n, freq=config.freq)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration,
            "humidity": humidity,
            "is_anomaly": is_anomaly,
        }
    )
