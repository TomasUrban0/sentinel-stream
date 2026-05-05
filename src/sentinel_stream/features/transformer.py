"""Inference-time feature transformer.

Maintains a rolling buffer of the most recent observations so that the
feature set produced for a single incoming record matches what the
PySpark pipeline produced at training time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .engineering import BASE_FEATURES, feature_columns


@dataclass
class StreamingFeatureTransformer:
    """Compute features incrementally for streaming inference."""

    base_features: tuple[str, ...] = BASE_FEATURES
    rolling_windows: tuple[int, ...] = (5, 30)
    lag_steps: tuple[int, ...] = (1, 5)
    _buffer: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=64))
    _timestamps: deque[datetime] = field(default_factory=lambda: deque(maxlen=64))

    def __post_init__(self) -> None:
        max_history = max(max(self.rolling_windows), max(self.lag_steps))
        self._buffer = deque(maxlen=max_history)
        self._timestamps = deque(maxlen=max_history)

    @property
    def feature_columns(self) -> list[str]:
        return feature_columns(self.base_features, self.rolling_windows, self.lag_steps)

    def push(self, record: dict[str, float], timestamp: datetime | None = None) -> None:
        """Append a new observation to the buffer."""
        self._buffer.append({k: float(record[k]) for k in self.base_features})
        self._timestamps.append(timestamp or datetime.now(UTC))

    def transform(self) -> np.ndarray | None:
        """Return the feature vector for the latest record, or ``None`` if not enough history."""
        required = max(max(self.rolling_windows), max(self.lag_steps))
        if len(self._buffer) < required:
            return None

        df = pd.DataFrame(list(self._buffer))
        out: dict[str, float] = {}

        for col in self.base_features:
            for w in self.rolling_windows:
                window = df[col].iloc[-w:]
                out[f"{col}_mean_{w}"] = float(window.mean())
                out[f"{col}_std_{w}"] = float(window.std(ddof=1)) if len(window) > 1 else 0.0
                out[f"{col}_min_{w}"] = float(window.min())
                out[f"{col}_max_{w}"] = float(window.max())
            for lag in self.lag_steps:
                out[f"{col}_lag_{lag}"] = float(df[col].iloc[-(lag + 1)])

        ts = self._timestamps[-1]
        out["hour"] = ts.hour
        # Match Spark's dayofweek() convention (Sunday = 1 .. Saturday = 7).
        out["dayofweek"] = (ts.weekday() + 1) % 7 + 1

        return np.array([out[c] for c in self.feature_columns], dtype=np.float32)
