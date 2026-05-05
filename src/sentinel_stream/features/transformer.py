"""Inference-time feature transformer.

Maintains a rolling buffer of the most recent observations so that the
feature set produced for a single incoming record matches what the
PySpark pipeline produced at training time, including the FFT-based
spectral features computed on the accelerometer channels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .engineering import BASE_FEATURES, DEFAULT_SPECTRAL_CHANNELS, feature_columns
from .spectral import DEFAULT_BANDS, DEFAULT_WINDOW, compute_spectral_features


@dataclass
class StreamingFeatureTransformer:
    """Compute features incrementally for streaming inference."""

    base_features: tuple[str, ...] = BASE_FEATURES
    rolling_windows: tuple[int, ...] = (5, 30)
    lag_steps: tuple[int, ...] = (1, 5)
    spectral_channels: tuple[str, ...] = DEFAULT_SPECTRAL_CHANNELS
    spectral_window: int = DEFAULT_WINDOW
    spectral_bands: int = DEFAULT_BANDS
    _buffer: deque[dict[str, float]] = field(default_factory=deque)
    _timestamps: deque[datetime] = field(default_factory=deque)

    def __post_init__(self) -> None:
        spectral_required = self.spectral_window if self.spectral_channels else 0
        max_history = max(
            max(self.rolling_windows),
            max(self.lag_steps) + 1,
            spectral_required,
        )
        self._buffer = deque(maxlen=max_history)
        self._timestamps = deque(maxlen=max_history)

    @property
    def feature_columns(self) -> list[str]:
        return feature_columns(
            self.base_features,
            self.rolling_windows,
            self.lag_steps,
            self.spectral_channels,
            self.spectral_bands,
        )

    def push(self, record: dict[str, float], timestamp: datetime | None = None) -> None:
        self._buffer.append({k: float(record[k]) for k in self.base_features})
        self._timestamps.append(timestamp or datetime.now(UTC))

    def transform(self) -> np.ndarray | None:
        spectral_required = self.spectral_window if self.spectral_channels else 0
        required = max(
            max(self.rolling_windows),
            max(self.lag_steps) + 1,
            spectral_required,
        )
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
        out["dayofweek"] = (ts.weekday() + 1) % 7 + 1

        from .spectral import spectral_feature_names

        spec_names = spectral_feature_names(self.spectral_bands)
        for ch in self.spectral_channels:
            window_values = df[ch].iloc[-self.spectral_window :].to_numpy()
            spec_vec = compute_spectral_features(window_values, n_bands=self.spectral_bands)
            for name, value in zip(spec_names, spec_vec, strict=True):
                out[f"{ch}_spec_{name}"] = float(value)

        return np.array([out[c] for c in self.feature_columns], dtype=np.float32)
