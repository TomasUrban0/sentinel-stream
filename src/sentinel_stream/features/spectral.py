"""Frequency-domain features for the accelerometer channels.

SKAB is sampled at 1 Hz, so the FFT here resolves sub-Hertz oscillations
(0.016 Hz to 0.5 Hz with a 64-sample window), not audio-band vibration.
That range still carries useful signal for valve faults: hunting and limit-
cycle behaviour from a partially-stuck valve produces low-frequency
periodic disturbances that rolling-window means and standard deviations
cannot represent.

Six features are emitted per channel and per window:
  - ``band{1..n}_power``: fraction of total spectral energy in each of
    ``n_bands`` equal-width frequency bands
  - ``centroid``: spectral centroid (mean frequency weighted by magnitude)
  - ``entropy``: Shannon entropy of the normalised magnitude spectrum
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_BANDS = 4
DEFAULT_WINDOW = 64

_EPS = 1e-12


def spectral_feature_names(n_bands: int = DEFAULT_BANDS) -> tuple[str, ...]:
    return (*[f"band{i + 1}_power" for i in range(n_bands)], "centroid", "entropy")


def compute_spectral_features(values: np.ndarray, n_bands: int = DEFAULT_BANDS) -> np.ndarray:
    """Return the spectral feature vector for a 1-D signal window.

    The signal is mean-centred before the rFFT so band-1 power is not dominated
    by the DC component, which carries no information about oscillations.
    """
    if values.ndim != 1:
        raise ValueError("compute_spectral_features expects a 1-D array")
    centred = values.astype(np.float64) - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centred))

    bands = np.array_split(spectrum, n_bands)
    band_power = np.array([float(np.sum(b * b)) for b in bands])
    total_power = float(band_power.sum()) + _EPS
    band_power_norm = band_power / total_power

    freqs = np.arange(spectrum.size, dtype=np.float64)
    spec_sum = float(spectrum.sum()) + _EPS
    centroid = float((freqs * spectrum).sum() / spec_sum)
    p = spectrum / spec_sum
    entropy = float(-(p * np.log(p + _EPS)).sum())

    return np.concatenate([band_power_norm, [centroid, entropy]]).astype(np.float32)


def add_spectral_columns(
    df: pd.DataFrame,
    channels: Iterable[str],
    window: int = DEFAULT_WINDOW,
    n_bands: int = DEFAULT_BANDS,
) -> pd.DataFrame:
    """Append rolling spectral features to ``df`` for each named channel.

    Rows whose history is shorter than ``window`` are filled with NaN; the
    PySpark feature builder downstream drops those rows after its own warm-up,
    so the two pipelines stay aligned.
    """
    out = df.copy()
    feature_names = spectral_feature_names(n_bands)
    for ch in channels:
        values = df[ch].to_numpy(dtype=np.float64)
        n = len(values)
        block = np.full((n, len(feature_names)), np.nan, dtype=np.float64)
        for i in range(window - 1, n):
            block[i] = compute_spectral_features(values[i - window + 1 : i + 1], n_bands=n_bands)
        for j, name in enumerate(feature_names):
            out[f"{ch}_spec_{name}"] = block[:, j]
    return out


def spectral_columns(channels: Iterable[str], n_bands: int = DEFAULT_BANDS) -> list[str]:
    return [f"{ch}_spec_{name}" for ch in channels for name in spectral_feature_names(n_bands)]
