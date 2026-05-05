"""Tests for the FFT-based spectral feature module."""

import numpy as np
import pandas as pd

from sentinel_stream.features.spectral import (
    add_spectral_columns,
    compute_spectral_features,
    spectral_columns,
    spectral_feature_names,
)


def test_spectral_features_separate_known_frequencies():
    rng = np.random.default_rng(0)
    n = 64
    t = np.arange(n)

    low = np.sin(2 * np.pi * t * (1 / 32)) + 0.05 * rng.normal(size=n)
    high = np.sin(2 * np.pi * t * (1 / 4)) + 0.05 * rng.normal(size=n)

    low_feats = compute_spectral_features(low, n_bands=4)
    high_feats = compute_spectral_features(high, n_bands=4)

    # Indices: band1..band4, centroid, entropy
    assert low_feats.shape == (6,)
    assert low_feats[0] > low_feats[3]  # low signal -> energy in band 1
    assert high_feats[3] > high_feats[0]  # high signal -> energy in band 4
    # The high-frequency signal must have a strictly larger spectral centroid.
    assert high_feats[4] > low_feats[4]


def test_add_spectral_columns_warmup_is_nan():
    df = pd.DataFrame({"x": np.arange(80, dtype=float)})
    out = add_spectral_columns(df, channels=("x",), window=32, n_bands=2)

    expected = [f"x_spec_{name}" for name in spectral_feature_names(2)]
    assert all(col in out.columns for col in expected)
    assert out.loc[: 32 - 2, expected].isna().all().all()
    assert not out.loc[32 - 1 :, expected].isna().any().any()


def test_spectral_columns_helper_matches_naming():
    cols = spectral_columns(["a", "b"], n_bands=3)
    assert cols == [
        "a_spec_band1_power",
        "a_spec_band2_power",
        "a_spec_band3_power",
        "a_spec_centroid",
        "a_spec_entropy",
        "b_spec_band1_power",
        "b_spec_band2_power",
        "b_spec_band3_power",
        "b_spec_centroid",
        "b_spec_entropy",
    ]
