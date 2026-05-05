"""Loader for the Skoltech Anomaly Benchmark (SKAB) dataset.

SKAB is a multivariate time-series anomaly detection benchmark released by
Skoltech (https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab).
Each record carries eight industrial sensor readings sampled at 1 Hz from a
water-circulation testbed with deliberately induced faults.

This module exposes a small loader API that the rest of the project consumes
without ever needing to know the on-disk layout.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

RAW_TO_FEATURE: dict[str, str] = {
    "Accelerometer1RMS": "accelerometer_1_rms",
    "Accelerometer2RMS": "accelerometer_2_rms",
    "Current": "current",
    "Pressure": "pressure",
    "Temperature": "temperature",
    "Thermocouple": "thermocouple",
    "Voltage": "voltage",
    "Volume Flow RateRMS": "volume_flow_rate",
}

SKAB_FEATURES: tuple[str, ...] = tuple(RAW_TO_FEATURE.values())

LABELED_SUBDIRS: tuple[str, ...] = ("valve1", "valve2", "other")
NORMAL_SUBDIR: str = "anomaly-free"


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df.rename(columns=RAW_TO_FEATURE)
    df["timestamp"] = pd.to_datetime(df["datetime"])
    df = df.drop(columns=["datetime"])
    if "anomaly" in df.columns:
        df["is_anomaly"] = df["anomaly"].astype(int)
    else:
        df["is_anomaly"] = 0
    if "changepoint" in df.columns:
        df = df.drop(columns=["changepoint", "anomaly"])
    df["source_file"] = path.name
    return df


def _resolve_root(root: str | Path) -> Path:
    root = Path(root)
    inner = root / "SKAB"
    return inner if inner.is_dir() else root


def load_normal(root: str | Path = "data/skab") -> pd.DataFrame:
    """Return the concatenated anomaly-free training partition."""
    base = _resolve_root(root) / NORMAL_SUBDIR
    files = sorted(base.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No anomaly-free CSVs under {base}. Run `make download-data`."
        )
    return pd.concat([_read_one(p) for p in files], ignore_index=True)


def load_labeled(
    root: str | Path = "data/skab",
    subdirs: Iterable[str] = LABELED_SUBDIRS,
) -> pd.DataFrame:
    """Return the concatenated labeled partition (used for evaluation)."""
    base = _resolve_root(root)
    frames: list[pd.DataFrame] = []
    for sub in subdirs:
        for p in sorted((base / sub).glob("*.csv")):
            frames.append(_read_one(p))
    if not frames:
        raise FileNotFoundError(
            f"No labeled CSVs under {base}. Run `make download-data`."
        )
    return pd.concat(frames, ignore_index=True)


def load_combined(root: str | Path = "data/skab") -> pd.DataFrame:
    """Concatenate the normal partition and the labeled partition.

    The normal rows are emitted first so a chronological split during training
    sees only normal data in the early portion of the series.
    """
    return pd.concat([load_normal(root), load_labeled(root)], ignore_index=True)
