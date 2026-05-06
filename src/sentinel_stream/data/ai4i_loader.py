"""Loader for the AI4I 2020 Predictive Maintenance dataset.

Source: https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
Originally published as the AI4I 2020 Predictive Maintenance Dataset by
Stephan Matzka, HTW Berlin (CC BY 4.0). 10 000 production runs, six sensor
readings per run, a binary ``Machine failure`` target, and five flags for
the underlying failure modes (TWF, HDF, PWF, OSF, RNF).

The original column names contain spaces and units, which would not survive
a Pydantic schema or a Spark column reference. ``load_ai4i`` returns the
canonical, snake-cased column layout the rest of the project expects.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_TO_FEATURE: dict[str, str] = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}

# Numeric-only sensor inputs (type is categorical and handled separately).
BASE_FEATURES: tuple[str, ...] = (
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("type",)
TARGET_COLUMN: str = "machine_failure"
FAILURE_MODES: tuple[str, ...] = ("TWF", "HDF", "PWF", "OSF", "RNF")


def _resolve_csv(root: str | Path) -> Path:
    root = Path(root)
    if root.is_file():
        return root
    candidates = sorted(root.glob("ai4i*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No ai4i*.csv under {root}. Run `make data` (or `python scripts/download_data.py`)."
        )
    return candidates[0]


def load_ai4i(root: str | Path = "data/ai4i") -> pd.DataFrame:
    """Return the AI4I dataset with normalised column names."""
    df = pd.read_csv(_resolve_csv(root))
    df = df.rename(columns={**RAW_TO_FEATURE, "Machine failure": TARGET_COLUMN})
    df = df.drop(columns=["UDI", "Product ID"], errors="ignore")
    return df
