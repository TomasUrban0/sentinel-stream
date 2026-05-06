"""Inference-time feature transformer.

The AI4I dataset is row-independent, so there is no rolling buffer here:
the transformer takes a single record and emits the full feature vector
that the deployed model consumes — the same 10 features the PySpark batch
pipeline produces during training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engineering import ALL_FEATURE_COLUMNS, BASE_FEATURES, TYPE_ORDINAL


@dataclass
class StreamingFeatureTransformer:
    """Compute the engineered feature vector for one incoming record."""

    base_features: tuple[str, ...] = BASE_FEATURES
    feature_order: tuple[str, ...] = ALL_FEATURE_COLUMNS

    @property
    def feature_columns(self) -> list[str]:
        return list(self.feature_order)

    def transform(self, record: dict[str, float | str]) -> np.ndarray:
        values: dict[str, float] = {}
        for feat in self.base_features:
            values[feat] = float(record[feat])

        type_value = str(record.get("type", "L"))
        values["type_ordinal"] = float(TYPE_ORDINAL.get(type_value, TYPE_ORDINAL["L"]))
        values["temperature_delta_k"] = (
            values["process_temperature_k"] - values["air_temperature_k"]
        )
        values["mechanical_power_w"] = (
            values["torque_nm"] * values["rotational_speed_rpm"] * (2 * math.pi / 60.0)
        )
        values["wear_torque_proxy"] = values["torque_nm"] * values["tool_wear_min"]
        values["wear_speed_proxy"] = (
            values["rotational_speed_rpm"] * values["tool_wear_min"]
        )

        return np.array([values[c] for c in self.feature_order], dtype=np.float32)
