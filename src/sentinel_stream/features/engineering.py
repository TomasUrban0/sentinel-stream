"""PySpark-based feature engineering for the AI4I predictive-maintenance dataset.

The same derived features are produced by the streaming transformer in
:mod:`sentinel_stream.features.transformer` so training and serving stay
in lockstep.

PySpark is imported lazily so importing this module does not require Spark
on the inference path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


BASE_FEATURES = (
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)

# "type" is a categorical {L, M, H} mapped to {1, 2, 3}.
TYPE_ORDINAL: dict[str, int] = {"L": 1, "M": 2, "H": 3}

DERIVED_FEATURES = (
    "type_ordinal",
    "temperature_delta_k",
    "mechanical_power_w",
    "wear_torque_proxy",
    "wear_speed_proxy",
)

ALL_FEATURE_COLUMNS = (*BASE_FEATURES, *DERIVED_FEATURES)


def get_or_create_spark(app_name: str = "sentinel-stream") -> SparkSession:
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def build_features(df: DataFrame) -> DataFrame:
    """Add engineered features to a Spark DataFrame.

    Adds:
      * ``type_ordinal`` — encodes the categorical product variant
      * ``temperature_delta_k`` — process minus ambient temperature
      * ``mechanical_power_w`` — torque × angular velocity, in watts
      * ``wear_torque_proxy`` — interaction term capturing overstrain risk
      * ``wear_speed_proxy`` — interaction term capturing tool-fatigue risk
    """
    from pyspark.sql import functions as F

    df = df.withColumn(
        "type_ordinal",
        F.when(F.col("type") == "L", F.lit(TYPE_ORDINAL["L"]))
        .when(F.col("type") == "M", F.lit(TYPE_ORDINAL["M"]))
        .when(F.col("type") == "H", F.lit(TYPE_ORDINAL["H"]))
        .otherwise(F.lit(TYPE_ORDINAL["L"])),
    )
    df = df.withColumn(
        "temperature_delta_k",
        F.col("process_temperature_k") - F.col("air_temperature_k"),
    )
    # Mechanical power = torque (Nm) × angular speed (rad/s); rpm to rad/s = 2π/60.
    df = df.withColumn(
        "mechanical_power_w",
        F.col("torque_nm") * F.col("rotational_speed_rpm") * F.lit(2 * math.pi / 60.0),
    )
    df = df.withColumn(
        "wear_torque_proxy",
        F.col("torque_nm") * F.col("tool_wear_min"),
    )
    df = df.withColumn(
        "wear_speed_proxy",
        F.col("rotational_speed_rpm") * F.col("tool_wear_min"),
    )
    return df


def feature_columns(extra: Iterable[str] = ()) -> list[str]:
    """The deterministic ordered feature list consumed by every model."""
    return list(ALL_FEATURE_COLUMNS) + list(extra)
