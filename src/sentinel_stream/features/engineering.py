"""PySpark-based feature engineering for batch training.

The same transformations are mirrored in :mod:`sentinel_stream.features.transformer`
for single-record inference, so training and serving use identical features.

PySpark is imported lazily so that importing this module does not require
PySpark on the inference path.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .spectral import DEFAULT_BANDS, DEFAULT_WINDOW, spectral_columns

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


BASE_FEATURES = (
    "accelerometer_1_rms",
    "accelerometer_2_rms",
    "current",
    "pressure",
    "temperature",
    "thermocouple",
    "voltage",
    "volume_flow_rate",
)

DEFAULT_SPECTRAL_CHANNELS = ("accelerometer_1_rms", "accelerometer_2_rms")


def get_or_create_spark(app_name: str = "sentinel-stream") -> SparkSession:
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def build_features(
    df: DataFrame,
    base_features: Iterable[str] = BASE_FEATURES,
    rolling_windows: Iterable[int] = (5, 30),
    lag_steps: Iterable[int] = (1, 5),
) -> DataFrame:
    """Add rolling, lag, and time-based features to a Spark DataFrame.

    Spectral columns, if present in the input DataFrame, pass through unchanged.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    ordered = Window.orderBy("timestamp")

    for col in base_features:
        for w in rolling_windows:
            window = ordered.rowsBetween(-(w - 1), 0)
            df = df.withColumn(f"{col}_mean_{w}", F.avg(col).over(window))
            df = df.withColumn(f"{col}_std_{w}", F.stddev(col).over(window))
            df = df.withColumn(f"{col}_min_{w}", F.min(col).over(window))
            df = df.withColumn(f"{col}_max_{w}", F.max(col).over(window))
        for lag in lag_steps:
            df = df.withColumn(f"{col}_lag_{lag}", F.lag(col, lag).over(ordered))

    df = df.withColumn("hour", F.hour("timestamp"))
    df = df.withColumn("dayofweek", F.dayofweek("timestamp"))

    drop_after = max(rolling_windows) if rolling_windows else 0
    drop_after = max(drop_after, max(lag_steps) if lag_steps else 0)
    if drop_after > 0:
        df = df.dropna()

    return df


def feature_columns(
    base_features: Iterable[str] = BASE_FEATURES,
    rolling_windows: Iterable[int] = (5, 30),
    lag_steps: Iterable[int] = (1, 5),
    spectral_channels: Iterable[str] = DEFAULT_SPECTRAL_CHANNELS,
    spectral_bands: int = DEFAULT_BANDS,
) -> list[str]:
    """Return the deterministic ordered list of feature columns produced above."""
    cols: list[str] = []
    for col in base_features:
        for w in rolling_windows:
            cols.extend([f"{col}_mean_{w}", f"{col}_std_{w}", f"{col}_min_{w}", f"{col}_max_{w}"])
        for lag in lag_steps:
            cols.append(f"{col}_lag_{lag}")
    cols.extend(["hour", "dayofweek"])
    cols.extend(spectral_columns(spectral_channels, n_bands=spectral_bands))
    return cols


__all__ = [
    "BASE_FEATURES",
    "DEFAULT_BANDS",
    "DEFAULT_SPECTRAL_CHANNELS",
    "DEFAULT_WINDOW",
    "build_features",
    "feature_columns",
    "get_or_create_spark",
]
