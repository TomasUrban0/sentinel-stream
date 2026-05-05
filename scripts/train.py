"""Train both detectors on a generated dataset and persist artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel_stream.features.engineering import (
    build_features,
    feature_columns,
    get_or_create_spark,
)
from sentinel_stream.models.autoencoder import AutoencoderDetector
from sentinel_stream.models.evaluation import evaluate
from sentinel_stream.models.isolation_forest import IsolationForestDetector
from sentinel_stream.utils.config import load_config
from sentinel_stream.utils.logger import get_logger

logger = get_logger("train")


def _to_numpy(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    logger.info("Building features with PySpark from %s", args.data)
    spark = get_or_create_spark()
    sdf = spark.read.option("header", True).option("inferSchema", True).csv(str(args.data))
    sdf = build_features(
        sdf,
        rolling_windows=tuple(config["data"]["rolling_windows"]),
        lag_steps=tuple(config["data"]["lag_steps"]),
    )
    pdf = sdf.toPandas()
    spark.stop()

    cols = feature_columns(
        rolling_windows=tuple(config["data"]["rolling_windows"]),
        lag_steps=tuple(config["data"]["lag_steps"]),
    )

    # Time-based 80/20 split: train on the first chunk only on "normal" rows,
    # evaluate on the held-out chunk against true labels.
    split = int(len(pdf) * 0.8)
    train_df = pdf.iloc[:split]
    eval_df = pdf.iloc[split:]

    train_normal = train_df[train_df["is_anomaly"] == 0]
    x_train = _to_numpy(train_normal, cols)
    x_eval = _to_numpy(eval_df, cols)
    y_eval = eval_df["is_anomaly"].to_numpy()

    logger.info("Training autoencoder on %d normal rows, %d features", *x_train.shape)
    ae = AutoencoderDetector(**config["model"]["autoencoder"]).fit(x_train)
    ae.save(str(args.out))

    logger.info("Training Isolation Forest baseline")
    iforest = IsolationForestDetector(**config["model"]["isolation_forest"]).fit(x_train)
    iforest.save(str(args.out))

    metrics = {
        "autoencoder": evaluate(ae.score(x_eval), y_eval, ae.threshold),
        "isolation_forest": evaluate(iforest.score(x_eval), y_eval, iforest.threshold),
    }
    with open(args.out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation: %s", json.dumps(metrics, indent=2))

    # Persist a reference sample for drift monitoring.
    ref_sample_size = min(config["monitoring"]["reference_sample_size"], len(x_train))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x_train), size=ref_sample_size, replace=False)
    np.save(args.out / "reference_features.npy", x_train[idx])
    logger.info("Saved %d reference rows for drift monitoring", ref_sample_size)


if __name__ == "__main__":
    main()
