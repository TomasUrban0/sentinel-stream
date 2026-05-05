"""Train both detectors on SKAB and persist artifacts.

Training is unsupervised: only the anomaly-free partition is fed to the
detectors. Evaluation uses the labeled partition (valve1 + valve2 + other),
so the test labels are real industrial faults, not synthetic injections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel_stream.data.skab_loader import load_labeled, load_normal
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


def _spark_features(
    pdf: pd.DataFrame,
    base_features: tuple[str, ...],
    rolling_windows: tuple[int, ...],
    lag_steps: tuple[int, ...],
    work_dir: Path,
    name: str,
) -> pd.DataFrame:
    # Spark on Windows can't reliably ingest a pandas DataFrame via
    # createDataFrame (Python worker socket timeouts), so we round-trip
    # through a CSV — Spark's native CSV reader is rock solid.
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = work_dir / f"{name}.csv"
    pdf.to_csv(csv_path, index=False)
    spark = get_or_create_spark()
    sdf = spark.read.option("header", True).option("inferSchema", True).csv(str(csv_path))
    sdf = build_features(
        sdf,
        base_features=base_features,
        rolling_windows=rolling_windows,
        lag_steps=lag_steps,
    )
    out = sdf.toPandas()
    spark.stop()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/skab"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    base_features = tuple(config["data"]["features"])
    rolling_windows = tuple(config["data"]["rolling_windows"])
    lag_steps = tuple(config["data"]["lag_steps"])

    logger.info("Loading SKAB from %s", args.data_root)
    normal_df = load_normal(args.data_root)
    labeled_df = load_labeled(args.data_root)
    logger.info(
        "Loaded %d normal rows and %d labeled rows (%.2f%% anomalies)",
        len(normal_df),
        len(labeled_df),
        100 * labeled_df["is_anomaly"].mean(),
    )

    logger.info("Building features with PySpark")
    work_dir = args.out / "_spark_inputs"
    train_pdf = _spark_features(
        normal_df, base_features, rolling_windows, lag_steps, work_dir, "train"
    )
    eval_pdf = _spark_features(
        labeled_df, base_features, rolling_windows, lag_steps, work_dir, "eval"
    )

    cols = feature_columns(base_features, rolling_windows, lag_steps)
    x_train = _to_numpy(train_pdf, cols)
    x_eval = _to_numpy(eval_pdf, cols)
    y_eval = eval_pdf["is_anomaly"].to_numpy()

    logger.info("Training autoencoder on %d normal rows, %d features", *x_train.shape)
    ae = AutoencoderDetector(**config["model"]["autoencoder"]).fit(x_train)
    ae.save(str(args.out))

    logger.info("Training Isolation Forest baseline")
    iforest = IsolationForestDetector(**config["model"]["isolation_forest"]).fit(x_train)
    iforest.save(str(args.out))

    metrics = {
        "dataset": "SKAB",
        "n_train_normal_rows": int(len(x_train)),
        "n_eval_rows": int(len(x_eval)),
        "eval_anomaly_rate": float(y_eval.mean()),
        "autoencoder": evaluate(ae.score(x_eval), y_eval, ae.threshold),
        "isolation_forest": evaluate(iforest.score(x_eval), y_eval, iforest.threshold),
    }
    with open(args.out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation: %s", json.dumps(metrics, indent=2))

    ref_sample_size = min(config["monitoring"]["reference_sample_size"], len(x_train))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x_train), size=ref_sample_size, replace=False)
    np.save(args.out / "reference_features.npy", x_train[idx])
    logger.info("Saved %d reference rows for drift monitoring", ref_sample_size)


if __name__ == "__main__":
    main()
