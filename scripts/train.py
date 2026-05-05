"""Train both detectors on SKAB and persist artifacts.

Training is unsupervised: only the anomaly-free partition is fed to the
detectors. The labelled partition is split by source file into
validation (20 %) and test (80 %): the validation slice is used to tune the
decision threshold by maximising F1, and the test slice gives the final,
unseen-data metrics that get reported. This avoids the over-flagging that
results from blindly using a percentile-of-training threshold when the
labelled domain differs from the training domain.
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
from sentinel_stream.features.spectral import add_spectral_columns
from sentinel_stream.models.autoencoder import AutoencoderDetector
from sentinel_stream.models.base import AnomalyDetector
from sentinel_stream.models.evaluation import best_f1_threshold, evaluate
from sentinel_stream.models.isolation_forest import IsolationForestDetector
from sentinel_stream.utils.config import load_config
from sentinel_stream.utils.logger import get_logger

logger = get_logger("train")

VAL_FRACTION = 0.20
SPLIT_SEED = 0


def _to_numpy(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32)


def _split_by_file(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labeled rows so a given source CSV lands entirely in val or test."""
    files = sorted(labeled["source_file"].unique())
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * VAL_FRACTION)))
    val_files = set(files[:n_val])
    val_mask = labeled["source_file"].isin(val_files)
    return labeled[val_mask].reset_index(drop=True), labeled[~val_mask].reset_index(drop=True)


def _spark_features(
    pdf: pd.DataFrame,
    base_features: tuple[str, ...],
    rolling_windows: tuple[int, ...],
    lag_steps: tuple[int, ...],
    spectral_channels: tuple[str, ...],
    spectral_window: int,
    spectral_bands: int,
    work_dir: Path,
    name: str,
) -> pd.DataFrame:
    # Spectral columns are computed in pandas first because rFFT inside a
    # Spark window is awkward to express; once they are columns on the
    # DataFrame, Spark passes them through unchanged.
    pdf = add_spectral_columns(
        pdf, channels=spectral_channels, window=spectral_window, n_bands=spectral_bands
    )

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


def _tune_and_evaluate(
    detector: AnomalyDetector,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, dict, float, float]:
    val_scores = detector.score(x_val)
    test_scores = detector.score(x_test)

    raw_threshold = detector.threshold
    val_raw = evaluate(val_scores, y_val, raw_threshold)
    test_raw = evaluate(test_scores, y_test, raw_threshold)

    tuned_threshold, val_f1 = best_f1_threshold(val_scores, y_val)
    val_tuned = evaluate(val_scores, y_val, tuned_threshold)
    test_tuned = evaluate(test_scores, y_test, tuned_threshold)

    return (
        {"val_raw": val_raw, "val_tuned": val_tuned, "test_raw": test_raw, "test_tuned": test_tuned},
        {"raw_threshold": float(raw_threshold), "tuned_threshold": float(tuned_threshold)},
        float(val_f1),
        float(test_tuned["f1"]),
    )


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
    spectral_cfg = config["data"].get("spectral", {})
    spectral_channels = tuple(spectral_cfg.get("channels", []))
    spectral_window = int(spectral_cfg.get("window", 64))
    spectral_bands = int(spectral_cfg.get("bands", 4))

    logger.info("Loading SKAB from %s", args.data_root)
    normal_df = load_normal(args.data_root)
    labeled_df = load_labeled(args.data_root)
    val_df, test_df = _split_by_file(labeled_df)
    logger.info(
        "Partition sizes: train=%d (normal), val=%d (anomalies=%.2f%%), test=%d (anomalies=%.2f%%)",
        len(normal_df),
        len(val_df),
        100 * val_df["is_anomaly"].mean(),
        len(test_df),
        100 * test_df["is_anomaly"].mean(),
    )

    work_dir = args.out / "_spark_inputs"
    logger.info("Building features with PySpark + FFT")
    feat_kwargs = {
        "base_features": base_features,
        "rolling_windows": rolling_windows,
        "lag_steps": lag_steps,
        "spectral_channels": spectral_channels,
        "spectral_window": spectral_window,
        "spectral_bands": spectral_bands,
    }
    train_pdf = _spark_features(normal_df, **feat_kwargs, work_dir=work_dir, name="train")
    val_pdf = _spark_features(val_df, **feat_kwargs, work_dir=work_dir, name="val")
    test_pdf = _spark_features(test_df, **feat_kwargs, work_dir=work_dir, name="test")

    cols = feature_columns(
        base_features, rolling_windows, lag_steps, spectral_channels, spectral_bands
    )
    x_train = _to_numpy(train_pdf, cols)
    x_val = _to_numpy(val_pdf, cols)
    x_test = _to_numpy(test_pdf, cols)
    y_val = val_pdf["is_anomaly"].to_numpy()
    y_test = test_pdf["is_anomaly"].to_numpy()

    logger.info("Training autoencoder on %d normal rows, %d features", *x_train.shape)
    ae = AutoencoderDetector(**config["model"]["autoencoder"]).fit(x_train)
    logger.info("Training Isolation Forest baseline")
    iforest = IsolationForestDetector(**config["model"]["isolation_forest"]).fit(x_train)

    ae_metrics, ae_thr, ae_val_f1, ae_test_f1 = _tune_and_evaluate(
        ae, x_val, y_val, x_test, y_test
    )
    if_metrics, if_thr, if_val_f1, if_test_f1 = _tune_and_evaluate(
        iforest, x_val, y_val, x_test, y_test
    )
    logger.info(
        "Autoencoder       — val F1 %.3f -> tuned %.3f, test F1 (tuned) %.3f",
        ae_metrics["val_raw"]["f1"],
        ae_val_f1,
        ae_test_f1,
    )
    logger.info(
        "Isolation Forest  — val F1 %.3f -> tuned %.3f, test F1 (tuned) %.3f",
        if_metrics["val_raw"]["f1"],
        if_val_f1,
        if_test_f1,
    )

    # Persist the tuned threshold inside the model artifacts so the API uses it at serving time.
    ae.threshold = ae_thr["tuned_threshold"]
    iforest.threshold = if_thr["tuned_threshold"]
    ae.save(str(args.out))
    iforest.save(str(args.out))

    metrics = {
        "dataset": "SKAB",
        "split": {
            "train_normal_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "val_anomaly_rate": float(y_val.mean()),
            "test_rows": int(len(x_test)),
            "test_anomaly_rate": float(y_test.mean()),
            "val_fraction_of_files": VAL_FRACTION,
        },
        "trivial_flag_all_baseline_test_f1": float(
            2 * y_test.mean() / (1 + y_test.mean())
        ),
        "autoencoder": {**ae_metrics, "thresholds": ae_thr},
        "isolation_forest": {**if_metrics, "thresholds": if_thr},
    }
    with open(args.out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Final metrics:\n%s", json.dumps(metrics, indent=2))

    ref_sample_size = min(config["monitoring"]["reference_sample_size"], len(x_train))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x_train), size=ref_sample_size, replace=False)
    np.save(args.out / "reference_features.npy", x_train[idx])
    logger.info("Saved %d reference rows for drift monitoring", ref_sample_size)


if __name__ == "__main__":
    main()
