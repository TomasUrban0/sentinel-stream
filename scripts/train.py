"""Train both classifiers on AI4I 2020 and persist artifacts.

Pipeline
========
1. Load the AI4I CSV (10 000 production runs, 6 raw sensor readings, 1 binary
   target plus 5 failure-mode flags).
2. Engineer derived features with PySpark (mechanical power, temperature
   delta, two interaction terms, ordinal product variant).
3. Stratified 70 / 15 / 15 train / val / test split with a fixed seed.
4. Train an XGBoost classifier (with ``scale_pos_weight`` handling the 3.4 %
   class imbalance) and a Keras dense net (with class weights) on the same
   feature matrix.
5. Tune each model's decision threshold on the validation slice by
   maximising F1; freeze it inside the persisted artifact.
6. Report final metrics on the held-out test slice and write
   ``artifacts/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentinel_stream.data.ai4i_loader import (
    BASE_FEATURES as RAW_FEATURES,
)
from sentinel_stream.data.ai4i_loader import (
    TARGET_COLUMN,
    load_ai4i,
)
from sentinel_stream.features.engineering import (
    ALL_FEATURE_COLUMNS,
    build_features,
    get_or_create_spark,
)
from sentinel_stream.models.base import FailurePredictor
from sentinel_stream.models.evaluation import best_f1_threshold, evaluate
from sentinel_stream.models.keras_classifier import KerasClassifier
from sentinel_stream.models.xgboost_classifier import XGBoostClassifier
from sentinel_stream.utils.config import load_config
from sentinel_stream.utils.logger import get_logger

logger = get_logger("train")


def _spark_features(pdf: pd.DataFrame, work_dir: Path, name: str) -> pd.DataFrame:
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = work_dir / f"{name}.csv"
    pdf.to_csv(csv_path, index=False)
    spark = get_or_create_spark()
    sdf = spark.read.option("header", True).option("inferSchema", True).csv(str(csv_path))
    sdf = build_features(sdf)
    out = sdf.toPandas()
    spark.stop()
    return out


def _stratified_split(
    df: pd.DataFrame,
    target: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=test_fraction,
        random_state=seed,
        stratify=df[target],
    )
    relative_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val[target],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _evaluate_with_tuning(
    predictor: FailurePredictor,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    val_proba = predictor.predict_proba(x_val)
    test_proba = predictor.predict_proba(x_test)

    raw_threshold = predictor.threshold
    val_raw = evaluate(val_proba, y_val, raw_threshold)
    test_raw = evaluate(test_proba, y_test, raw_threshold)

    tuned_threshold, val_f1 = best_f1_threshold(val_proba, y_val)
    val_tuned = evaluate(val_proba, y_val, tuned_threshold)
    test_tuned = evaluate(test_proba, y_test, tuned_threshold)

    predictor.threshold = float(tuned_threshold)

    return {
        "val_raw": val_raw,
        "val_tuned": val_tuned,
        "test_raw": test_raw,
        "test_tuned": test_tuned,
        "thresholds": {
            "raw_threshold": float(raw_threshold),
            "tuned_threshold": float(tuned_threshold),
            "val_f1_at_tuned": float(val_f1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ai4i"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading AI4I from %s", args.data_root)
    df = load_ai4i(args.data_root)
    logger.info(
        "Loaded %d rows, %d failures (%.2f%%)",
        len(df),
        int(df[TARGET_COLUMN].sum()),
        100 * df[TARGET_COLUMN].mean(),
    )

    split_cfg = config["data"]["split"]
    train_df, val_df, test_df = _stratified_split(
        df,
        target=TARGET_COLUMN,
        val_fraction=float(split_cfg["val_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
        seed=int(split_cfg["seed"]),
    )
    logger.info(
        "Split sizes: train=%d (pos=%.2f%%), val=%d (pos=%.2f%%), test=%d (pos=%.2f%%)",
        len(train_df),
        100 * train_df[TARGET_COLUMN].mean(),
        len(val_df),
        100 * val_df[TARGET_COLUMN].mean(),
        len(test_df),
        100 * test_df[TARGET_COLUMN].mean(),
    )

    work_dir = args.out / "_spark_inputs"
    logger.info("Building features with PySpark")
    train_pdf = _spark_features(train_df, work_dir, "train")
    val_pdf = _spark_features(val_df, work_dir, "val")
    test_pdf = _spark_features(test_df, work_dir, "test")

    cols = list(ALL_FEATURE_COLUMNS)
    x_train = train_pdf[cols].to_numpy(dtype=np.float32)
    x_val = val_pdf[cols].to_numpy(dtype=np.float32)
    x_test = test_pdf[cols].to_numpy(dtype=np.float32)
    y_train = train_pdf[TARGET_COLUMN].to_numpy(dtype=int)
    y_val = val_pdf[TARGET_COLUMN].to_numpy(dtype=int)
    y_test = test_pdf[TARGET_COLUMN].to_numpy(dtype=int)

    logger.info("Training XGBoost on %d rows × %d features", *x_train.shape)
    xgb = XGBoostClassifier(**config["model"]["xgboost"]).fit(
        x_train, y_train, feature_names=cols
    )
    logger.info("Training Keras dense classifier")
    keras_clf = KerasClassifier(**config["model"]["keras_dense"]).fit(x_train, y_train)

    xgb_metrics = _evaluate_with_tuning(xgb, x_val, y_val, x_test, y_test)
    keras_metrics = _evaluate_with_tuning(keras_clf, x_val, y_val, x_test, y_test)

    xgb.save(str(args.out))
    keras_clf.save(str(args.out))
    logger.info("Saved both classifiers to %s", args.out)

    feature_importance = xgb.feature_importance() or {}

    metrics = {
        "dataset": "AI4I 2020 Predictive Maintenance",
        "raw_features": list(RAW_FEATURES),
        "engineered_features": list(ALL_FEATURE_COLUMNS),
        "split": {
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
            "train_pos_rate": float(y_train.mean()),
            "val_pos_rate": float(y_val.mean()),
            "test_pos_rate": float(y_test.mean()),
        },
        "trivial_flag_all_baseline_test_f1": float(
            2 * y_test.mean() / (1 + y_test.mean())
        ),
        "xgboost": xgb_metrics,
        "keras_dense": keras_metrics,
        "xgboost_feature_importance": feature_importance,
    }
    with open(args.out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Final test metrics:")
    logger.info(
        "  XGBoost      F1=%.3f  ROC-AUC=%.3f  PR-AUC=%.3f  P=%.3f  R=%.3f",
        xgb_metrics["test_tuned"]["f1"],
        xgb_metrics["test_tuned"]["roc_auc"],
        xgb_metrics["test_tuned"]["pr_auc"],
        xgb_metrics["test_tuned"]["precision"],
        xgb_metrics["test_tuned"]["recall"],
    )
    logger.info(
        "  Keras dense  F1=%.3f  ROC-AUC=%.3f  PR-AUC=%.3f  P=%.3f  R=%.3f",
        keras_metrics["test_tuned"]["f1"],
        keras_metrics["test_tuned"]["roc_auc"],
        keras_metrics["test_tuned"]["pr_auc"],
        keras_metrics["test_tuned"]["precision"],
        keras_metrics["test_tuned"]["recall"],
    )

    ref_size = min(int(config["monitoring"]["reference_sample_size"]), len(x_train))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x_train), size=ref_size, replace=False)
    np.save(args.out / "reference_features.npy", x_train[idx])
    logger.info("Saved %d reference rows for drift monitoring", ref_size)


if __name__ == "__main__":
    main()
