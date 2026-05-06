"""Render evaluation plots used in the README and the notebook.

Reproduces the full evaluation: re-loads AI4I, redoes the same stratified
split with the same seed, re-engineers features, then loads the trained
classifiers from ``artifacts/`` and dumps PNGs into ``docs/img/``:

  - dataset_overview.png    : class balance + feature distributions by class
  - roc_curves.png          : ROC for both classifiers, with AUC in legend
  - pr_curves.png           : Precision-Recall for both classifiers
  - confusion_matrix.png    : XGBoost confusion matrix at the tuned threshold
  - feature_importance.png  : XGBoost gain-based feature importance
  - threshold_sweep.png     : precision / recall / F1 vs threshold (XGBoost)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from sentinel_stream.data.ai4i_loader import TARGET_COLUMN, load_ai4i
from sentinel_stream.features.engineering import (
    ALL_FEATURE_COLUMNS,
    build_features,
    get_or_create_spark,
)
from sentinel_stream.models.keras_classifier import KerasClassifier
from sentinel_stream.models.xgboost_classifier import XGBoostClassifier
from sentinel_stream.utils.config import load_config
from sentinel_stream.utils.logger import get_logger

logger = get_logger("generate_plots")

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True


def _spark_features(pdf, work_dir: Path, name: str):
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = work_dir / f"{name}.csv"
    pdf.to_csv(csv_path, index=False)
    spark = get_or_create_spark()
    sdf = spark.read.option("header", True).option("inferSchema", True).csv(str(csv_path))
    sdf = build_features(sdf)
    out = sdf.toPandas()
    spark.stop()
    return out


def _stratified_split(df, target, val_fraction, test_fraction, seed):
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df, test_size=test_fraction, random_state=seed, stratify=df[target]
    )
    relative_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=seed, stratify=train_val[target]
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def plot_dataset_overview(df, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    ax = axes[0, 0]
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    bars = ax.bar(["No failure", "Failure"], counts.values, color=["#3b82f6", "#ef4444"])
    for bar, value in zip(bars, counts.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({value / len(df):.1%})",
            ha="center",
            va="bottom",
        )
    ax.set_title("Class balance")
    ax.set_ylabel("rows")

    sensors = [
        ("air_temperature_k", "Air temperature [K]"),
        ("process_temperature_k", "Process temperature [K]"),
        ("rotational_speed_rpm", "Rotational speed [rpm]"),
        ("torque_nm", "Torque [Nm]"),
        ("tool_wear_min", "Tool wear [min]"),
    ]
    flat = axes.ravel()
    for ax, (col, title) in zip(flat[1:], sensors, strict=False):
        ax.hist(df.loc[df[TARGET_COLUMN] == 0, col], bins=40, alpha=0.55, label="No failure",
                color="#3b82f6", density=True)
        ax.hist(df.loc[df[TARGET_COLUMN] == 1, col], bins=40, alpha=0.7, label="Failure",
                color="#ef4444", density=True)
        ax.set_title(title)
    flat[1].legend(loc="upper right")
    fig.suptitle("AI4I 2020 — class balance and per-sensor distributions")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc(y_test, probas, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2.0, label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.0, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test partition")
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr(y_test, probas, baseline_rate, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probas.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision, lw=2.0, label=f"{name} (PR-AUC = {ap:.3f})")
    ax.axhline(baseline_rate, ls="--", color="grey", lw=1.0,
               label=f"random ({baseline_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves on the held-out test partition")
    ax.legend(loc="lower left")
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(y_test, y_pred, out_path: Path, model_name: str, threshold: float) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["No failure", "Failure"])
    ax.set_yticks([0, 1], ["No failure", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — confusion matrix\n(threshold = {threshold:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(importance: dict[str, float], out_path: Path) -> None:
    items = sorted(importance.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(names, values, color="#1f77b4")
    ax.set_xlabel("Gain-based importance")
    ax.set_title("XGBoost feature importance")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_threshold_sweep(y_test, proba, out_path: Path, model_name: str) -> None:
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions = []
    recalls = []
    f1s = []
    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    best_idx = int(np.argmax(f1s))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="precision", color="#1f77b4")
    ax.plot(thresholds, recalls, label="recall", color="#ff7f0e")
    ax.plot(thresholds, f1s, label="F1", color="#2ca02c", lw=2.5)
    ax.axvline(thresholds[best_idx], ls="--", color="grey",
               label=f"argmax F1 = {thresholds[best_idx]:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{model_name} — precision / recall / F1 vs threshold (test set)")
    ax.legend(loc="lower left")
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ai4i"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    df = load_ai4i(args.data_root)
    plot_dataset_overview(df, args.out / "dataset_overview.png")
    logger.info("Wrote dataset_overview.png")

    split_cfg = config["data"]["split"]
    _, _, test_df = _stratified_split(
        df,
        target=TARGET_COLUMN,
        val_fraction=float(split_cfg["val_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
        seed=int(split_cfg["seed"]),
    )

    work_dir = args.artifacts / "_spark_inputs"
    test_pdf = _spark_features(test_df, work_dir, "test_for_plots")
    cols = list(ALL_FEATURE_COLUMNS)
    x_test = test_pdf[cols].to_numpy(dtype=np.float32)
    y_test = test_pdf[TARGET_COLUMN].to_numpy(dtype=int)

    xgb = XGBoostClassifier.load(str(args.artifacts))
    keras_clf = KerasClassifier.load(str(args.artifacts))
    proba_xgb = xgb.predict_proba(x_test)
    proba_keras = keras_clf.predict_proba(x_test)

    probas = {"XGBoost": proba_xgb, "Keras dense": proba_keras}
    plot_roc(y_test, probas, args.out / "roc_curves.png")
    logger.info("Wrote roc_curves.png")

    baseline_rate = float(y_test.mean())
    plot_pr(y_test, probas, baseline_rate, args.out / "pr_curves.png")
    logger.info("Wrote pr_curves.png")

    preds_xgb = (proba_xgb >= xgb.threshold).astype(int)
    plot_confusion(
        y_test,
        preds_xgb,
        args.out / "confusion_matrix.png",
        model_name="XGBoost",
        threshold=xgb.threshold,
    )
    logger.info("Wrote confusion_matrix.png")

    metrics_path = args.artifacts / "metrics.json"
    importance: dict[str, float] = {}
    if metrics_path.exists():
        importance = json.loads(metrics_path.read_text()).get(
            "xgboost_feature_importance", {}
        )
    if importance:
        plot_feature_importance(importance, args.out / "feature_importance.png")
        logger.info("Wrote feature_importance.png")

    plot_threshold_sweep(
        y_test, proba_xgb, args.out / "threshold_sweep.png", model_name="XGBoost"
    )
    logger.info("Wrote threshold_sweep.png")


if __name__ == "__main__":
    main()
