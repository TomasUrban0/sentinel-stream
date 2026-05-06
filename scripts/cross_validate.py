"""5-fold stratified cross-validation for the XGBoost classifier.

Documents that the headline single-split metrics in the README are not the
result of an unusually easy partition by reporting the mean and standard
deviation of every metric over five disjoint folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from sentinel_stream.data.ai4i_loader import TARGET_COLUMN, load_ai4i
from sentinel_stream.features.engineering import ALL_FEATURE_COLUMNS
from sentinel_stream.features.transformer import StreamingFeatureTransformer
from sentinel_stream.models.evaluation import best_f1_threshold
from sentinel_stream.models.xgboost_classifier import XGBoostClassifier
from sentinel_stream.utils.config import load_config
from sentinel_stream.utils.logger import get_logger

logger = get_logger("cross_validate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/ai4i"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cv_results.json"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_ai4i(args.data_root)
    transformer = StreamingFeatureTransformer()
    features = np.vstack([transformer.transform(r) for r in df.to_dict(orient="records")])
    labels = df[TARGET_COLUMN].to_numpy(dtype=int)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows: list[dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels), start=1):
        x_trval, x_test = features[train_idx], features[test_idx]
        y_trval, y_test = labels[train_idx], labels[test_idx]

        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold)
        val_idx, _ = next(inner.split(x_trval, y_trval))
        val_mask = np.zeros(len(x_trval), bool)
        val_mask[val_idx] = True

        clf = XGBoostClassifier(**config["model"]["xgboost"]).fit(
            x_trval[~val_mask], y_trval[~val_mask], feature_names=list(ALL_FEATURE_COLUMNS)
        )
        threshold, _ = best_f1_threshold(clf.predict_proba(x_trval[val_mask]), y_trval[val_mask])

        proba = clf.predict_proba(x_test)
        preds = (proba >= threshold).astype(int)
        rows.append(
            {
                "fold": fold,
                "f1": float(f1_score(y_test, preds)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "pr_auc": float(average_precision_score(y_test, proba)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds)),
                "tuned_threshold": float(threshold),
            }
        )
        logger.info(
            "fold %d  F1=%.3f  ROC-AUC=%.3f  PR-AUC=%.3f  P=%.3f  R=%.3f",
            fold,
            rows[-1]["f1"],
            rows[-1]["roc_auc"],
            rows[-1]["pr_auc"],
            rows[-1]["precision"],
            rows[-1]["recall"],
        )

    summary: dict[str, dict[str, float]] = {}
    for metric in ("f1", "roc_auc", "pr_auc", "precision", "recall"):
        values = np.array([row[metric] for row in rows])
        summary[metric] = {"mean": float(values.mean()), "std": float(values.std())}

    payload = {"folds": rows, "summary": summary, "n_splits": args.n_splits, "seed": args.seed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", args.out)
    logger.info("CV summary (mean ± std):")
    for metric, stats in summary.items():
        logger.info("  %-9s  %.3f ± %.3f", metric, stats["mean"], stats["std"])


if __name__ == "__main__":
    main()
