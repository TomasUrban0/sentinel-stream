"""XGBoost gradient-boosted-tree classifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .base import FailurePredictor


class XGBoostClassifier(FailurePredictor):
    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        min_child_weight: int = 1,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "random_state": random_state,
        }
        self.model = None
        self.feature_names_: list[str] | None = None

    def _build(self, scale_pos_weight: float):
        from xgboost import XGBClassifier

        return XGBClassifier(
            **self.params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> XGBoostClassifier:
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        spw = n_neg / max(n_pos, 1.0)
        self.model = self._build(scale_pos_weight=spw)
        self.feature_names_ = feature_names
        self.model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoostClassifier has not been fitted.")
        return self.model.predict_proba(x)[:, 1]

    def feature_importance(self) -> dict[str, float] | None:
        if self.model is None or self.feature_names_ is None:
            return None
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names_, importances.tolist(), strict=True))

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / "xgboost.json"))
        meta = {
            "params": self.params,
            "threshold": self.threshold,
            "feature_names": self.feature_names_,
        }
        with open(path / "xgboost_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if self.feature_names_:
            joblib.dump(self.feature_names_, path / "xgboost_feature_names.joblib")

    @classmethod
    def load(cls, directory: str) -> XGBoostClassifier:
        from xgboost import XGBClassifier

        path = Path(directory)
        with open(path / "xgboost_meta.json") as f:
            meta = json.load(f)
        instance = cls(**meta["params"])
        instance.model = XGBClassifier()
        instance.model.load_model(str(path / "xgboost.json"))
        instance.threshold = float(meta.get("threshold", 0.5))
        instance.feature_names_ = meta.get("feature_names")
        return instance
