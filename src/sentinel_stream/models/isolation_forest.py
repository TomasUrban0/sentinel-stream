"""Isolation Forest baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, contamination: float = 0.02, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: IsolationForest | None = None
        self.threshold = 0.0

    def fit(self, x: np.ndarray) -> IsolationForestDetector:
        x_scaled = self.scaler.fit_transform(x)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(x_scaled)
        # Use the model's own decision boundary (negated, so higher = more anomalous).
        self.threshold = float(-self.model.offset_)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        x_scaled = self.scaler.transform(x)
        return -self.model.score_samples(x_scaled)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "iforest.joblib")
        joblib.dump(self.scaler, path / "iforest_scaler.joblib")
        with open(path / "iforest_meta.json", "w") as f:
            json.dump({"threshold": self.threshold}, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> IsolationForestDetector:
        path = Path(directory)
        instance = cls()
        instance.model = joblib.load(path / "iforest.joblib")
        instance.scaler = joblib.load(path / "iforest_scaler.joblib")
        with open(path / "iforest_meta.json") as f:
            instance.threshold = float(json.load(f)["threshold"])
        return instance
