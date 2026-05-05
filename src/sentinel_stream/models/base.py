"""Common detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AnomalyDetector(ABC):
    """Minimal interface every detector implements."""

    name: str = "base"
    threshold: float = 0.0

    @abstractmethod
    def fit(self, x: np.ndarray) -> AnomalyDetector: ...

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray:
        """Return one anomaly score per row. Higher = more anomalous."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) > self.threshold).astype(int)

    @abstractmethod
    def save(self, directory: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> AnomalyDetector: ...
