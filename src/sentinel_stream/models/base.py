"""Common predictor interface.

Every classifier in this project exposes the same four methods so the
training loop, the FastAPI service, and the evaluation utilities can stay
model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FailurePredictor(ABC):
    """Minimal interface every supervised classifier implements."""

    name: str = "base"
    threshold: float = 0.5

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> FailurePredictor: ...

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Probability of the positive (failure) class, shape (n,)."""

    def score(self, x: np.ndarray) -> np.ndarray:
        """Alias for ``predict_proba`` so legacy callers keep working."""
        return self.predict_proba(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= self.threshold).astype(int)

    @abstractmethod
    def save(self, directory: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> FailurePredictor: ...
