"""Distribution drift detection using the Kolmogorov-Smirnov two-sample test."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class DriftMonitor:
    """Compares the most recent live samples against a fixed reference window."""

    reference: np.ndarray  # shape: (n_ref, n_features)
    feature_names: list[str]
    threshold: float = 0.15
    window_size: int = 1000
    _live: deque[np.ndarray] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._live = deque(maxlen=self.window_size)

    def observe(self, x: np.ndarray) -> None:
        """Add a single feature vector to the live buffer."""
        self._live.append(np.asarray(x, dtype=np.float32))

    def report(self) -> dict[str, dict[str, float | bool]]:
        """Return per-feature KS statistic and a drift flag."""
        if len(self._live) < max(50, self.window_size // 4):
            return {}

        live = np.stack(list(self._live), axis=0)
        out: dict[str, dict[str, float | bool]] = {}
        for i, name in enumerate(self.feature_names):
            stat, pvalue = ks_2samp(self.reference[:, i], live[:, i])
            out[name] = {
                "ks_statistic": float(stat),
                "p_value": float(pvalue),
                "drift": bool(stat > self.threshold),
            }
        return out
