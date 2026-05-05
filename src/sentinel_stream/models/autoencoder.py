"""Dense autoencoder anomaly detector."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import AnomalyDetector


def _build_keras_model(input_dim: int, hidden_layers: list[int], learning_rate: float):
    # Imported lazily so importing the package doesn't pull TensorFlow in.
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


class AutoencoderDetector(AnomalyDetector):
    name = "autoencoder"

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        threshold_percentile: float = 99.0,
    ):
        self.hidden_layers = hidden_layers or [16, 8, 4, 8, 16]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = 0.0
        self._input_dim: int | None = None

    def fit(self, x: np.ndarray) -> AutoencoderDetector:
        # Pin the global random seed so two runs on the same data produce
        # reproducible weights, scores, and tuned thresholds. Without this the
        # reported metrics drift by ±0.02 F1 between runs.
        from tensorflow import keras

        keras.utils.set_random_seed(42)

        x_scaled = self.scaler.fit_transform(x)
        self._input_dim = x_scaled.shape[1]
        self.model = _build_keras_model(self._input_dim, self.hidden_layers, self.learning_rate)
        self.model.fit(
            x_scaled,
            x_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0,
        )
        train_scores = self.score(x)
        self.threshold = float(np.percentile(train_scores, self.threshold_percentile))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        x_scaled = self.scaler.transform(x)
        reconstructed = self.model.predict(x_scaled, verbose=0)
        return np.mean(np.square(x_scaled - reconstructed), axis=1)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "autoencoder.keras")
        joblib.dump(self.scaler, path / "autoencoder_scaler.joblib")
        meta = {
            "threshold": self.threshold,
            "hidden_layers": self.hidden_layers,
            "threshold_percentile": self.threshold_percentile,
            "input_dim": self._input_dim,
        }
        with open(path / "autoencoder_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> AutoencoderDetector:
        from tensorflow import keras

        path = Path(directory)
        with open(path / "autoencoder_meta.json") as f:
            meta = json.load(f)
        instance = cls(
            hidden_layers=meta["hidden_layers"],
            threshold_percentile=meta["threshold_percentile"],
        )
        instance.model = keras.models.load_model(path / "autoencoder.keras")
        instance.scaler = joblib.load(path / "autoencoder_scaler.joblib")
        instance.threshold = float(meta["threshold"])
        instance._input_dim = meta["input_dim"]
        return instance
