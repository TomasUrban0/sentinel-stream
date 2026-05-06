"""Dense Keras classifier — included alongside XGBoost so the project keeps
demonstrating the TensorFlow / Keras stack listed on the CV.

Architecture is intentionally compact: two hidden layers with dropout, sigmoid
output, binary cross-entropy loss, class-imbalance handled via class weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import FailurePredictor


def _build_keras_model(input_dim: int, hidden_layers: list[int], dropout: float, lr: float):
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(42)
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


class KerasClassifier(FailurePredictor):
    name = "keras_dense"

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        epochs: int = 60,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
    ):
        self.hidden_layers = hidden_layers or [64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.scaler = StandardScaler()
        self.model = None
        self._input_dim: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> KerasClassifier:
        x_scaled = self.scaler.fit_transform(x)
        self._input_dim = x_scaled.shape[1]
        self.model = _build_keras_model(
            self._input_dim, self.hidden_layers, self.dropout, self.learning_rate
        )

        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        total = n_pos + n_neg
        class_weight = {0: total / (2.0 * n_neg), 1: total / (2.0 * max(n_pos, 1.0))}

        self.model.fit(
            x_scaled,
            y.astype(np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            shuffle=True,
            verbose=0,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("KerasClassifier has not been fitted.")
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled, verbose=0).reshape(-1)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "keras_classifier.keras")
        joblib.dump(self.scaler, path / "keras_scaler.joblib")
        meta = {
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "input_dim": self._input_dim,
            "threshold": self.threshold,
        }
        with open(path / "keras_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> KerasClassifier:
        from tensorflow import keras

        path = Path(directory)
        with open(path / "keras_meta.json") as f:
            meta = json.load(f)
        instance = cls(
            hidden_layers=meta["hidden_layers"],
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            learning_rate=meta["learning_rate"],
            dropout=meta["dropout"],
        )
        instance.model = keras.models.load_model(path / "keras_classifier.keras")
        instance.scaler = joblib.load(path / "keras_scaler.joblib")
        instance.threshold = float(meta.get("threshold", 0.5))
        instance._input_dim = meta["input_dim"]
        return instance
