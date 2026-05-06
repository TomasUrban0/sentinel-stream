"""FastAPI inference service for predictive maintenance."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from ..features.transformer import StreamingFeatureTransformer
from ..models.base import FailurePredictor
from ..models.keras_classifier import KerasClassifier
from ..models.xgboost_classifier import XGBoostClassifier
from ..monitoring.drift import DriftMonitor
from ..monitoring.metrics import MetricsRegistry
from ..utils.config import load_config
from ..utils.logger import get_logger
from .schemas import HealthResponse, MetricsResponse, PredictionResponse, SensorReading

logger = get_logger(__name__)


def _load_predictor(name: str, artifacts_dir: Path) -> FailurePredictor:
    if name == "xgboost":
        return XGBoostClassifier.load(str(artifacts_dir))
    if name == "keras_dense":
        return KerasClassifier.load(str(artifacts_dir))
    raise ValueError(f"Unknown model: {name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    artifacts_dir = Path(os.environ.get("SENTINEL_ARTIFACTS_DIR", "artifacts"))
    model_name = os.environ.get("SENTINEL_MODEL", config["api"]["default_model"])

    app.state.config = config
    app.state.transformer = StreamingFeatureTransformer()
    app.state.metrics = MetricsRegistry()
    app.state.model_name = model_name
    app.state.predictor = None
    app.state.drift_monitor: DriftMonitor | None = None

    if artifacts_dir.exists():
        try:
            app.state.predictor = _load_predictor(model_name, artifacts_dir)
            logger.info("Loaded predictor '%s' from %s", model_name, artifacts_dir)
            ref_path = artifacts_dir / "reference_features.npy"
            if ref_path.exists():
                reference = np.load(ref_path)
                app.state.drift_monitor = DriftMonitor(
                    reference=reference,
                    feature_names=app.state.transformer.feature_columns,
                    threshold=config["monitoring"]["drift_threshold"],
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load predictor: %s", exc)
    else:
        logger.warning(
            "No artifacts directory at %s — API will return 503 for predictions.", artifacts_dir
        )

    yield


app = FastAPI(
    title="Sentinel Stream",
    description="Predictive maintenance: real-time failure prediction from sensor telemetry.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=app.state.predictor is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading) -> PredictionResponse:
    predictor: FailurePredictor | None = app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    transformer: StreamingFeatureTransformer = app.state.transformer
    features = transformer.transform(reading.model_dump())

    start = time.perf_counter()
    prob = float(predictor.predict_proba(features.reshape(1, -1))[0])
    latency_ms = (time.perf_counter() - start) * 1000.0

    will_fail = prob >= predictor.threshold
    app.state.metrics.record(latency_ms, will_fail)
    if app.state.drift_monitor is not None:
        app.state.drift_monitor.observe(features)

    return PredictionResponse(
        failure_probability=prob,
        will_fail=will_fail,
        threshold=float(predictor.threshold),
        model=predictor.name,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    snapshot = app.state.metrics.snapshot()
    drift = app.state.drift_monitor.report() if app.state.drift_monitor else {}
    return MetricsResponse(
        total_predictions=snapshot["total_predictions"],
        total_flagged=snapshot["total_anomalies"],
        p50_ms=snapshot["p50_ms"],
        p95_ms=snapshot["p95_ms"],
        p99_ms=snapshot["p99_ms"],
        drift=drift,
    )
