"""FastAPI inference service."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from ..features.transformer import StreamingFeatureTransformer
from ..models.autoencoder import AutoencoderDetector
from ..models.base import AnomalyDetector
from ..models.isolation_forest import IsolationForestDetector
from ..monitoring.drift import DriftMonitor
from ..monitoring.metrics import MetricsRegistry
from ..utils.config import load_config
from ..utils.logger import get_logger
from .schemas import HealthResponse, MetricsResponse, PredictionResponse, SensorReading

logger = get_logger(__name__)


def _load_detector(name: str, artifacts_dir: Path) -> AnomalyDetector:
    if name == "autoencoder":
        return AutoencoderDetector.load(str(artifacts_dir))
    if name == "isolation_forest":
        return IsolationForestDetector.load(str(artifacts_dir))
    raise ValueError(f"Unknown model: {name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    artifacts_dir = Path(os.environ.get("SENTINEL_ARTIFACTS_DIR", "artifacts"))
    model_name = os.environ.get("SENTINEL_MODEL", config["api"]["default_model"])

    app.state.config = config
    app.state.transformer = StreamingFeatureTransformer(
        base_features=tuple(config["data"]["features"]),
        rolling_windows=tuple(config["data"]["rolling_windows"]),
        lag_steps=tuple(config["data"]["lag_steps"]),
    )
    app.state.metrics = MetricsRegistry()
    app.state.model_name = model_name
    app.state.detector = None
    app.state.drift_monitor: DriftMonitor | None = None

    if artifacts_dir.exists():
        try:
            app.state.detector = _load_detector(model_name, artifacts_dir)
            logger.info("Loaded detector '%s' from %s", model_name, artifacts_dir)
            ref_path = artifacts_dir / "reference_features.npy"
            if ref_path.exists():
                reference = np.load(ref_path)
                app.state.drift_monitor = DriftMonitor(
                    reference=reference,
                    feature_names=app.state.transformer.feature_columns,
                    threshold=config["monitoring"]["drift_threshold"],
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load detector: %s", exc)
    else:
        logger.warning("No artifacts directory at %s — API will return 503 for predictions.", artifacts_dir)

    yield


app = FastAPI(
    title="Sentinel Stream",
    description="Real-time anomaly detection for time-series data.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=app.state.detector is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading) -> PredictionResponse:
    detector: AnomalyDetector | None = app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    transformer: StreamingFeatureTransformer = app.state.transformer
    transformer.push(reading.model_dump(exclude={"timestamp"}), timestamp=reading.timestamp)
    features = transformer.transform()

    if features is None:
        return PredictionResponse(
            anomaly_score=0.0,
            is_anomaly=False,
            threshold=detector.threshold,
            model=detector.name,
            warming_up=True,
        )

    start = time.perf_counter()
    score = float(detector.score(features.reshape(1, -1))[0])
    latency_ms = (time.perf_counter() - start) * 1000.0

    is_anomaly = score > detector.threshold
    app.state.metrics.record(latency_ms, is_anomaly)
    if app.state.drift_monitor is not None:
        app.state.drift_monitor.observe(features)

    return PredictionResponse(
        anomaly_score=score,
        is_anomaly=is_anomaly,
        threshold=float(detector.threshold),
        model=detector.name,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    snapshot = app.state.metrics.snapshot()
    drift = app.state.drift_monitor.report() if app.state.drift_monitor else {}
    return MetricsResponse(**snapshot, drift=drift)
