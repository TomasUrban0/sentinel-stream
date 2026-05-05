"""Pydantic schemas for the API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    temperature: float = Field(..., description="Temperature in degrees Celsius")
    pressure: float = Field(..., description="Pressure in kPa")
    vibration: float = Field(..., description="Vibration amplitude")
    humidity: float = Field(..., description="Relative humidity (%)")
    timestamp: datetime | None = Field(default=None)


class PredictionResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    model: str
    warming_up: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    total_predictions: int
    total_anomalies: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    drift: dict[str, dict[str, float | bool]]
