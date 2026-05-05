"""Pydantic schemas for the API.

The eight sensor channels mirror the Skoltech Anomaly Benchmark (SKAB) feed:
two RMS accelerometers, current, pressure, temperature, thermocouple, voltage,
and a volumetric flow-rate RMS reading.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    accelerometer_1_rms: float = Field(..., description="Accelerometer 1 RMS")
    accelerometer_2_rms: float = Field(..., description="Accelerometer 2 RMS")
    current: float = Field(..., description="Motor current (A)")
    pressure: float = Field(..., description="Line pressure (bar)")
    temperature: float = Field(..., description="Fluid temperature (deg C)")
    thermocouple: float = Field(..., description="Thermocouple reading (deg C)")
    voltage: float = Field(..., description="Motor voltage (V)")
    volume_flow_rate: float = Field(..., description="Volumetric flow-rate RMS")
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
