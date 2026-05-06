"""Pydantic schemas for the API.

The fields mirror the AI4I 2020 predictive-maintenance dataset: six raw sensor
readings per production run plus a categorical product variant.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    type: Literal["L", "M", "H"] = Field(..., description="Product variant: Low / Medium / High quality")
    air_temperature_k: float = Field(..., description="Ambient air temperature (K)")
    process_temperature_k: float = Field(..., description="Process temperature (K)")
    rotational_speed_rpm: float = Field(..., description="Tool rotational speed (rpm)")
    torque_nm: float = Field(..., description="Cutting torque (Nm)")
    tool_wear_min: float = Field(..., description="Cumulative tool wear (minutes)")


class PredictionResponse(BaseModel):
    failure_probability: float
    will_fail: bool
    threshold: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    total_predictions: int
    total_flagged: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    drift: dict[str, dict[str, float | bool]]
