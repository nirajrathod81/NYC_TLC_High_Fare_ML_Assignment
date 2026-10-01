from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class TripFeatures(BaseModel):
    VendorID: int | None = None
    tpep_pickup_datetime: datetime
    passenger_count: float | None = Field(default=None, ge=0, le=8)
    RatecodeID: float | None = None
    store_and_fwd_flag: str | None = None
    PULocationID: int = Field(ge=1, le=265)

    @field_validator("store_and_fwd_flag")
    @classmethod
    def normalize_flag(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.upper().strip()
        if value not in {"Y", "N"}:
            raise ValueError("store_and_fwd_flag must be Y or N")
        return value


class PredictionRequest(BaseModel):
    records: list[TripFeatures] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    high_fare_probability: float
    high_fare_prediction: bool
    fare_threshold: float


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
