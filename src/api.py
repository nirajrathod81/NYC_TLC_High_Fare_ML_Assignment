from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.predict import Predictor
from src.schemas import HealthResponse, Prediction, PredictionRequest, PredictionResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "high-fare-v1")

app = FastAPI(
    title="NYC TLC High-Fare Prediction API",
    version="1.0.0",
    description="Pickup-time high-fare classification using the same artifact as batch scoring.",
)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(MODEL_PATH)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse()


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    try:
        get_predictor()
        return HealthResponse()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        rows = [record.model_dump() for record in request.records]
        frame = pd.DataFrame(rows)
        scored = get_predictor().predict_dataframe(frame)
        predictions = [Prediction(**row) for row in scored.to_dict(orient="records")]
        return PredictionResponse(predictions=predictions, model_version=MODEL_VERSION)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact not found: {exc}") from exc
