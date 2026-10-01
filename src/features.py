from __future__ import annotations

from typing import Iterable
import pandas as pd

TARGET_COLUMN = "high_fare"
FARE_COLUMN = "fare_amount"
PICKUP_COLUMN = "tpep_pickup_datetime"

RAW_FEATURE_COLUMNS = [
    "VendorID",
    PICKUP_COLUMN,
    "passenger_count",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
]

TRAINING_COLUMNS = RAW_FEATURE_COLUMNS + [FARE_COLUMN]

MODEL_FEATURE_COLUMNS = [
    "VendorID",
    "passenger_count",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "VendorID",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
]

NUMERIC_FEATURES = ["passenger_count"]

LEAKAGE_EXCLUDED_COLUMNS = {
    "fare_amount": "Defines the target.",
    "total_amount": "Direct post-trip monetary outcome.",
    "tip_amount": "Post-trip monetary outcome.",
    "tolls_amount": "Accumulated during the trip.",
    "extra": "Post-trip monetary component.",
    "mta_tax": "Post-trip monetary component.",
    "improvement_surcharge": "Post-trip monetary component.",
    "congestion_surcharge": "Post-trip monetary component.",
    "Airport_fee": "Monetary component associated with the completed trip.",
    "tpep_dropoff_datetime": "Only known after trip completion.",
    "DOLocationID": "Excluded to keep the contract strictly pickup-time only.",
    "trip_distance": "Final metered distance is not known at pickup.",
    "payment_type": "Recorded at trip completion/payment time.",
}


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, leakage-safe data-quality filters before training."""
    require_columns(df, TRAINING_COLUMNS)
    out = df.copy()
    out[PICKUP_COLUMN] = pd.to_datetime(out[PICKUP_COLUMN], errors="coerce")
    out[FARE_COLUMN] = pd.to_numeric(out[FARE_COLUMN], errors="coerce")
    out["passenger_count"] = pd.to_numeric(out["passenger_count"], errors="coerce")
    out["PULocationID"] = pd.to_numeric(out["PULocationID"], errors="coerce")

    # Keep target-defining fare values that are physically meaningful.
    out = out[out[PICKUP_COLUMN].notna()]
    out = out[out[FARE_COLUMN].notna() & (out[FARE_COLUMN] > 0)]
    out = out[out["PULocationID"].notna() & out["PULocationID"].between(1, 265)]

    # Passenger count is optional in TLC records; impossible values are mapped to missing.
    invalid_passenger = ~out["passenger_count"].between(0, 8) & out["passenger_count"].notna()
    out.loc[invalid_passenger, "passenger_count"] = pd.NA
    return out.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw pickup-time fields into the exact model feature contract."""
    require_columns(df, RAW_FEATURE_COLUMNS)
    out = df[RAW_FEATURE_COLUMNS].copy()
    out[PICKUP_COLUMN] = pd.to_datetime(out[PICKUP_COLUMN], errors="coerce")
    if out[PICKUP_COLUMN].isna().any():
        bad = int(out[PICKUP_COLUMN].isna().sum())
        raise ValueError(f"{bad} rows contain invalid {PICKUP_COLUMN} values")

    out["pickup_hour"] = out[PICKUP_COLUMN].dt.hour.astype("int16")
    out["pickup_dayofweek"] = out[PICKUP_COLUMN].dt.dayofweek.astype("int16")
    out["pickup_month"] = out[PICKUP_COLUMN].dt.month.astype("int16")
    out["is_weekend"] = (out["pickup_dayofweek"] >= 5).astype("int8")
    out = out.drop(columns=[PICKUP_COLUMN])

    # Normalize categorical-like fields to strings so online and batch paths match.
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("string").fillna("UNKNOWN")
    out["passenger_count"] = pd.to_numeric(out["passenger_count"], errors="coerce")
    return out[MODEL_FEATURE_COLUMNS]


def add_target(df: pd.DataFrame, fare_threshold: float) -> pd.DataFrame:
    require_columns(df, [FARE_COLUMN])
    out = df.copy()
    out[TARGET_COLUMN] = (out[FARE_COLUMN] >= fare_threshold).astype("int8")
    return out
