from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_features,
)


@dataclass
class ModelBundle:
    pipeline: Any
    fare_threshold: float
    metadata: dict[str, Any]


def make_pipeline(random_seed: int = 42) -> Pipeline:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        max_iter=400,
        solver="liblinear",
        random_state=random_seed,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("classifier", classifier)])


def evaluate_pipeline(pipeline: Pipeline, raw_frame: pd.DataFrame) -> dict[str, float]:
    X = build_features(raw_frame)
    y = raw_frame[TARGET_COLUMN].astype(int)
    score = pipeline.predict_proba(X)[:, 1]
    pred = (score >= 0.5).astype(int)
    metrics = {
        "rows": int(len(raw_frame)),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, score)),
    }
    if y.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, score))
    return metrics


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": bundle.pipeline,
            "fare_threshold": bundle.fare_threshold,
            "metadata": bundle.metadata,
        },
        path,
    )


def load_bundle(path: str | Path) -> ModelBundle:
    obj = joblib.load(path)
    return ModelBundle(
        pipeline=obj["pipeline"],
        fare_threshold=float(obj["fare_threshold"]),
        metadata=dict(obj.get("metadata", {})),
    )


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def artifact_metadata(**kwargs: Any) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_time": "trip pickup",
        "target_definition": "fare_amount >= training-set quantile threshold",
        **kwargs,
    }
