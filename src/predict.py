from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features import build_features
from src.model import ModelBundle, load_bundle


class Predictor:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.bundle: ModelBundle = load_bundle(self.model_path)

    @property
    def ready(self) -> bool:
        return True

    def predict_dataframe(self, raw: pd.DataFrame) -> pd.DataFrame:
        X = build_features(raw)
        probabilities = self.bundle.pipeline.predict_proba(X)[:, 1]
        labels = (probabilities >= 0.5).astype(bool)
        return pd.DataFrame(
            {
                "high_fare_probability": probabilities,
                "high_fare_prediction": labels,
                "fare_threshold": self.bundle.fare_threshold,
            },
            index=raw.index,
        )
