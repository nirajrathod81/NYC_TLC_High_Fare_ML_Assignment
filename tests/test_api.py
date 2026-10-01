import os
import pandas as pd
from fastapi.testclient import TestClient

from src.features import add_target, build_features, clean_training_frame
from src.model import ModelBundle, make_pipeline, save_bundle


def make_artifact(path):
    rows = []
    for i in range(60):
        rows.append(
            {
                "VendorID": 1 + (i % 2),
                "tpep_pickup_datetime": f"2024-02-{1 + (i % 20):02d} {i % 24:02d}:00:00",
                "passenger_count": 1,
                "RatecodeID": 1,
                "store_and_fwd_flag": "N",
                "PULocationID": 120 + (i % 10),
                "fare_amount": 8 + i,
            }
        )
    frame = clean_training_frame(pd.DataFrame(rows))
    threshold = float(frame["fare_amount"].quantile(0.9))
    labeled = add_target(frame, threshold)
    pipeline = make_pipeline()
    pipeline.fit(build_features(labeled), labeled["high_fare"])
    save_bundle(ModelBundle(pipeline, threshold, {}), path)


def test_predict_endpoint(tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    make_artifact(artifact)
    monkeypatch.setenv("MODEL_PATH", str(artifact))

    import importlib
    import src.api as api
    importlib.reload(api)
    client = TestClient(api.app)

    payload = {
        "records": [
            {
                "VendorID": 2,
                "tpep_pickup_datetime": "2024-07-15T18:30:00",
                "passenger_count": 1,
                "RatecodeID": 1,
                "store_and_fwd_flag": "N",
                "PULocationID": 161,
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 1
    assert "high_fare_probability" in body["predictions"][0]
