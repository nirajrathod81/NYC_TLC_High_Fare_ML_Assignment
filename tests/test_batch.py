import pandas as pd

from src.features import add_target, build_features, clean_training_frame
from src.model import ModelBundle, make_pipeline, save_bundle
from src.predict import Predictor


def training_frame():
    rows = []
    for i in range(80):
        rows.append(
            {
                "VendorID": 1 + (i % 2),
                "tpep_pickup_datetime": f"2024-01-{1 + (i % 20):02d} {i % 24:02d}:00:00",
                "passenger_count": 1 + (i % 3),
                "RatecodeID": 1,
                "store_and_fwd_flag": "N",
                "PULocationID": 100 + (i % 20),
                "fare_amount": 10 + i,
            }
        )
    return pd.DataFrame(rows)


def test_predictor_returns_expected_columns(tmp_path):
    frame = clean_training_frame(training_frame())
    threshold = float(frame["fare_amount"].quantile(0.9))
    labeled = add_target(frame, threshold)
    pipeline = make_pipeline()
    pipeline.fit(build_features(labeled), labeled["high_fare"])
    path = tmp_path / "model.joblib"
    save_bundle(ModelBundle(pipeline, threshold, {"test": True}), path)

    scored = Predictor(path).predict_dataframe(frame.head(3))
    assert list(scored.columns) == [
        "high_fare_probability",
        "high_fare_prediction",
        "fare_threshold",
    ]
    assert len(scored) == 3
