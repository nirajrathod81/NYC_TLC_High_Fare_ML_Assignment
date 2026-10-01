import pandas as pd

from src.features import build_features, clean_training_frame


def raw_frame():
    return pd.DataFrame(
        {
            "VendorID": [1],
            "tpep_pickup_datetime": ["2024-01-06 22:15:00"],
            "passenger_count": [2.0],
            "RatecodeID": [1.0],
            "store_and_fwd_flag": ["N"],
            "PULocationID": [161],
            "fare_amount": [42.0],
        }
    )


def test_build_features_derives_pickup_time_fields():
    features = build_features(raw_frame())
    assert int(features.loc[0, "pickup_hour"]) == 22
    assert int(features.loc[0, "pickup_dayofweek"]) == 5
    assert str(features.loc[0, "is_weekend"]) == "1"


def test_clean_training_frame_drops_negative_fare():
    frame = pd.concat([raw_frame(), raw_frame()], ignore_index=True)
    frame.loc[1, "fare_amount"] = -5
    clean = clean_training_frame(frame)
    assert len(clean) == 1
