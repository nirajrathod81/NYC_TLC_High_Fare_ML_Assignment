from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_monthly_training_files
from src.features import FARE_COLUMN, add_target, build_features, clean_training_frame
from src.model import (
    ModelBundle,
    artifact_metadata,
    evaluate_pipeline,
    make_pipeline,
    save_bundle,
    save_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NYC TLC high-fare classifier")
    parser.add_argument("--train", nargs="+", required=True, help="Training parquet files")
    parser.add_argument("--validation", nargs="+", required=True, help="Validation parquet files")
    parser.add_argument("--test", nargs="+", required=True, help="Test parquet files")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--metrics-out", default="models/metrics.json")
    parser.add_argument("--threshold-quantile", type=float, default=0.90)
    parser.add_argument("--max-rows-per-file", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.5 < args.threshold_quantile < 1.0:
        raise ValueError("--threshold-quantile must be between 0.5 and 1.0")

    train_raw = clean_training_frame(
        load_monthly_training_files(args.train, args.max_rows_per_file, args.seed)
    )
    validation_raw = clean_training_frame(
        load_monthly_training_files(args.validation, args.max_rows_per_file, args.seed)
    )
    test_raw = clean_training_frame(
        load_monthly_training_files(args.test, args.max_rows_per_file, args.seed)
    )

    fare_threshold = float(train_raw[FARE_COLUMN].quantile(args.threshold_quantile))
    train = add_target(train_raw, fare_threshold)
    validation = add_target(validation_raw, fare_threshold)
    test = add_target(test_raw, fare_threshold)

    pipeline = make_pipeline(args.seed)
    pipeline.fit(build_features(train), train["high_fare"].astype(int))

    metrics = {
        "fare_threshold": fare_threshold,
        "threshold_quantile": args.threshold_quantile,
        "train": evaluate_pipeline(pipeline, train),
        "validation": evaluate_pipeline(pipeline, validation),
        "test": evaluate_pipeline(pipeline, test),
    }
    metadata = artifact_metadata(
        threshold_quantile=args.threshold_quantile,
        train_files=[Path(p).name for p in args.train],
        validation_files=[Path(p).name for p in args.validation],
        test_files=[Path(p).name for p in args.test],
        seed=args.seed,
        max_rows_per_file=args.max_rows_per_file,
        metrics=metrics,
    )
    save_bundle(ModelBundle(pipeline, fare_threshold, metadata), args.model_out)
    save_metrics(metrics, args.metrics_out)

    print(f"Saved model to {args.model_out}")
    print(f"Frozen high-fare threshold: ${fare_threshold:.2f}")
    print(f"Validation F1: {metrics['validation']['f1']:.4f}")
    print(f"Test F1: {metrics['test']['f1']:.4f}")


if __name__ == "__main__":
    main()
