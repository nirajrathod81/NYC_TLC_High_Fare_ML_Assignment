from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data import iter_scoring_batches
from src.predict import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score NYC TLC trip rows")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--batch-size", type=int, default=50_000)
    return parser.parse_args()


def score_file(input_path: str, output_path: str, model_path: str, batch_size: int) -> int:
    predictor = Predictor(model_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[pd.DataFrame] = []
    rows = 0

    for raw in iter_scoring_batches(input_path, batch_size=batch_size):
        scores = predictor.predict_dataframe(raw)
        keep = raw.reset_index(drop=True).copy()
        scores = scores.reset_index(drop=True)
        chunks.append(pd.concat([keep, scores], axis=1))
        rows += len(raw)

    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if output.suffix.lower() == ".parquet":
        result.to_parquet(output, index=False, engine="pyarrow")
    elif output.suffix.lower() == ".csv":
        result.to_csv(output, index=False)
    else:
        raise ValueError("--output must end in .parquet or .csv")
    return rows


def main() -> None:
    args = parse_args()
    rows = score_file(args.input, args.output, args.model, args.batch_size)
    print(f"Scored {rows:,} rows -> {args.output}")


if __name__ == "__main__":
    main()
