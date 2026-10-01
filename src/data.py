from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence
import pandas as pd

from src.features import TRAINING_COLUMNS


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    # pandas delegates Parquet IO to pyarrow installed from requirements.txt.
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def load_monthly_training_files(
    paths: Sequence[str | Path],
    max_rows_per_file: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Read only required columns, sampling each month deterministically when requested."""
    frames: list[pd.DataFrame] = []
    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = _read_parquet_columns(path, TRAINING_COLUMNS)
        if max_rows_per_file and len(frame) > max_rows_per_file:
            frame = frame.sample(n=max_rows_per_file, random_state=random_seed)
        frame["__source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("No input files supplied")
    return pd.concat(frames, ignore_index=True)


def read_scoring_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .parquet or .csv")


def iter_scoring_batches(path: str | Path, batch_size: int = 50_000) -> Iterator[pd.DataFrame]:
    """Stream scoring inputs so batch inference does not require loading the whole file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, chunksize=batch_size)
        return
    if suffix != ".parquet":
        raise ValueError("Input must be .parquet or .csv")

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        yield record_batch.to_pandas()


def write_scored_output(frames: list[pd.DataFrame], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output.suffix.lower() == ".parquet":
        result.to_parquet(output, index=False, engine="pyarrow")
    elif output.suffix.lower() == ".csv":
        result.to_csv(output, index=False)
    else:
        raise ValueError("Output must be .parquet or .csv")
