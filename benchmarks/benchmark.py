from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from pathlib import Path

import httpx
import pandas as pd
import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the running FastAPI endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--input", required=True, help="Parquet or CSV containing TLC rows")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--output", default="benchmarks/results.json")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def load_payload(path: str) -> dict:
    p = Path(path)
    columns = [
        "VendorID",
        "tpep_pickup_datetime",
        "passenger_count",
        "RatecodeID",
        "store_and_fwd_flag",
        "PULocationID",
    ]
    if p.suffix.lower() == ".parquet":
        frame = pd.read_parquet(p, columns=columns, engine="pyarrow").head(1)
    else:
        frame = pd.read_csv(p, usecols=columns, nrows=1)
    row = frame.iloc[0].to_dict()
    row["tpep_pickup_datetime"] = pd.Timestamp(row["tpep_pickup_datetime"]).isoformat()
    for key, value in list(row.items()):
        if pd.isna(value):
            row[key] = None
        elif hasattr(value, "item"):
            row[key] = value.item()
    return {"records": [row]}


def one_request(url: str, payload: dict) -> float:
    start = time.perf_counter()
    with httpx.Client(timeout=5.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input)

    for _ in range(args.warmup):
        one_request(args.url, payload)

    rss_before = None
    process = None
    if args.server_pid:
        process = psutil.Process(args.server_pid)
        rss_before = process.memory_info().rss

    started = time.perf_counter()
    if args.concurrency <= 1:
        latencies = [one_request(args.url, payload) for _ in range(args.requests)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(one_request, args.url, payload) for _ in range(args.requests)]
            latencies = [f.result() for f in futures]
    elapsed = time.perf_counter() - started

    rss_after = process.memory_info().rss if process else None
    result = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_seconds": elapsed,
        "throughput_requests_per_second": args.requests / elapsed,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "server_rss_mb_before": None if rss_before is None else rss_before / 1024 / 1024,
        "server_rss_mb_after": None if rss_after is None else rss_after / 1024 / 1024,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
