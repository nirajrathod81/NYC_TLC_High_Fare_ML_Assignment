# NYC TLC High-Fare Prediction — One Model, Two Serving Paths

A production-oriented ML implementation for **high-fare Yellow Taxi trip prediction** with one shared model artifact used by both:

1. a low-latency FastAPI REST service, and
2. an offline batch scorer for CSV/Parquet inputs.

## 1. Problem definition

**Prediction time:** trip pickup.

**Target:** `high_fare = fare_amount >= q90(training fare_amount)`.

The 90th-percentile threshold is calculated **only from the training months** and frozen inside the serialized model artifact. Validation, test, online inference, and batch inference all reuse the same frozen threshold.

This formulation intentionally favors a clear, leakage-safe production contract over maximizing offline accuracy.

## 2. Data split

The supplied 2024 Yellow Taxi monthly files are used chronologically:

| Split | Months | Purpose |
|---|---|---|
| Train | Jan-May 2024 | Fit preprocessing + classifier and derive the fare threshold |
| Validation | Jun 2024 | Model-selection / sanity-check metrics |
| Test | Jul 2024 | Final out-of-time evaluation and batch demo |

For local development, training defaults to a deterministic sample of **200,000 rows per monthly file**. This keeps the project practical on a laptop while still using up to 1,000,000 training examples. Change `--max-rows-per-file` if more compute is available.

## 3. Leakage policy

The model is defined as a **pickup-time** prediction. Therefore it only uses fields that are plausibly available at pickup:

- `VendorID`
- `tpep_pickup_datetime` -> hour/day-of-week/month/weekend features
- `passenger_count`
- `RatecodeID`
- `store_and_fwd_flag`
- `PULocationID`

Explicitly excluded post-outcome or trip-completion fields include:

- `fare_amount` (target definition)
- `total_amount`, `tip_amount`, `tolls_amount`
- `extra`, `mta_tax`, `improvement_surcharge`, `congestion_surcharge`, `Airport_fee`
- `tpep_dropoff_datetime`
- `payment_type`
- final `trip_distance`
- `DOLocationID` (excluded to keep the contract strictly pickup-time only)

A different product requirement could justify destination-aware prediction, but that is intentionally outside this version.

## 4. Repository structure

text
ml-serving-assignment/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── model.py
│   ├── schemas.py
│   ├── predict.py
│   ├── api.py
│   └── batch.py
│
├── models/
│   └── model.joblib        # created by training
│
├── benchmarks/
│   ├── benchmark.py
│   └── results.json        # created by benchmark run
│
├── tests/
│   ├── test_features.py
│   ├── test_api.py
│   └── test_batch.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── architecture.png


## 5. Setup

Python 3.14.2 is recommended.


python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt


## 6. Train


python -m src.train \
  --train \
    data/raw/yellow_tripdata_2024-01.parquet \
    data/raw/yellow_tripdata_2024-02.parquet \
    data/raw/yellow_tripdata_2024-03.parquet \
    data/raw/yellow_tripdata_2024-04.parquet \
    data/raw/yellow_tripdata_2024-05.parquet \
  --validation data/raw/yellow_tripdata_2024-06.parquet \
  --test data/raw/yellow_tripdata_2024-07.parquet \
  --model-out models/model.joblib \
  --metrics-out models/metrics.json \
  --threshold-quantile 0.90 \
  --max-rows-per-file 200000


Outputs:

- `models/model.joblib` — preprocessing, classifier, frozen threshold, metadata
- `models/metrics.json` — train/validation/test metrics

The artifact is self-contained, so API and batch scoring cannot silently drift to different preprocessing rules.

## 7. Run the REST API


uvicorn src.api:app --host 0.0.0.0 --port 8000


Health endpoints:


curl http://localhost:8000/healthz
curl http://localhost:8000/readyz


Single-record request:


curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "records": [{
      "VendorID": 2,
      "tpep_pickup_datetime": "2024-07-15T18:30:00",
      "passenger_count": 1,
      "RatecodeID": 1,
      "store_and_fwd_flag": "N",
      "PULocationID": 161
    }]
  }'


Example response:

{
  "predictions": [
    {
      "high_fare_probability": 0.37,
      "high_fare_prediction": false,
      "fare_threshold": 42.50
    }
  ],
  "model_version": "high-fare-v1"
}


`fare_threshold` above is only the response schema example; the real value is learned from Jan-May training data and returned from the model artifact.

The endpoint accepts 1-100 records per request.

## 8. Batch scoring

Score the complete July file:


python -m src.batch \
  --input data/raw/yellow_tripdata_2024-07.parquet \
  --output data/processed/yellow_tripdata_2024-07_scored.parquet \
  --model models/model.joblib \
  --batch-size 50000


The batch scorer streams Parquet record batches and appends:

- `high_fare_probability`
- `high_fare_prediction`
- `fare_threshold`

The same `Predictor` and `build_features()` implementation is used by both online and batch paths.

## 9. Tests


pytest -q


Tests use synthetic rows and do not depend on the large TLC files.

## 10. Docker

Train the model first so `models/model.joblib` exists, then:


docker build -t tlc-high-fare:local .
docker run --rm -p 8000:8000 tlc-high-fare:local


Test:


curl http://localhost:8000/readyz


### Batch scoring in Docker

Mount data and model directories:


docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  tlc-high-fare:local \
  python -m src.batch \
    --input data/raw/yellow_tripdata_2024-07.parquet \
    --output data/processed/july_scored.parquet


## 11. Online benchmark

Start the API, obtain the server PID if you want RSS measurements, and run:


python benchmarks/benchmark.py \
  --input data/raw/yellow_tripdata_2024-07.parquet \
  --requests 500 \
  --warmup 25 \
  --concurrency 1 \
  --output benchmarks/results.json


Optionally add `--server-pid <PID>` for before/after server RSS observations.

Report at minimum:

- p50 latency
- p95 latency
- p99 latency
- requests/second
- server RSS if available

The assignment's p95 target should be validated on the evaluator's machine rather than claimed before running the benchmark.

## 12. Shared vs separate logic

### Shared

- input feature contract
- feature engineering
- serialized preprocessing pipeline
- classifier
- target threshold metadata
- prediction implementation

### Separate

- HTTP/Pydantic validation and response formatting
- batch file reading/writing and chunking
- operational concerns such as concurrency vs throughput

This keeps the two serving paths semantically identical while allowing each to optimize for its workload.

## 13. Deployment readiness notes

### Kubernetes starting point

Suggested initial resources for one API worker:

yaml
resources:
  requests:
    cpu: "250m"
    memory: "256Mi"
  limits:
    cpu: "1000m"
    memory: "512Mi"


These are starting values, not measured production requirements. Tune them from actual benchmark and load-test results.

### Probes

- **Liveness:** `GET /healthz` — verifies the process is responsive.
- **Readiness:** `GET /readyz` — verifies the model artifact can be loaded.

### Autoscaling

Start with CPU-based horizontal autoscaling, then consider request-rate or latency metrics if available. Avoid excessive Uvicorn worker counts inside a memory-constrained pod because each worker can duplicate model memory.

### Batch workloads

For larger offline jobs, run batch scoring as a Kubernetes Job rather than routing it through the online API. This isolates latency-sensitive online traffic from throughput-oriented batch work.

## 14. Assumption log

1. “High fare” means the top 10% of positive `fare_amount` values in the training period.
2. The prediction is made at pickup, before final route, distance, tolls, payment, or fare are observed.
3. `PULocationID` values outside 1-265 are treated as invalid training rows.
4. Passenger count values outside 0-8 are treated as missing rather than dropping the row.
5. The month-based split is chronological to better approximate future-data performance than a random split.
6. Local training samples each month to control memory and runtime; the seed is fixed for reproducibility.
7. The raw TLC data remains outside source control.

## 15. Design rationale

A logistic-regression pipeline is deliberately used here. The take-home emphasizes serving correctness, schema handling, reproducibility, latency, and operational maturity more than model novelty. One-hot encoded pickup context is fast to train, fast to serve, easy to inspect, and an appropriate baseline for demonstrating the production system.

The strongest modeling limitation is also intentional: a strictly pickup-time feature contract contains less signal than one that uses destination or final trip distance. This avoids leakage ambiguity and creates a defensible real-time interface.


