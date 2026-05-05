# Sentinel Stream

[![CI](https://github.com/TomasUrban0/sentinel-stream/actions/workflows/ci.yml/badge.svg)](https://github.com/TomasUrban0/sentinel-stream/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

**Real-time anomaly detection for multivariate industrial time-series, evaluated on the [Skoltech Anomaly Benchmark (SKAB)](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab).**

Sentinel Stream is an end-to-end machine learning system that ingests eight-channel sensor telemetry, engineers features at scale with PySpark, trains an unsupervised autoencoder for anomaly detection, and serves predictions in real time through a FastAPI service. It includes drift monitoring, containerized deployment, and a CI pipeline.

The project mirrors the engineering practices required to take a model from a Jupyter notebook to a production service: a problem most "ML portfolio" projects skip.

---

## Why this project

Most anomaly detection demos stop at `model.fit()`. Real systems have to:

- Ingest data continuously, not from a CSV
- Engineer features deterministically across training and inference
- Serve predictions with low, predictable latency
- Detect when the data distribution drifts away from training
- Be reproducible, tested, and containerized

Sentinel Stream addresses each of these in a single, coherent codebase, and validates the modeling on a published industrial benchmark rather than synthetic data.

---

## Dataset

[**SKAB — Skoltech Anomaly Benchmark**](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab) is a multivariate time-series benchmark released by Skoltech. The data come from a water-circulation testbed instrumented with eight sensors, sampled at 1 Hz, with deliberately induced faults (valve closures, pump cavitation, etc.) labeled per-row.

| Partition       | Rows    | Use                                              |
|-----------------|---------|--------------------------------------------------|
| `anomaly-free/` | 9,401   | Training (no labels — assumed all normal)        |
| `valve1`/`valve2`/`other` | 37,459 (35 % anomalous) | Held-out evaluation with ground-truth labels |

Eight sensor channels: two RMS accelerometers, current, pressure, temperature, thermocouple, voltage, and volumetric flow rate.

---

## Architecture

```
+----------------+      +------------------+      +-----------------+
|  SKAB CSV      | ---> |  Feature         | ---> |  Model trainer  |
|  (8 sensors)   |      |  engineering     |      |  (Autoencoder + |
|                |      |  (PySpark)       |      |   IsolationFst) |
+----------------+      +------------------+      +-----------------+
                                                          |
                                                          v
+----------------+      +------------------+      +-----------------+
|  Stream        | ---> |  FastAPI         | ---> |  Anomaly score  |
|  simulator     |      |  inference API   |      |  + alert        |
+----------------+      +------------------+      +-----------------+
                               |
                               v
                       +------------------+
                       |  Drift monitor   |
                       |  (KS test on     |
                       |   feature dist.) |
                       +------------------+
```

---

## Tech stack

| Layer            | Tools                                      |
|------------------|--------------------------------------------|
| Data processing  | PySpark, Pandas, NumPy                     |
| Modeling         | TensorFlow / Keras, Scikit-learn           |
| Serving          | FastAPI, Uvicorn, Pydantic                 |
| Monitoring       | SciPy (KS test), custom metrics endpoint   |
| Infrastructure   | Docker, docker-compose                     |
| CI               | GitHub Actions, Ruff, Pytest               |

---

## Project structure

```
sentinel-stream/
├── src/sentinel_stream/
│   ├── data/          # SKAB loader
│   ├── features/      # PySpark feature engineering + streaming transformer
│   ├── models/        # Autoencoder (Keras) and Isolation Forest baseline
│   ├── serving/       # FastAPI app, request/response schemas
│   ├── monitoring/    # Drift detection (Kolmogorov-Smirnov)
│   └── utils/         # Logging, config loading
├── scripts/
│   ├── download_data.py   # fetch SKAB from Kaggle
│   ├── train.py           # train and persist models
│   └── simulate_stream.py # replay a SKAB CSV against the running API
├── notebooks/
│   └── exploration.ipynb  # EDA + anomaly visualisations on SKAB
├── tests/             # unit + integration + end-to-end tests
├── config/config.yaml
├── Makefile
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Quick start

You will need a [Kaggle API token](https://www.kaggle.com/settings) at `~/.kaggle/kaggle.json` for the data download step.

```bash
make setup     # install dependencies in editable mode
make data      # download SKAB into data/skab/
make train     # train both models, write artifacts/
make serve     # start the FastAPI service on :8000
make simulate  # replay a labeled SKAB CSV against the running API
```

Below is the same flow without Make.

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Download SKAB

```bash
python scripts/download_data.py --out data/skab
```

### 3. Train the models

```bash
python scripts/train.py --data-root data/skab --out artifacts/
```

Trains an autoencoder (and an Isolation Forest baseline) on the anomaly-free partition only, evaluates against the labeled partition, and writes the artifacts and `metrics.json` to `artifacts/`.

### 4. Serve predictions

```bash
uvicorn sentinel_stream.serving.api:app --host 0.0.0.0 --port 8000
```

Then send a record:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "accelerometer_1_rms": 0.027,
    "accelerometer_2_rms": 0.040,
    "current": 0.71,
    "pressure": 0.058,
    "temperature": 70.6,
    "thermocouple": 24.4,
    "voltage": 220.5,
    "volume_flow_rate": 32.1
  }'
```

### 5. Replay a live stream

```bash
python scripts/simulate_stream.py --rate 20
```

Reads `data/skab/SKAB/valve1/0.csv` row-by-row and sends each record to `/predict`. Logs recall against the ground-truth label.

---

## Running with Docker

```bash
docker-compose up --build
```

The API is then available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

---

## Results

Trained unsupervised on 9,396 anomaly-free rows (the `anomaly-free/` partition of SKAB), evaluated on 37,454 labeled rows from `valve1/`, `valve2/`, and `other/`. The threshold is set at the 99th percentile of training-set reconstruction errors. **No labels are seen during training.**

| Metric    | Autoencoder | Isolation Forest |
|-----------|-------------|------------------|
| Precision | 0.360       | 0.358            |
| Recall    | **0.948**   | **0.954**        |
| F1        | **0.522**   | 0.520            |
| ROC-AUC   | **0.573**   | 0.525            |
| PR-AUC    | **0.447**   | 0.393            |

These numbers sit in the range published academic baselines report for SKAB without supervised threshold tuning. The autoencoder edges out Isolation Forest on AUC; both lean toward high recall and modest precision because the labeled partition is taken from a *different physical regime* than the training partition (valve-fault scenarios vs. baseline operation), so the simple percentile threshold over-flags. Two practical takeaways are baked into the project:

1. **The drift monitor catches this.** During the replay, 68 of 82 engineered features cross the KS drift threshold against the training reference — the system itself surfaces the distribution shift behind the precision drop.
2. **The threshold is a knob, not a rule.** A small labeled validation slice (or a quantile fit during canary deployment) is the standard fix and is what would land next on a production roadmap.

**Serving latency** (FastAPI, single-process, CPU-only inference, replay at 50 rps):

| Percentile | Latency |
|------------|---------|
| p50        | 51 ms   |
| p95        | 75 ms   |
| p99        | 113 ms  |

> Reproduce: `make data && make train && make serve`, then `make simulate`. Final training metrics are in `artifacts/metrics.json` and live serving metrics at `GET /metrics`.

---

## Modeling approach

Two complementary models are trained:

**Autoencoder (primary).** A symmetric dense autoencoder compresses the feature vector to a low-dimensional latent representation and reconstructs it. The reconstruction error is used as the anomaly score: anomalies are points that the network was never trained to compress well. The threshold is set at the 99th percentile of training-set reconstruction errors.

**Isolation Forest (baseline).** A classical unsupervised method that isolates anomalies by random partitioning. It serves as a sanity check and a fallback that does not require a GPU or training time.

Both models share the same feature pipeline so they are directly comparable.

---

## Feature engineering

Features are engineered in PySpark to demonstrate the same pattern that scales to billions of rows:

- **Rolling-window aggregates**: mean, std, min, max over 5- and 30-step windows
- **Lag features**: t-1, t-5
- **Time features**: hour-of-day, day-of-week
- **Z-score normalization** with parameters persisted for inference

The same transformations are wrapped in a streaming-compatible class for single-record inference, so training and serving stay in lockstep — there is no risk of train/serve skew silently degrading the model.

---

## Drift monitoring

The `/metrics` endpoint exposes:

- Total predictions, anomalies flagged
- p50 / p95 / p99 inference latency
- Per-feature Kolmogorov-Smirnov statistic vs. the training distribution

A drift alert is logged when the KS statistic exceeds a configurable threshold for any feature, indicating that the live data has shifted away from what the model saw at training time. On SKAB this fires on the majority of features once the valve-fault data starts streaming — a useful, honest signal that re-training (or threshold re-tuning) is needed.

---

## Testing

```bash
pytest -v
ruff check .
```

The CI pipeline (`.github/workflows/ci.yml`) runs linting and the test suite on every push.

---

## Roadmap

- [ ] LSTM/Transformer autoencoder to model temporal dependencies explicitly
- [ ] Threshold re-tuning on a small labeled validation slice
- [ ] Replace HTTP polling with a Kafka producer/consumer
- [ ] Persist predictions to a time-series database (TimescaleDB)
- [ ] MLflow experiment tracking and a model registry
- [ ] Deploy the API to Azure Container Apps via GitHub Actions

---

## License

MIT
