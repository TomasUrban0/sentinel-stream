# Sentinel Stream

**Real-time anomaly detection for multivariate time-series data.**

Sentinel Stream is an end-to-end machine learning system that ingests time-series sensor data, engineers features at scale with PySpark, trains an unsupervised autoencoder for anomaly detection, and serves predictions in real time through a FastAPI service. It includes drift monitoring, containerized deployment, and a CI pipeline.

The project is designed to mirror the engineering practices required to take a model from a Jupyter notebook to a production service: a problem most "ML portfolio" projects skip.

---

## Why this project

Most anomaly detection demos stop at `model.fit()`. Real systems have to:

- Ingest data continuously, not from a CSV
- Engineer features deterministically across training and inference
- Serve predictions with sub-100ms latency
- Detect when the data distribution drifts away from training
- Be reproducible, tested, and containerized

Sentinel Stream addresses each of these in a single, coherent codebase.

---

## Architecture

```
+----------------+      +------------------+      +-----------------+
|  Data source   | ---> |  Feature         | ---> |  Model trainer  |
|  (synthetic /  |      |  engineering     |      |  (Autoencoder + |
|   IoT stream)  |      |  (PySpark)       |      |   IsolationFst) |
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
│   ├── data/          # synthetic data generator + ingestion helpers
│   ├── features/      # PySpark feature engineering + sklearn-compatible transformer
│   ├── models/        # Autoencoder (Keras) and Isolation Forest baseline
│   ├── serving/       # FastAPI app, request/response schemas
│   ├── monitoring/    # Drift detection (Kolmogorov-Smirnov)
│   └── utils/         # Logging, config loading
├── scripts/
│   ├── generate_data.py    # build a synthetic dataset with injected anomalies
│   ├── train.py            # train and persist models
│   └── simulate_stream.py  # send live records to the API
├── tests/             # unit + integration tests
├── config/config.yaml
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Quick start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Generate a dataset

```bash
python scripts/generate_data.py --rows 50000 --anomaly-rate 0.02 --out data/sensors.csv
```

### 3. Train the models

```bash
python scripts/train.py --data data/sensors.csv --out artifacts/
```

This trains both an autoencoder and an Isolation Forest baseline, evaluates them on a held-out window, and writes the artifacts to `artifacts/`.

### 4. Serve predictions

```bash
uvicorn sentinel_stream.serving.api:app --host 0.0.0.0 --port 8000
```

Then send a record:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 72.4, "pressure": 101.3, "vibration": 0.12, "humidity": 45.1}'
```

Response:

```json
{
  "anomaly_score": 0.0184,
  "is_anomaly": false,
  "threshold": 0.0421,
  "model": "autoencoder"
}
```

### 5. Simulate a live stream

```bash
python scripts/simulate_stream.py --rate 10 --duration 60
```

Sends 10 records per second for 60 seconds, with sporadic anomalies injected.

---

## Running with Docker

```bash
docker-compose up --build
```

The API is then available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

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

The same transformations are wrapped in a sklearn-compatible class for single-record inference, so training and serving stay in lockstep.

---

## Drift monitoring

The `/metrics` endpoint exposes:

- Total predictions, anomalies flagged
- p50 / p95 / p99 inference latency
- Per-feature Kolmogorov-Smirnov statistic vs. the training distribution

A drift alert is logged when the KS statistic exceeds a configurable threshold for any feature, indicating that the live data has shifted away from what the model saw at training time.

---

## Testing

```bash
pytest -v
ruff check .
```

The CI pipeline (`.github/workflows/ci.yml`) runs linting and the test suite on every push.

---

## Roadmap

- [ ] Replace the simulated stream with a Kafka producer/consumer
- [ ] Persist predictions to a time-series database (TimescaleDB)
- [ ] Replace dense autoencoder with an LSTM autoencoder for sequential signals
- [ ] Add MLflow experiment tracking
- [ ] Deploy the API to Azure Container Apps via GitHub Actions

---

## License

MIT
