.PHONY: help setup data train serve simulate test lint format docker clean

help:
	@echo "Available targets:"
	@echo "  setup     Install dependencies in editable mode"
	@echo "  data      Download the SKAB dataset from Kaggle into data/skab"
	@echo "  train     Train autoencoder + isolation forest on SKAB"
	@echo "  serve     Run the FastAPI inference service on :8000"
	@echo "  simulate  Replay a SKAB CSV against the running API"
	@echo "  test      Run the test suite with coverage"
	@echo "  lint      Run Ruff lint checks"
	@echo "  format    Auto-fix formatting and lint issues with Ruff"
	@echo "  docker    Build and start the API container"
	@echo "  clean     Remove caches, artifacts, and generated data"

setup:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

data:
	python scripts/download_data.py --out data/skab

train:
	python scripts/train.py --data-root data/skab --out artifacts/

serve:
	uvicorn sentinel_stream.serving.api:app --host 0.0.0.0 --port 8000

simulate:
	python scripts/simulate_stream.py --rate 20

test:
	pytest -v --cov=sentinel_stream --cov-report=term-missing

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

docker:
	docker-compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov .mypy_cache
	rm -rf artifacts/ data/ logs/ spark-warehouse/ metastore_db/ derby.log
	find . -type d -name __pycache__ -exec rm -rf {} +
