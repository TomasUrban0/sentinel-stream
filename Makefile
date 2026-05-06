.PHONY: help setup data train serve simulate test lint format docker clean

help:
	@echo "Available targets:"
	@echo "  setup     Install dependencies in editable mode"
	@echo "  data      Download the AI4I 2020 dataset from Kaggle"
	@echo "  train     Train XGBoost + Keras classifiers on AI4I"
	@echo "  plots     Render evaluation plots into docs/img/"
	@echo "  serve     Run the FastAPI inference service on :8000"
	@echo "  simulate  Replay AI4I rows against the running API"
	@echo "  test      Run the test suite with coverage"
	@echo "  lint      Run Ruff lint checks"
	@echo "  format    Auto-fix formatting and lint issues with Ruff"
	@echo "  docker    Build and start the API container"
	@echo "  clean     Remove caches, artifacts, and generated data"

setup:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

data:
	python scripts/download_data.py --out data/ai4i

train:
	python scripts/train.py --data-root data/ai4i --out artifacts/

plots:
	python scripts/generate_plots.py --data-root data/ai4i --artifacts artifacts/ --out docs/img/

serve:
	uvicorn sentinel_stream.serving.api:app --host 0.0.0.0 --port 8000

simulate:
	python scripts/simulate_stream.py --rate 50 --limit 2000

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
