.PHONY: setup train serve test lint format docker-up docker-down drift-check monitor promote mlflow-ui clean

setup:
	uv sync --extra dev

train:
	uv run python -m src.pipelines.training_pipeline

serve:
	uv run uvicorn src.serving.app:app --reload --port 8000

test:
	uv run pytest tests/ -v --cov=src --cov-report=term

lint:
	uv run ruff check src/ tests/ send_predictions.py
	uv run ruff format --check src/ tests/ send_predictions.py

format:
	uv run ruff format src/ tests/ send_predictions.py
	uv run ruff check --fix src/ tests/ send_predictions.py

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down

drift-check:
	uv run python -m src.pipelines.monitoring_pipeline

monitor:
	uv run python -m src.pipelines.monitoring_pipeline --serve

promote:
	uv run python -m src.registry.promote

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache
