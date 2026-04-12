.PHONY: setup train serve test lint docker-up drift-check promote clean

setup:
	pip install -e ".[dev]"

train:
	python -m src.pipelines.training_pipeline

serve:
	uvicorn src.serving.app:app --reload --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

docker-up:
	docker-compose -f docker/docker-compose.yml up --build

docker-down:
	docker-compose -f docker/docker-compose.yml down

drift-check:
	python -m src.pipelines.monitoring_pipeline

promote:
	python -m src.registry.promote

mlflow-ui:
	mlflow ui --backend-store-uri file:///C:/Project/Fraud_Detection/mlruns

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache
