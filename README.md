# Fraud Detection MLOps

An end-to-end MLOps pipeline for credit card fraud detection: validated data, tracked training, gated model promotion, a real-time API, drift monitoring with automatic retraining, and dashboards.

| Stage | Tools | Code |
|---|---|---|
| Orchestration | Prefect | `src/pipelines/` |
| Validation and features | Pandera, pandas | `src/data/`, `src/features/` |
| Training | XGBoost with SMOTE, scikit-learn baseline | `src/training/` |
| Tracking and registry | MLflow | `src/tracking.py`, `src/registry/` |
| Serving | FastAPI | `src/serving/` |
| Monitoring | Evidently, Prometheus, Grafana | `src/monitoring/`, `dashboards/` |
| Delivery | Docker, Kubernetes, GitHub Actions | `docker/`, `k8s/`, `.github/` |

## Quick start

These steps rebuild everything from a fresh clone. They need [uv](https://docs.astral.sh/uv/), which also installs Python 3.11 if you don't have it.

**1. Install the locked dependencies.**

```bash
uv sync --extra dev
```

**2. Get the dataset.** It's too large for git, so `data/raw/creditcard.csv` has to be fetched. Download the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), or use TensorFlow's public copy:

```bash
mkdir -p data/raw
curl -o data/raw/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

The file should be about 151 MB, with 284,807 transactions of which 492 are fraud.

**3. Train.** This validates the data, trains both models, scores the winner on the test set, and promotes it if it beats the current Production model.

```bash
uv run python -m src.pipelines.training_pipeline
```

**4. Serve.**

```bash
uv run uvicorn src.serving.app:app --port 8000
```

Interactive API docs are at http://localhost:8000/docs. To send sample traffic from a second terminal:

```bash
uv run python send_predictions.py
```

**5. Check for drift.** This needs at least 1,000 logged predictions. `send_predictions.py` sends 50 per run unless you set `N_REQUESTS`, and its random values don't look like real transactions, so expect them to be reported as drift.

```bash
uv run python -m src.pipelines.monitoring_pipeline
```

The commands above also have `make` shortcuts if you have make installed; see the [Makefile](Makefile).

## API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score one transaction. Returns `fraud_probability`, a `block` or `allow` `decision`, and `latency_ms` |
| `GET /health` | Liveness: always 200 while the process is up |
| `GET /ready` | Readiness: 503 until a model is loaded |
| `POST /reload` | Load the current Production model without restarting |
| `GET /metrics` | Prometheus metrics |

The API doesn't notice a newly promoted model by itself, so call `/reload` after training.

## How the pipeline behaves

**Promotion.** The best model on validation PR-AUC replaces Production only if it improves by at least 0.02 (`configs/model_config.yaml`). Otherwise it's moved to Staging, and the API never serves a model that isn't in Production.

**Drift.** Each column is compared with the training data using Jensen-Shannon distance, which measures how big a difference is. Data counts as drifted when more than 15% of columns exceed 0.15 (`configs/monitoring_config.yaml`). `Time` is left out because it's a running clock, so live values always differ from training.

Those settings come from testing stretches of the real transactions. Spending patterns change through the day, so tests of statistical significance flag nearly every window, while these settings flagged no normal stretch of 1,000 rows or more and caught a half standard deviation shift in a third of the columns. The dataset covers only two days, so treat them as a starting point. Smaller shifts look like normal daily variation and won't be detected.

**Retraining.** When drift is detected and `auto_retrain` is on, the training flow runs, at most once per `cooldown_hours`. To check on a schedule rather than once:

```bash
uv run python -m src.pipelines.monitoring_pipeline --serve
```

Retraining reads the same `data/raw/creditcard.csv`, because live predictions have no fraud labels. Add newly labelled transactions to that file for a retrain to learn anything new.

**Rolling features.** The dataset has no cardholder ID, so `amount_rolling_*` covers the last five transactions overall. The API keeps those amounts in memory, which means each process has its own window and it restarts empty.

## Where things are stored

| Path | Contents |
|---|---|
| `mlruns/` | MLflow database (`mlflow.db`), runs and models |
| `data/raw/` | Source dataset |
| `data/processed/reference.csv` | Training data that drift checks compare against |
| `data/predictions/recent.csv` | Log of every served prediction |
| `reports/drift_report.html` | Latest Evidently report |

All of these are gitignored, so **they exist only on the machine that created them.** The `.dvc/` folder is configured, but no dataset was ever tracked with it and `dvc` isn't a project dependency, so it can't restore anything. Keep a copy of `mlruns/` somewhere safe if you need to preserve trained models.

To browse runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Docker

Train locally first, then start the API, MLflow, Prometheus and Grafana:

```bash
docker compose -f docker/docker-compose.yml up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| MLflow | http://localhost:5001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (dashboard is viewable without signing in) |

The API reads models from the mounted `mlruns/` folder.

## Kubernetes

Pushes to `main` build and publish `ghcr.io/ajmaldeenv4/fraud-detection-api`. To deploy it with an MLflow server:

```bash
kubectl apply -f k8s/
```

The API pods stay unready until the new MLflow server has a Production model. To train into it, forward its port and point training at it:

```bash
kubectl port-forward service/mlflow-service 5000:5000
```

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m src.pipelines.training_pipeline
```

Then call `/reload` on each API pod, or restart the deployment.

## Development

```bash
uv run pytest tests/ -v --cov=src
```

```bash
uv run ruff check src/ tests/ send_predictions.py
```

Dependency versions are locked in `uv.lock`, and CI and the Docker image both install from it. After changing `pyproject.toml`, run `uv lock` and commit the result.
