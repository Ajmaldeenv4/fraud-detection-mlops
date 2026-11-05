"""End-to-end tests that run the real Prefect flows."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from prefect.testing.utilities import prefect_test_harness

from src.data import ingest
from src.pipelines import monitoring_pipeline, training_pipeline
from src.registry.promote import get_production_model_version
from src.serving.predict import FraudPredictor


@pytest.fixture(scope="module", autouse=True)
def prefect_backend():
    with prefect_test_harness():
        yield


@pytest.fixture
def raw_csv(tmp_path):
    """Write a small dataset with the Kaggle schema and a learnable fraud signal."""
    n = 3000
    rng = np.random.default_rng(1)
    labels = (rng.random(n) < 0.05).astype(int)
    data = {"Time": np.sort(rng.uniform(0, 172800, n)), "Amount": rng.exponential(88, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n) + labels * (2.0 if i <= 3 else 0.0)
    data["Class"] = labels

    path = tmp_path / "creditcard.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_training_flow_produces_a_servable_model(raw_csv, tmp_path, mlflow_store, monkeypatch):
    reference = tmp_path / "processed" / "reference.csv"
    monkeypatch.setattr(ingest, "RAW_CSV", raw_csv)
    monkeypatch.setattr(training_pipeline, "REFERENCE_PATH", reference)
    monkeypatch.chdir(tmp_path)

    training_pipeline.training_flow()

    production = get_production_model_version()
    assert production is not None
    assert production["metrics"]["val_pr_auc"] > 0.5
    assert "test_pr_auc" in production["metrics"]
    assert reference.exists()

    predictor = FraudPredictor()
    predictor.load_model()
    row = pd.read_csv(raw_csv).drop(columns=["Class"]).iloc[0].to_dict()
    result = predictor.predict(row)

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["decision"] in {"block", "allow"}


def test_monitoring_flow_retrains_only_on_drift(monkeypatch):
    retrain = MagicMock(return_value={"retrained": True, "reason": "drift"})
    monkeypatch.setattr(monitoring_pipeline, "trigger_retrain_if_needed", retrain)

    monkeypatch.setattr(monitoring_pipeline, "check_drift", lambda: {"is_drifted": False})
    monitoring_pipeline.monitoring_flow()
    retrain.assert_not_called()

    monkeypatch.setattr(
        monitoring_pipeline,
        "check_drift",
        lambda: {"is_drifted": True, "n_drifted_columns": 12},
    )
    monitoring_pipeline.monitoring_flow()
    retrain.assert_called_once()


def test_schedule_uses_configured_interval(monkeypatch):
    serve = MagicMock()
    monkeypatch.setattr(monitoring_pipeline.monitoring_flow, "serve", serve)

    monitoring_pipeline.serve_monitoring()

    assert serve.call_args.kwargs["interval"].total_seconds() == 60 * 60
