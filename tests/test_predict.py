"""Tests for model loading and inference with a real MLflow registry."""

import shutil

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.features.engineer import engineer_features
from src.registry.promote import promote_if_better
from src.serving import predict as predict_module
from src.serving.predict import FraudPredictor


class RecordingModel:
    """Stands in for a trained model and keeps the features it was given."""

    def __init__(self):
        self.seen = []

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self.seen.append(df)
        return np.array([[0.75, 0.25]])


def promote_small_model() -> str:
    """Log and promote a model whose probabilities are strictly between 0 and 1."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    with mlflow.start_run() as run:
        model = LogisticRegression().fit(X, [0, 0, 1, 1])
        mlflow.sklearn.log_model(model, name="model")
        mlflow.log_metric("val_pr_auc", 0.9)
    promote_if_better(run.info.run_id)
    return run.info.run_id


class TestInference:
    def test_streamed_features_match_training_features(self, sample_raw_data):
        """One-at-a-time serving must build the features the model was trained on."""
        raw = sample_raw_data.drop(columns=["Class"])
        batch = engineer_features(raw).head(20)

        predictor = FraudPredictor()
        predictor.model = RecordingModel()
        for row in raw.head(20).to_dict("records"):
            predictor.predict(row)

        streamed = pd.concat(predictor.model.seen, ignore_index=True)
        pd.testing.assert_frame_equal(streamed, batch.reset_index(drop=True), check_exact=False)

    def test_decision_uses_configured_threshold(self):
        predictor = FraudPredictor()
        predictor.model = RecordingModel()
        predictor.threshold = 0.2

        result = predictor.predict({"Time": 0.0, "Amount": 10.0})

        assert result["decision"] == "block"
        assert result["fraud_probability"] == pytest.approx(0.25)


class TestLoadModel:
    def test_loads_production_model_with_probabilities(self, mlflow_store):
        promote_small_model()

        predictor = FraudPredictor()
        predictor.load_model()

        assert predictor.model_version == "Production (v1)"
        probability = predictor.model.predict_proba(pd.DataFrame({"x": [1.5]}))[0, 1]
        assert 0.0 < probability < 1.0

    def test_loads_after_store_moves_to_another_location(self, mlflow_store, tmp_path, monkeypatch):
        """Train on the host, serve in a container: recorded paths no longer exist."""
        promote_small_model()

        moved = tmp_path / "mounted" / "mlruns"
        shutil.copytree(mlflow_store, moved)
        shutil.rmtree(mlflow_store / "models")  # the open database can't be moved on Windows
        monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(moved))

        predictor = FraudPredictor()
        predictor.load_model()

        assert predictor.model is not None

    def test_refuses_model_that_failed_promotion(self, mlflow_store):
        """A Staging model lost its promotion check and must never serve traffic."""
        run_id = promote_small_model()
        client = MlflowClient()
        client.transition_model_version_stage("fraud-detector", "1", "Staging")
        assert run_id

        predictor = FraudPredictor()
        with pytest.raises(RuntimeError, match="No Production version"):
            predictor.load_model()

        assert predictor.model is None

    def test_reports_empty_registry(self, mlflow_store):
        with pytest.raises(RuntimeError, match="No Production version"):
            FraudPredictor().load_model()

    def test_remote_store_loads_through_the_registry(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")

        class Version:
            name = "fraud-detector"
            version = "7"
            source = "models:/m-abc"

        assert predict_module._resolve_model_uri(Version()) == "models:/fraud-detector/7"
