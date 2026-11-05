"""Tests for model promotion against a real MLflow registry."""

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.registry import promote

MODEL_NAME = "fraud-detector"


def log_candidate(pr_auc: float) -> str:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(LogisticRegression().fit(X, y), name="model")
        mlflow.log_metric("val_pr_auc", pr_auc)
    return run.info.run_id


def stages() -> dict[str, str]:
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    return {str(v.version): v.current_stage for v in versions}


@pytest.mark.usefixtures("mlflow_store")
class TestPromotion:
    def test_first_model_goes_to_production(self):
        assert promote.promote_if_better(log_candidate(0.70)) is True
        assert stages() == {"1": "Production"}

    def test_better_model_replaces_production(self):
        promote.promote_if_better(log_candidate(0.70))

        assert promote.promote_if_better(log_candidate(0.80)) is True

        assert stages() == {"1": "Archived", "2": "Production"}

    def test_small_improvement_is_only_staged(self):
        promote.promote_if_better(log_candidate(0.70))

        assert promote.promote_if_better(log_candidate(0.71)) is False

        assert stages() == {"1": "Production", "2": "Staging"}

    def test_worse_model_is_only_staged(self):
        promote.promote_if_better(log_candidate(0.70))

        assert promote.promote_if_better(log_candidate(0.50)) is False

        assert stages() == {"1": "Production", "2": "Staging"}

    def test_explicit_zero_threshold_is_respected(self):
        promote.promote_if_better(log_candidate(0.70))

        assert promote.promote_if_better(log_candidate(0.705), threshold=0.0) is True

    def test_registry_failure_does_not_promote(self, monkeypatch):
        """A failed Production lookup must not read as 'no Production model yet'."""
        promote.promote_if_better(log_candidate(0.70))

        def broken_lookup(*args, **kwargs):
            raise ConnectionError("registry unavailable")

        monkeypatch.setattr(MlflowClient, "search_model_versions", broken_lookup)

        with pytest.raises(ConnectionError):
            promote.promote_if_better(log_candidate(0.10))

        monkeypatch.undo()
        versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
        assert sum(v.current_stage == "Production" for v in versions) == 1

    def test_production_lookup_returns_metrics(self):
        promote.promote_if_better(log_candidate(0.70))

        production = promote.get_production_model_version()

        assert production["version"] == 1
        assert production["metrics"]["val_pr_auc"] == pytest.approx(0.70)

    def test_no_production_model_returns_none(self):
        assert promote.get_production_model_version() is None
