"""Tests for the FastAPI prediction API."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY


@pytest.fixture
def mock_predictor():
    """Create a mock predictor that returns a fixed prediction."""
    predictor = MagicMock()
    predictor.model = MagicMock()
    predictor.model_version = "test-v1"
    predictor.predict.return_value = {
        "fraud_probability": 0.85,
        "decision": "block",
        "latency_ms": 5.0,
    }
    predictor.log_prediction = MagicMock()
    return predictor


@pytest.fixture
def client(mock_predictor):
    """Create a test client with mocked model."""
    from src.serving.app import app

    app.state.predictor = mock_predictor
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_version"] == "test-v1"


class TestPredictEndpoint:
    def test_predict_returns_fraud_decision(self, client):
        payload = {
            "Time": 0.0,
            "Amount": 149.62,
            **{f"V{i}": 0.0 for i in range(1, 29)},
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "fraud_probability" in data
        assert "decision" in data
        assert "latency_ms" in data
        assert data["decision"] in ["block", "allow"]

    def test_predict_rejects_negative_amount(self, client):
        payload = {
            "Time": 0.0,
            "Amount": -10.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_requires_amount(self, client):
        payload = {"Time": 0.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_no_model_returns_503(self, client, mock_predictor):
        mock_predictor.model = None
        response = client.post(
            "/predict",
            json={
                "Time": 0.0,
                "Amount": 100.0,
                **{f"V{i}": 0.0 for i in range(1, 29)},
            },
        )
        assert response.status_code == 503

    def test_predict_records_fraud_probability_metric(self, client):
        before = REGISTRY.get_sample_value("fraud_probability_distribution_sum") or 0.0

        response = client.post(
            "/predict",
            json={"Time": 0.0, "Amount": 1.0, **{f"V{i}": 0.0 for i in range(1, 29)}},
        )

        assert response.status_code == 200
        after = REGISTRY.get_sample_value("fraud_probability_distribution_sum")
        assert after - before == pytest.approx(0.85)


class TestReadyEndpoint:
    def test_ready_with_model(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["model_version"] == "test-v1"

    def test_not_ready_without_model(self, client, mock_predictor):
        mock_predictor.model = None
        assert client.get("/ready").status_code == 503
        # Liveness must stay green so the pod isn't restarted while waiting for a model.
        assert client.get("/health").status_code == 200
