"""Tests for the FastAPI prediction API."""

import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


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
        response = client.post("/predict", json={
            "Time": 0.0,
            "Amount": 100.0,
            **{f"V{i}": 0.0 for i in range(1, 29)},
        })
        assert response.status_code == 503
