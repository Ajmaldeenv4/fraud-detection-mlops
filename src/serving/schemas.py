"""Pydantic schemas for the prediction API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for /predict endpoint.

    Expects the 30 original features (V1-V28, Time, Amount) plus any
    engineered features. In production, the API would receive raw
    transaction data and compute features internally.
    """

    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = Field(..., ge=0, description="Transaction amount")

    model_config = {"json_schema_extra": {"examples": [{"Time": 0.0, "Amount": 149.62, "V1": -1.36, "V2": -0.07}]}}


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""

    fraud_probability: float = Field(..., ge=0, le=1, description="Predicted probability of fraud")
    decision: str = Field(..., description="'block' or 'allow'")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""

    status: str = "healthy"
    model_version: str | None = None
