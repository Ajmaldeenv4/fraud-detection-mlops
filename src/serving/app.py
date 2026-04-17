"""FastAPI application for fraud detection inference."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.serving.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.serving.predict import FraudPredictor
from src.monitoring.metrics import track_prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    app.state.predictor = FraudPredictor()
    try:
        app.state.predictor.load_model()
        print(f"Model loaded: {app.state.predictor.model_version}")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Start the server anyway — use /reload to load model later.")
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with ML",
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus metrics instrumentation
Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    predictor = app.state.predictor
    return HealthResponse(
        status="healthy" if predictor.model is not None else "no_model",
        model_version=predictor.model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict whether a transaction is fraudulent.

    Returns fraud probability, block/allow decision, and inference latency.
    """
    predictor = app.state.predictor
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /reload first.")

    features = request.model_dump()
    result = predictor.predict(features)

    # Track metrics for Prometheus
    track_prediction(result["decision"], result["latency_ms"])

    # Log for drift monitoring
    predictor.log_prediction(features, result)

    return PredictionResponse(**result)


@app.post("/reload")
async def reload_model():
    """Hot-reload the Production model without restarting the server."""
    predictor = app.state.predictor
    try:
        predictor.load_model()
        return {"status": "reloaded", "model_version": predictor.model_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")
