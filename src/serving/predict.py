"""Prediction module — load model and run inference."""

import os
import time
from pathlib import Path

import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml

from src.features.engineer import engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_URI = f"file:///{PROJECT_ROOT / 'mlruns'}".replace("\\", "/")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_URI)


def load_serving_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "serving_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _uri_to_path(uri: str) -> Path:
    """Convert a file:/// URI to a Path, handling both Windows and Linux."""
    path_str = uri.removeprefix("file:///")
    # Linux path: "app/mlruns" → need leading slash → "/app/mlruns"
    # Windows path: "C:/Project/..." → already absolute, no slash needed
    if not (len(path_str) >= 2 and path_str[1] == ":"):
        path_str = "/" + path_str
    return Path(path_str)


class FraudPredictor:
    """Loads a model from MLflow registry and serves predictions."""

    def __init__(self):
        self.model = None
        self.model_version = None
        self.config = load_serving_config()
        self.threshold = self.config["decision_threshold"]

    def load_model(self):
        """Load the Production model from MLflow Model Registry."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_name = self.config["model_name"]
        model_stage = self.config["model_stage"]

        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise RuntimeError(f"No model found in registry: {model_name}")

            # Find Production version first, fallback to latest
            prod_versions = [v for v in versions if v.current_stage == model_stage]
            target = (
                prod_versions[0]
                if prod_versions
                else sorted(versions, key=lambda v: int(v.version))[-1]
            )

            source = target.source  # e.g. "models:/m-427e36460aed4f87a3a3090335c24437"

            if source.startswith("models:/"):
                # File-based registry stores the artifact at:
                #   {tracking_root}/{experiment_id}/models/{model_id}/artifacts/
                # The stored host path is wrong inside Docker, so search the
                # mounted filesystem directly using the model ID.
                model_id = source.removeprefix("models:/")
                tracking_root = _uri_to_path(MLFLOW_TRACKING_URI)
                matches = list(tracking_root.glob(f"*/models/{model_id}/artifacts"))
                if not matches:
                    raise RuntimeError(
                        f"Model artifact for '{model_id}' not found under {tracking_root}"
                    )
                model_uri = str(matches[0])
            elif "mlruns/" in source:
                # Direct file path — rewrite host prefix to our tracking root
                relative = source.split("mlruns/", 1)[1]
                model_uri = f"{MLFLOW_TRACKING_URI.rstrip('/')}/{relative}"
            else:
                model_uri = source

            # Load with native flavor (xgboost/sklearn) so predict_proba is available.
            # pyfunc.load_model returns class labels (0/1), not probabilities.
            try:
                self.model = mlflow.xgboost.load_model(model_uri)
            except Exception:
                try:
                    self.model = mlflow.sklearn.load_model(model_uri)
                except Exception:
                    self.model = mlflow.pyfunc.load_model(model_uri)
            self.model_version = f"{target.current_stage} (v{target.version})"
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    def predict(self, features: dict) -> dict:
        """Run inference on a single transaction.

        Args:
            features: Dict of feature name -> value (raw transaction fields).

        Returns:
            Dict with fraud_probability, decision, and latency_ms.
        """
        start = time.perf_counter()

        # Create DataFrame from single transaction
        df = pd.DataFrame([features])

        # Apply feature engineering
        df = engineer_features(df)

        # Drop target if present
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        # Predict
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(df)[0, 1])
        else:
            # pyfunc models return predictions directly
            pred = self.model.predict(df)
            scalar = np.isscalar(pred[0]) or pred[0].ndim == 0
            prob = float(pred[0]) if scalar else float(pred[0][1])

        decision = "block" if prob >= self.threshold else "allow"
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "fraud_probability": round(prob, 6),
            "decision": decision,
            "latency_ms": round(latency_ms, 2),
        }

    def log_prediction(self, features: dict, result: dict):
        """Append prediction to local CSV for drift monitoring."""
        log_path = PROJECT_ROOT / self.config["predictions_log"]
        log_path.parent.mkdir(parents=True, exist_ok=True)

        row = {**features, **result, "timestamp": pd.Timestamp.now().isoformat()}
        df = pd.DataFrame([row])

        if log_path.exists():
            df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(log_path, index=False)
