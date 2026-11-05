"""Prediction module — load model and run inference."""

import time
from collections import deque
from pathlib import Path

import mlflow.models
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from src.features.engineer import engineer_features, load_feature_config
from src.tracking import artifact_root, configure_tracking, is_remote_store

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_serving_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "serving_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_model_uri(version: ModelVersion) -> str:
    """Return a URI the model artifacts can be loaded from in this environment."""
    if is_remote_store():
        return f"models:/{version.name}/{version.version}"

    if not version.source.startswith("models:/"):
        return version.source

    # The database records the path used at training time, which does not exist if the
    # model was trained on the host and is served from a mounted volume in Docker.
    # Find the artifacts under this environment's store using the model ID instead.
    model_id = version.source.removeprefix("models:/")
    matches = list(artifact_root().glob(f"**/models/{model_id}/artifacts"))
    if not matches:
        raise RuntimeError(f"Model artifact for '{model_id}' not found under {artifact_root()}")
    return str(matches[0])


def _load_native_model(model_uri: str):
    """Load a model in its native flavor so predict_proba is available.

    pyfunc is deliberately not a fallback: it returns class labels (0/1), which would
    be reported to callers as fraud probabilities.
    """
    flavors = mlflow.models.get_model_info(model_uri).flavors
    if "xgboost" in flavors:
        return mlflow.xgboost.load_model(model_uri)
    if "sklearn" in flavors:
        return mlflow.sklearn.load_model(model_uri)
    raise RuntimeError(f"Unsupported model flavors: {sorted(flavors)}")


class FraudPredictor:
    """Loads a model from MLflow registry and serves predictions."""

    def __init__(self):
        self.model = None
        self.model_version = None
        self.config = load_serving_config()
        self.threshold = self.config["decision_threshold"]
        # The rolling window covers the current row plus the previous window-1 amounts.
        window = load_feature_config()["rolling_windows"]["amount_window"]
        self.amount_history = deque(maxlen=window - 1)

    def load_model(self):
        """Load the Production model from MLflow Model Registry."""
        configure_tracking()
        model_name = self.config["model_name"]
        model_stage = self.config["model_stage"]

        client = MlflowClient()

        try:
            versions = client.search_model_versions(f"name='{model_name}'")

            # Never fall back to another stage: those versions failed the promotion check.
            candidates = [v for v in versions if v.current_stage == model_stage]
            if not candidates:
                raise RuntimeError(f"No {model_stage} version in the registry")
            target = max(candidates, key=lambda v: int(v.version))

            self.model = _load_native_model(_resolve_model_uri(target))
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

        # Apply feature engineering, continuing the rolling window from earlier requests
        df = engineer_features(df, amount_history=self.amount_history)
        self.amount_history.append(features["Amount"])

        # Drop target if present
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        # Predict
        prob = float(self.model.predict_proba(df)[0, 1])

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
