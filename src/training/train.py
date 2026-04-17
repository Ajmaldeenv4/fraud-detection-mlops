"""Model training module — LogisticRegression and XGBoost."""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.training.imbalance import apply_smote, compute_scale_pos_weight
from src.training.evaluate import evaluate_model, log_metrics_to_mlflow, log_confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_URI = f"file:///{PROJECT_ROOT / 'mlruns'}".replace("\\", "/")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_URI)


def load_model_config() -> dict:
    """Load model configuration."""
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    config = load_model_config()
    mlflow.set_experiment(config["experiment_name"])


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[LogisticRegression, dict]:
    """Train a Logistic Regression baseline model.

    Uses class_weight='balanced' to handle imbalance without SMOTE.
    """
    config = load_model_config()["logistic_regression"]

    with mlflow.start_run(run_name="logistic_regression") as run:
        mlflow.log_params(config)
        mlflow.set_tag("model_type", "logistic_regression")

        model = LogisticRegression(**config)
        model.fit(X_train, y_train)

        # Evaluate
        val_metrics = evaluate_model(model, X_val, y_val, "val")
        train_metrics = evaluate_model(model, X_train, y_train, "train")
        all_metrics = {**train_metrics, **val_metrics}
        log_metrics_to_mlflow(all_metrics)

        # Log confusion matrix
        y_pred = model.predict(X_val)
        log_confusion_matrix(y_val, y_pred, "val")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        return model, {**all_metrics, "run_id": run.info.run_id}


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    use_smote: bool = True,
) -> tuple[XGBClassifier, dict]:
    """Train an XGBoost model with imbalance handling.

    Uses either SMOTE (oversampling) or scale_pos_weight (cost-sensitive).
    """
    config = load_model_config()["xgboost"]

    with mlflow.start_run(run_name="xgboost") as run:
        # Handle imbalance
        if use_smote:
            X_train_fit, y_train_fit = apply_smote(X_train, y_train)
            mlflow.set_tag("imbalance_method", "SMOTE")
        else:
            X_train_fit, y_train_fit = X_train, y_train
            config["scale_pos_weight"] = compute_scale_pos_weight(y_train)
            mlflow.set_tag("imbalance_method", "scale_pos_weight")

        mlflow.log_params(config)
        mlflow.set_tag("model_type", "xgboost")

        model = XGBClassifier(**config)
        model.fit(
            X_train_fit,
            y_train_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        val_metrics = evaluate_model(model, X_val, y_val, "val")
        train_metrics = evaluate_model(model, X_train, y_train, "train")
        all_metrics = {**train_metrics, **val_metrics}
        log_metrics_to_mlflow(all_metrics)

        # Log confusion matrix
        y_pred = model.predict(X_val)
        log_confusion_matrix(y_val, y_pred, "val")

        # Log model
        mlflow.xgboost.log_model(model, "model")

        return model, {**all_metrics, "run_id": run.info.run_id}
