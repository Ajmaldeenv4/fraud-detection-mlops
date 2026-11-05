"""Model training module — LogisticRegression and XGBoost."""

from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.tracking import artifact_root, configure_tracking, is_remote_store
from src.training.evaluate import evaluate_model, log_confusion_matrix, log_metrics_to_mlflow
from src.training.imbalance import apply_smote, compute_scale_pos_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_model_config() -> dict:
    """Load model configuration."""
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_mlflow():
    """Configure MLflow tracking."""
    configure_tracking()
    experiment_name = load_model_config()["experiment_name"]
    # Keep local artifacts in the store directory rather than wherever Python was started.
    if not is_remote_store() and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_root().as_uri())
    mlflow.set_experiment(experiment_name)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, dict]:
    """Train a Logistic Regression baseline model.

    Uses class_weight='balanced' to handle imbalance without SMOTE. Features are
    scaled first because Time and Amount are orders of magnitude larger than V1-V28,
    which stops lbfgs from converging.
    """
    config = load_model_config()["logistic_regression"]

    with mlflow.start_run(run_name="logistic_regression") as run:
        mlflow.log_params(config)
        mlflow.set_tag("model_type", "logistic_regression")

        model = make_pipeline(StandardScaler(), LogisticRegression(**config))
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
        mlflow.sklearn.log_model(model, name="model")

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
        mlflow.xgboost.log_model(model, name="model")

        return model, {**all_metrics, "run_id": run.info.run_id}


def evaluate_on_test(model, X_test: pd.DataFrame, y_test: pd.Series, run_id: str) -> dict:
    """Score a selected model on the held-out test set and log it to its own run.

    Model selection and promotion use validation metrics, so this is the unbiased
    estimate of how the chosen model will perform.
    """
    with mlflow.start_run(run_id=run_id):
        metrics = evaluate_model(model, X_test, y_test, "test")
        log_metrics_to_mlflow(metrics)
        log_confusion_matrix(y_test, model.predict(X_test), "test")
    return metrics
