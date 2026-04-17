"""Model evaluation module — metrics focused on imbalanced classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
import mlflow


def evaluate_model(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    dataset_name: str = "validation",
) -> dict:
    """Evaluate a model and return metrics.

    Primary metric: PR-AUC (Average Precision) — because accuracy is
    meaningless at 0.17% fraud rate.

    Args:
        model: Trained sklearn-compatible model.
        X: Feature matrix.
        y: True labels.
        dataset_name: Name prefix for logged metrics.

    Returns:
        Dict of metric name -> value.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        f"{dataset_name}_pr_auc": average_precision_score(y, y_prob),
        f"{dataset_name}_roc_auc": roc_auc_score(y, y_prob),
        f"{dataset_name}_f1": f1_score(y, y_pred),
        f"{dataset_name}_precision": np.sum((y_pred == 1) & (y == 1)) / max(np.sum(y_pred == 1), 1),
        f"{dataset_name}_recall": np.sum((y_pred == 1) & (y == 1)) / max(np.sum(y == 1), 1),
    }

    return metrics


def log_metrics_to_mlflow(metrics: dict) -> None:
    """Log all metrics to the active MLflow run."""
    mlflow.log_metrics(metrics)


def log_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "validation",
) -> None:
    """Log confusion matrix as an MLflow artifact."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Legit", "Fraud"])

    artifact_text = f"Confusion Matrix ({dataset_name}):\n{cm}\n\n{report}"
    artifact_path = f"{dataset_name}_confusion_matrix.txt"

    with open(artifact_path, "w") as f:
        f.write(artifact_text)

    mlflow.log_artifact(artifact_path)

    import os
    os.remove(artifact_path)
