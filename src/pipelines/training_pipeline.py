"""Training pipeline — Prefect flow orchestrating the full training DAG."""

from pathlib import Path

import pandas as pd
from prefect import flow, task

from src.data.ingest import load_raw_data
from src.data.split import get_xy, stratified_split
from src.data.validate import validate_features, validate_raw
from src.features.engineer import engineer_features
from src.registry.promote import promote_if_better
from src.training.train import (
    evaluate_on_test,
    setup_mlflow,
    train_logistic_regression,
    train_xgboost,
)

REFERENCE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "reference.csv"


@task(name="load-data", retries=1)
def load_data_task() -> pd.DataFrame:
    """Load raw data from disk."""
    df = load_raw_data()
    print(f"Loaded {len(df)} transactions")
    return df


@task(name="validate-raw-data")
def validate_raw_task(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw data against Pandera schema."""
    validated = validate_raw(df)
    print("Raw data validation passed")
    return validated


@task(name="engineer-features")
def engineer_features_task(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering."""
    df = engineer_features(df)
    print(f"Engineered features. New columns: {len(df.columns)}")
    return df


@task(name="validate-features")
def validate_features_task(df: pd.DataFrame) -> pd.DataFrame:
    """Validate feature-engineered data."""
    validated = validate_features(df)
    print("Feature validation passed")
    return validated


@task(name="split-data")
def split_data_task(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Stratified train/val/test split."""
    train_df, val_df, test_df = stratified_split(df)

    # Save reference data for drift detection later
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(REFERENCE_PATH, index=False)

    X_train, y_train = get_xy(train_df)
    X_val, y_val = get_xy(val_df)
    X_test, y_test = get_xy(test_df)

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


@task(name="train-logistic-regression")
def train_lr_task(X_train, y_train, X_val, y_val):
    """Train Logistic Regression baseline."""
    model, metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    print(f"LR — val_pr_auc: {metrics['val_pr_auc']:.4f}")
    return model, metrics


@task(name="train-xgboost")
def train_xgb_task(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    model, metrics = train_xgboost(X_train, y_train, X_val, y_val, use_smote=True)
    print(f"XGB — val_pr_auc: {metrics['val_pr_auc']:.4f}")
    return model, metrics


@task(name="evaluate-on-test-set")
def evaluate_test_task(model, run_id: str, X_test, y_test) -> dict:
    """Score the selected model on the held-out test set."""
    metrics = evaluate_on_test(model, X_test, y_test, run_id)
    print(
        f"Test set — pr_auc: {metrics['test_pr_auc']:.4f}, "
        f"precision: {metrics['test_precision']:.4f}, recall: {metrics['test_recall']:.4f}"
    )
    return metrics


@task(name="promote-best")
def promote_best_task(best_name: str, run_id: str) -> str:
    """Attempt promotion of the selected model to Production."""
    promoted = promote_if_better(run_id)
    return f"{'Promoted' if promoted else 'Staged'}: {best_name}"


@flow(name="fraud-detection-training", log_prints=True)
def training_flow():
    """Full pipeline: load -> validate -> features -> split -> train -> test -> promote."""
    setup_mlflow()

    # Data preparation (sequential)
    df = load_data_task()
    df = validate_raw_task(df)
    df = engineer_features_task(df)
    df = validate_features_task(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data_task(df)

    # Train models (can run in parallel with Prefect)
    lr_future = train_lr_task.submit(X_train, y_train, X_val, y_val)
    xgb_future = train_xgb_task.submit(X_train, y_train, X_val, y_val)

    lr_model, lr_metrics = lr_future.result()
    xgb_model, xgb_metrics = xgb_future.result()

    # Select on validation PR-AUC; the test set stays out of the decision
    if xgb_metrics["val_pr_auc"] >= lr_metrics["val_pr_auc"]:
        best_name, best_model, best_metrics = "XGBoost", xgb_model, xgb_metrics
    else:
        best_name, best_model, best_metrics = "LogisticRegression", lr_model, lr_metrics
    print(f"Best model: {best_name} (val_pr_auc: {best_metrics['val_pr_auc']:.4f})")

    evaluate_test_task(best_model, best_metrics["run_id"], X_test, y_test)

    # Promote best model
    result = promote_best_task(best_name, best_metrics["run_id"])
    print(f"\nPipeline complete: {result}")


if __name__ == "__main__":
    training_flow()
