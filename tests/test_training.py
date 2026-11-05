"""Tests for training modules."""

import re
import warnings
from concurrent.futures import ThreadPoolExecutor

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.exceptions import ConvergenceWarning

from src.training.imbalance import apply_smote, compute_scale_pos_weight


class TestImbalance:
    def test_smote_balances_classes(self):
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
        y = pd.Series([0] * 90 + [1] * 10, name="Class")

        X_res, y_res = apply_smote(X, y)
        # After SMOTE, classes should be balanced
        assert abs(y_res.value_counts()[0] - y_res.value_counts()[1]) == 0

    def test_smote_preserves_dataframe_type(self):
        np.random.seed(42)
        n = 50
        X = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
        y = pd.Series([0] * 40 + [1] * 10)

        X_res, y_res = apply_smote(X, y)
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)
        assert list(X_res.columns) == ["a", "b"]

    def test_scale_pos_weight(self):
        y = pd.Series([0] * 990 + [1] * 10)
        weight = compute_scale_pos_weight(y)
        assert weight == pytest.approx(99.0)


class TestSplit:
    def test_stratified_split_preserves_ratio(self, sample_raw_data):
        from src.data.split import stratified_split

        train, val, test = stratified_split(sample_raw_data)
        original_ratio = sample_raw_data["Class"].mean()

        # Each split should have roughly the same fraud rate
        for split, name in [(train, "train"), (val, "val"), (test, "test")]:
            ratio = split["Class"].mean()
            # Allow some tolerance due to small sample size
            assert abs(ratio - original_ratio) < 0.05, (
                f"{name} ratio {ratio} too far from {original_ratio}"
            )

    def test_no_data_leakage(self, sample_raw_data):
        from src.data.split import stratified_split

        train, val, test = stratified_split(sample_raw_data)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)

        assert len(train_idx & val_idx) == 0, "Train/val overlap"
        assert len(train_idx & test_idx) == 0, "Train/test overlap"
        assert len(val_idx & test_idx) == 0, "Val/test overlap"


class TestTraining:
    def test_logistic_regression_converges(self, sample_split, mlflow_store):
        from src.training.train import train_logistic_regression

        X_train, y_train, X_val, y_val, _, _ = sample_split
        X_train = X_train.assign(Time=X_train["Time"] * 1000, Amount=X_train["Amount"] * 1000)

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model, metrics = train_logistic_regression(X_train, y_train, X_val, y_val)

        assert 0.0 <= model.predict_proba(X_val)[0, 1] <= 1.0
        assert "val_pr_auc" in metrics

    def test_xgboost_config_has_no_removed_parameters(self):
        from src.training.train import load_model_config

        assert "use_label_encoder" not in load_model_config()["xgboost"]

    def test_parallel_runs_keep_their_own_confusion_matrix(self, mlflow_store):
        """Runs train in parallel threads, so artifacts must not share a temp file."""
        from src.training.evaluate import log_confusion_matrix

        def log(n_fraud: int) -> tuple[str, int]:
            y_true = np.array([0] * 20 + [1] * 20)
            y_pred = np.array([0] * (40 - n_fraud) + [1] * n_fraud)
            with mlflow.start_run() as run:
                log_confusion_matrix(y_true, y_pred, "val")
            return run.info.run_id, n_fraud

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(log, [5, 10, 15, 20] * 4))

        client = MlflowClient()
        for run_id, n_fraud in results:
            local = client.download_artifacts(run_id, "val_confusion_matrix.txt")
            with open(local) as f:
                matrix = f.read().split("\n\n")[0]
            assert [int(n) for n in re.findall(r"\d+", matrix)][-1] == n_fraud

    def test_test_metrics_are_logged_to_the_candidate_run(self, sample_split, mlflow_store):
        from src.training.train import evaluate_on_test, train_logistic_regression

        X_train, y_train, X_val, y_val, X_test, y_test = sample_split
        model, metrics = train_logistic_regression(X_train, y_train, X_val, y_val)

        test_metrics = evaluate_on_test(model, X_test, y_test, metrics["run_id"])

        logged = MlflowClient().get_run(metrics["run_id"]).data.metrics
        assert logged["test_pr_auc"] == pytest.approx(test_metrics["test_pr_auc"])
