"""Tests for training modules."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

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
            assert abs(ratio - original_ratio) < 0.05, f"{name} ratio {ratio} too far from {original_ratio}"

    def test_no_data_leakage(self, sample_raw_data):
        from src.data.split import stratified_split

        train, val, test = stratified_split(sample_raw_data)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)

        assert len(train_idx & val_idx) == 0, "Train/val overlap"
        assert len(train_idx & test_idx) == 0, "Train/test overlap"
        assert len(val_idx & test_idx) == 0, "Val/test overlap"
