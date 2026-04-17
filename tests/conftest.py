"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data():
    """Create a small sample of raw credit card data."""
    np.random.seed(42)
    n = 200
    data = {
        "Time": np.random.uniform(0, 172800, n),
        "Amount": np.random.exponential(100, n),
        "Class": np.concatenate([np.zeros(195), np.ones(5)]).astype(int),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n)

    np.random.shuffle(data["Class"])
    return pd.DataFrame(data)


@pytest.fixture
def sample_features(sample_raw_data):
    """Create a sample with engineered features."""
    from src.features.engineer import engineer_features

    return engineer_features(sample_raw_data)


@pytest.fixture
def sample_split(sample_features):
    """Create train/val/test split from sample data."""
    from src.data.split import stratified_split, get_xy

    train, val, test = stratified_split(sample_features, test_size=0.2, val_size=0.2)
    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(val)
    X_test, y_test = get_xy(test)
    return X_train, y_train, X_val, y_val, X_test, y_test
