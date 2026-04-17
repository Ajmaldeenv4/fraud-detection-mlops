"""Class imbalance handling — SMOTE and class weight utilities."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    random_state: int = 42,
) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
    """Apply SMOTE oversampling to the training set ONLY.

    IMPORTANT: Never apply to validation or test sets — this would leak
    synthetic data and invalidate evaluation.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed.

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Preserve DataFrame/Series types if input was pandas
    if isinstance(X_train, pd.DataFrame):
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
    if isinstance(y_train, pd.Series):
        y_res = pd.Series(y_res, name=y_train.name)

    return X_res, y_res


def compute_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost (ratio of negatives to positives)."""
    n_positive = np.sum(y == 1)
    n_negative = np.sum(y == 0)
    return n_negative / n_positive
