"""Data splitting module — stratified train/val/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets preserving class distribution.

    Args:
        df: Full dataset with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        val_size: Fraction for validation set (from remaining after test).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # First split: separate test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    # Second split: separate validation from training
    # Adjust val_size relative to train_val size
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[target_col],
        random_state=random_state,
    )

    return train, val, test


def get_xy(df: pd.DataFrame, target_col: str = "Class"):
    """Split DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


if __name__ == "__main__":
    from src.data.ingest import load_raw_data

    df = load_raw_data()
    train, val, test = stratified_split(df)
    print(f"Train: {len(train)} ({train['Class'].mean():.4%} fraud)")
    print(f"Val:   {len(val)} ({val['Class'].mean():.4%} fraud)")
    print(f"Test:  {len(test)} ({test['Class'].mean():.4%} fraud)")
