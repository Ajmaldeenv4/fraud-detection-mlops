"""Feature engineering module — create derived features from raw transaction data."""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_feature_config() -> dict:
    """Load feature engineering configuration."""
    config_path = PROJECT_ROOT / "configs" / "feature_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to the raw dataset.

    Creates:
        - hour_of_day: Hour extracted from Time (seconds since first tx)
        - amount_log: log1p(Amount) to reduce skew
        - amount_rolling_mean: Rolling mean of Amount over a window
        - amount_rolling_std: Rolling std of Amount over a window
        - is_night: Binary flag for transactions between midnight and 6am

    Note: The Kaggle dataset has no cardholder ID, so rolling features
    use positional windows rather than per-user windows. In production,
    you would group by cardholder.
    """
    config = load_feature_config()
    df = df.copy()

    # Time-based features
    seconds_per_day = config["time_features"]["seconds_per_day"]
    df["hour_of_day"] = (df["Time"] % seconds_per_day) / 3600

    # Night flag (0am - 6am)
    night_start = config["time_features"]["night_start"]
    night_end = config["time_features"]["night_end"]
    df["is_night"] = ((df["hour_of_day"] >= night_start) & (df["hour_of_day"] < night_end)).astype(
        int
    )

    # Amount features
    df["amount_log"] = np.log1p(df["Amount"])

    # Rolling features (positional window — no cardholder ID available)
    window = config["rolling_windows"]["amount_window"]
    df["amount_rolling_mean"] = df["Amount"].rolling(window=window, min_periods=1).mean()
    df["amount_rolling_std"] = df["Amount"].rolling(window=window, min_periods=1).std().fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature column names (excludes target)."""
    exclude = {"Class"}
    return [col for col in df.columns if col not in exclude]


if __name__ == "__main__":
    from src.data.ingest import load_raw_data

    df = load_raw_data()
    df = engineer_features(df)
    print(f"Features: {list(df.columns)}")
    print(f"\nNew feature stats:")
    for col in ["hour_of_day", "amount_log", "amount_rolling_mean", "amount_rolling_std", "is_night"]:
        print(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")
