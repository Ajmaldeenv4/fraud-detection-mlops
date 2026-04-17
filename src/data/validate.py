"""Data validation module — Pandera schemas for raw and processed data."""

import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd


# Schema for the raw Kaggle credit card fraud dataset
raw_schema = DataFrameSchema(
    columns={
        "Time": Column(float, Check.ge(0), nullable=False),
        "Amount": Column(float, Check.ge(0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
        **{
            f"V{i}": Column(float, nullable=False)
            for i in range(1, 29)
        },
    },
    strict=False,  # Allow extra columns
    coerce=True,
)

# Schema for feature-engineered data
feature_schema = DataFrameSchema(
    columns={
        "Time": Column(float, Check.ge(0), nullable=False),
        "Amount": Column(float, Check.ge(0), nullable=False),
        "Class": Column(int, Check.isin([0, 1]), nullable=False),
        "hour_of_day": Column(float, [Check.ge(0), Check.lt(24)], nullable=False),
        "amount_log": Column(float, Check.ge(0), nullable=False),
        "amount_rolling_mean": Column(float, nullable=True),  # NaN for first rows
        "amount_rolling_std": Column(float, nullable=True),
        "is_night": Column(int, Check.isin([0, 1]), nullable=False),
        **{
            f"V{i}": Column(float, nullable=False)
            for i in range(1, 29)
        },
    },
    strict=False,
    coerce=True,
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw dataset against the expected schema."""
    return raw_schema.validate(df)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate feature-engineered dataset."""
    return feature_schema.validate(df)


if __name__ == "__main__":
    from src.data.ingest import load_raw_data

    df = load_raw_data()
    validated = validate_raw(df)
    print(f"Validation passed! {len(validated)} rows OK.")
