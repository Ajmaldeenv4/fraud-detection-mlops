"""Data ingestion module — loads the credit card fraud dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV = RAW_DATA_DIR / "creditcard.csv"


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw credit card fraud dataset from CSV.

    Args:
        path: Path to the CSV file. Defaults to data/raw/creditcard.csv.

    Returns:
        DataFrame with raw transaction data.
    """
    csv_path = path or RAW_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place creditcard.csv in {RAW_DATA_DIR}"
        )
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} transactions")
    print(f"Fraud rate: {df['Class'].mean():.4%}")
    print(f"Columns: {list(df.columns)}")
