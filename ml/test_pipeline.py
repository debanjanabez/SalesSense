"""Smoke test for preprocessing and feature engineering."""

from pathlib import Path

import pandas as pd

from ml.feature_engineering import create_features
from ml.preprocessing import load_and_clean_data

DATA_PATH = Path(__file__).parent.parent / "data" / "sample_sales.csv"


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    print(f"Original data shape: {raw.shape}")

    daily = load_and_clean_data(DATA_PATH)
    print(f"After daily aggregation shape: {daily.shape}")

    featured = create_features(daily)

    feature_cols = [
        c
        for c in featured.columns
        if c not in ("Date", "Sales")
    ]
    print(f"Feature columns: {feature_cols}")
    print(f"Final shape (after feature engineering): {featured.shape}")
    print()
    print("First 5 rows:")
    print(featured.head().to_string(index=False))
    print()
    print("Final 5 rows:")
    print(featured.tail().to_string(index=False))


if __name__ == "__main__":
    main()
