"""Load and clean raw sales data for daily forecasting."""

import warnings
from pathlib import Path

import pandas as pd


def load_and_clean_data(filepath: str | Path) -> pd.DataFrame:
    """Load sales CSV, clean invalid rows, and aggregate to daily totals.

    Returns a dataframe with columns: Date, Sales
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)

    required = {"Date", "Sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    initial_rows = len(df)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates:
        warnings.warn(f"Dropping {invalid_dates} row(s) with invalid Date values.")
        df = df.dropna(subset=["Date"])

    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    invalid_sales = df["Sales"].isna().sum()
    if invalid_sales:
        warnings.warn(f"Dropping {invalid_sales} row(s) with invalid Sales values.")
        df = df.dropna(subset=["Sales"])

    negative_sales = (df["Sales"] < 0).sum()
    if negative_sales:
        warnings.warn(f"Dropping {negative_sales} row(s) with negative Sales values.")
        df = df[df["Sales"] >= 0]

    removed = initial_rows - len(df)
    if removed:
        warnings.warn(f"Removed {removed} invalid row(s) total ({initial_rows} -> {len(df)}).")

    df = df.sort_values("Date").reset_index(drop=True)

    daily = (
        df.groupby("Date", as_index=False)["Sales"]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return daily
