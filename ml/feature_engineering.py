"""Create time-based and derived features from daily sales data."""

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lag, rolling, and calendar features for daily sales forecasting.

    Rolling features use only prior observations (shifted) to avoid leakage.
    Rows with NaN from lag/rolling windows are dropped before returning.
    """
    result = df.copy()
    sales = result["Sales"]

    result["lag_1"] = sales.shift(1)
    result["lag_7"] = sales.shift(7)
    result["lag_14"] = sales.shift(14)
    result["lag_30"] = sales.shift(30)

    prior_sales = sales.shift(1)
    result["rolling_mean_7"] = prior_sales.rolling(window=7).mean()
    result["rolling_mean_14"] = prior_sales.rolling(window=14).mean()
    result["rolling_mean_30"] = prior_sales.rolling(window=30).mean()

    result["day_of_week"] = result["Date"].dt.dayofweek
    result["month"] = result["Date"].dt.month
    result["quarter"] = result["Date"].dt.quarter
    result["week_of_year"] = result["Date"].dt.isocalendar().week.astype(int)
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

    feature_cols = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
    ]
    result = result.dropna(subset=feature_cols).reset_index(drop=True)

    return result
