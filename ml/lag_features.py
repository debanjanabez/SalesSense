"""Add lag features for time-series forecasting."""

import pandas as pd


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add lagged sales columns for autoregressive modeling."""
    raise NotImplementedError
