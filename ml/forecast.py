"""Generate future sales forecasts from a trained model."""

import pandas as pd
from xgboost import XGBRegressor


def forecast_future_sales(
    model: XGBRegressor, features: pd.DataFrame, periods: int
) -> pd.DataFrame:
    """Predict sales for the next `periods` time steps."""
    raise NotImplementedError
