"""XGBoost regression model training and persistence."""

from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train an XGBoost regressor with sensible defaults for daily sales."""
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    return model


def save_model(model: XGBRegressor, path: str | Path) -> None:
    """Serialize trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
