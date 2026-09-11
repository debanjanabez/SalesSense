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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        })

    return results