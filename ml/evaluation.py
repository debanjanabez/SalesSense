"""Model evaluation metrics and visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def compute_mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_r2(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    return float(r2_score(y_true, y_pred))


def evaluate_predictions(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> dict[str, float]:
    """Compute MAE, RMSE, and R² for actual vs predicted values."""
    return {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
    }


def evaluate_model(
    model: XGBRegressor, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """Generate predictions and compute regression metrics."""
    y_pred = model.predict(X)
    return evaluate_predictions(y, y_pred)


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    save_path: str | Path,
) -> None:
    """Save a scatter plot of actual vs predicted sales."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
