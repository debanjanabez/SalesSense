"""Train and evaluate the XGBoost sales forecasting model."""

import json
import warnings
from pathlib import Path

import pandas as pd

from ml.evaluation import evaluate_model, plot_actual_vs_predicted
from ml.feature_engineering import create_features
from ml.model import save_model, train_xgboost
from ml.preprocessing import load_and_clean_data

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_sales.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "xgboost_sales_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
PLOT_PATH = MODELS_DIR / "actual_vs_predicted.png"

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "day_of_week",
    "month",
    "quarter",
    "week_of_year",
    "is_weekend",
]


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically without shuffling."""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def main() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        daily = load_and_clean_data(DATA_PATH)
        featured = create_features(daily)

        train_df, test_df = chronological_split(featured, train_ratio=0.8)

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["Sales"]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["Sales"]

        model = train_xgboost(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(model, X_test, y_test)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_model(model, MODEL_PATH)
        FEATURE_COLUMNS_PATH.write_text(json.dumps(FEATURE_COLUMNS, indent=2))
        plot_actual_vs_predicted(y_test, y_pred, PLOT_PATH)

    print("=" * 50)
    print("SalesSense XGBoost Training Results")
    print("=" * 50)
    print(f"Training samples:   {len(X_train)}")
    print(f"Testing samples:    {len(X_test)}")
    print(f"MAE:                {metrics['mae']:.2f}")
    print(f"RMSE:               {metrics['rmse']:.2f}")
    print(f"R²:                 {metrics['r2']:.4f}")
    print()
    print("First 10 actual vs predicted values:")
    comparison = pd.DataFrame(
        {
            "Date": test_df["Date"].iloc[:10].dt.strftime("%Y-%m-%d").values,
            "Actual": y_test.iloc[:10].values,
            "Predicted": y_pred[:10],
        }
    )
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print()
    print(f"Model saved to:           {MODEL_PATH}")
    print(f"Feature columns saved to: {FEATURE_COLUMNS_PATH}")
    print(f"Plot saved to:            {PLOT_PATH}")

    if caught:
        print()
        print(f"Warnings ({len(caught)}):")
        for w in caught:
            print(f"  - {w.category.__name__}: {w.message}")
    else:
        print()
        print("Warnings: None")
    print("Errors: None")


if __name__ == "__main__":
    main()
