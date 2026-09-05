"""End-to-end ML pipeline orchestration."""

from pathlib import Path

from ml.data_cleaning import clean_sales_data
from ml.evaluation import evaluate_model
from ml.feature_engineering import engineer_features
from ml.forecast import forecast_future_sales
from ml.lag_features import add_lag_features
from ml.model import save_model, train_model


def run_pipeline(data_path: Path) -> None:
    """Run the full sales forecasting pipeline."""
    # TODO: implement each stage
    # df = pd.read_csv(data_path)
    # df = clean_sales_data(df)
    # df = engineer_features(df)
    # df = add_lag_features(df)
    # model = train_model(X, y)
    # metrics = evaluate_model(model, X_test, y_test)
    # forecast = forecast_future_sales(model, features, periods=30)
    # save_model(model, "models/xgboost_sales.pkl")
    raise NotImplementedError
