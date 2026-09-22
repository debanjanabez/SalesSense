import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.rossmann_preprocessing import load_rossmann_data


def train_rossmann_model(store_id=1):
    """
    Train an XGBoost model on Rossmann sales data.
    """

    # Load processed Rossmann data
    df = load_rossmann_data(store_id=store_id)

    # Features available for forecasting
    feature_columns = [
        "DayOfWeek",
        "Promo",
        "SchoolHoliday",
        "day",
        "month",
        "year",
        "week_of_year",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
    ]

    X = df[feature_columns].copy()
    y = df["Sales"].copy()

    # Convert StateHoliday to numeric
    # 0 = no state holiday
    # a/b/c = different holiday types
    # We don't use it as a raw string feature.
    state_holiday = df["StateHoliday"].astype(str)

    X["state_holiday"] = state_holiday.map({
        "0": 0,
        "a": 1,
        "b": 2,
        "c": 3
    }).fillna(0)

    # Chronological train/test split
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Rossmann model")
    print("--------------")
    print("Store:", store_id)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Create model
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance")
    print("-----------------")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))

    # Show sample predictions
    results = pd.DataFrame({
        "Date": df.iloc[split_index:]["Date"].values,
        "Actual Sales": y_test.values,
        "Predicted Sales": predictions
    })

    print("\nSample Predictions")
    print("------------------")
    print(results.head(10).to_string(index=False))

    return model, results


if __name__ == "__main__":
    train_rossmann_model(store_id=1)