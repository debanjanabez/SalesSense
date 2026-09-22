import pandas as pd


def load_rossmann_data(store_id=1):
    """
    Load and prepare Rossmann sales data for one store.
    """

    # Load datasets
    train_path = "data/rossmann/train.csv"
    store_path = "data/rossmann/store.csv"

    sales = pd.read_csv(train_path)
    stores = pd.read_csv(store_path)

    # Convert date
    sales["Date"] = pd.to_datetime(sales["Date"])

    # Select requested store
    sales = sales[sales["Store"] == store_id].copy()

    # Merge store information
    sales = sales.merge(
        stores,
        on="Store",
        how="left"
    )

    # Sort chronologically
    sales = sales.sort_values("Date").reset_index(drop=True)

    # Keep only days when store was open
    sales = sales[sales["Open"] == 1].copy()

    # Calendar features
    sales["day"] = sales["Date"].dt.day
    sales["month"] = sales["Date"].dt.month
    sales["year"] = sales["Date"].dt.year
    sales["week_of_year"] = (
        sales["Date"].dt.isocalendar().week.astype(int)
    )

    # Lag features
    sales["lag_1"] = sales["Sales"].shift(1)
    sales["lag_7"] = sales["Sales"].shift(7)
    sales["lag_14"] = sales["Sales"].shift(14)
    sales["lag_30"] = sales["Sales"].shift(30)

    # Rolling sales averages
    sales["rolling_mean_7"] = (
        sales["Sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    sales["rolling_mean_14"] = (
        sales["Sales"]
        .shift(1)
        .rolling(14)
        .mean()
    )

    sales["rolling_mean_30"] = (
        sales["Sales"]
        .shift(1)
        .rolling(30)
        .mean()
    )

    # Fill optional store information
    sales["Promo2"] = sales["Promo2"].fillna(0)
    sales["Promo2SinceWeek"] = sales["Promo2SinceWeek"].fillna(0)
    sales["Promo2SinceYear"] = sales["Promo2SinceYear"].fillna(0)
    sales["PromoInterval"] = sales["PromoInterval"].fillna("None")

    # Remove only rows where lag/rolling features cannot be calculated
    required_features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30"
    ]

    sales = sales.dropna(
        subset=required_features
    ).reset_index(drop=True)

    return sales


if __name__ == "__main__":
    df = load_rossmann_data(store_id=1)

    print("Rossmann processed data")
    print("-----------------------")
    print("Shape:", df.shape)
    print(
        "Date range:",
        df["Date"].min().date(),
        "to",
        df["Date"].max().date()
    )
    print("Average sales:", round(df["Sales"].mean(), 2))

    print("\nColumns:")
    print(list(df.columns))

    print("\nSample:")
    print(df.head().to_string(index=False))