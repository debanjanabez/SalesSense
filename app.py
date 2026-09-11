import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from ml.model import compare_models
from ml.feature_engineering import create_features
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

DATA_PATH = PROJECT_ROOT / "data" / "sample_sales.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_sales_model.joblib"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.json"


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="SalesSense",
    page_icon="📈",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL + DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_features():
    with open(FEATURE_COLUMNS_PATH, "r") as f:
        return json.load(f)


model = load_model()
df = load_data()
feature_columns = load_features()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📈 SalesSense")

st.markdown(
    """
    ### AI-Powered Sales Forecasting Dashboard

    SalesSense uses machine learning to analyze historical sales
    patterns and generate intelligent sales predictions.
    """
)

st.divider()


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Sales",
        f"₹{df['Sales'].sum():,.0f}"
    )

with col2:
    st.metric(
        "Average Daily Sales",
        f"₹{df['Sales'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Highest Sales",
        f"₹{df['Sales'].max():,.0f}"
    )

with col4:
    st.metric(
        "Sales Records",
        f"{len(df):,}"
    )


st.divider()


# --------------------------------------------------
# SALES TREND
# --------------------------------------------------

st.subheader("📊 Sales Trend")

chart_data = df.set_index("Date")[["Sales"]]

st.line_chart(chart_data)
# --------------------------------------------------
# CATEGORY PERFORMANCE
# --------------------------------------------------

st.divider()

st.subheader("📊 Sales by Category")

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(category_sales)
# --------------------------------------------------
# TOP PRODUCTS
# --------------------------------------------------

st.divider()

st.subheader("🏆 Top Performing Products")

product_sales = (
    df.groupby("Product")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(product_sales)
# --------------------------------------------------
# BUSINESS INSIGHTS
# --------------------------------------------------

st.divider()

st.subheader("💡 Business Insights")

total_sales = df["Sales"].sum()
average_daily_sales = df["Sales"].mean()
highest_daily_sales = df["Sales"].max()

best_category = category_sales.index[0]
best_category_sales = category_sales.iloc[0]

best_product = product_sales.index[0]
best_product_sales = product_sales.iloc[0]

st.markdown(
    f"""
    ### 📈 Overall Performance

    - **Total sales:** ₹{total_sales:,.0f}
    - **Average daily sales:** ₹{average_daily_sales:,.0f}
    - **Highest recorded sale:** ₹{highest_daily_sales:,.0f}

    ### 🏆 Top Performers

    - **Best category:** {best_category} — ₹{best_category_sales:,.0f}
    - **Top product:** {best_product} — ₹{best_product_sales:,.0f}
    """
)
# --------------------------------------------------
# SALES TREND ANALYSIS
# --------------------------------------------------

recent_sales = df.sort_values("Date").tail(7)["Sales"].mean()
previous_sales = df.sort_values("Date").iloc[-14:-7]["Sales"].mean()

change_percentage = (
    (recent_sales - previous_sales)
    / previous_sales
) * 100

st.subheader("📈 Sales Trend")

if change_percentage > 5:
    st.success(
        f"📈 Sales are increasing. "
        f"Recent sales are {change_percentage:.1f}% higher "
        f"than the previous 7-day period."
    )

elif change_percentage < -5:
    st.warning(
        f"📉 Sales are declining. "
        f"Recent sales are {abs(change_percentage):.1f}% lower "
        f"than the previous 7-day period."
    )

else:
    st.info(
        f"➡️ Sales are relatively stable. "
        f"The change over the previous 7-day period is "
        f"{change_percentage:+.1f}%."
    )
# --------------------------------------------------
# PREPARE DAILY SALES DATA
# --------------------------------------------------

daily_sales = (
    df.groupby("Date", as_index=False)["Sales"]
    .sum()
    .sort_values("Date")
    .reset_index(drop=True)
)
# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

st.divider()

st.subheader("🚨 Sales Anomaly Detection")

daily_analysis = daily_sales.copy()

# Calculate rolling statistics using the previous 30 days
daily_analysis["rolling_mean"] = (
    daily_analysis["Sales"]
    .rolling(window=30)
    .mean()
    .shift(1)
)

daily_analysis["rolling_std"] = (
    daily_analysis["Sales"]
    .rolling(window=30)
    .std()
    .shift(1)
)

# Define upper and lower boundaries
daily_analysis["upper_limit"] = (
    daily_analysis["rolling_mean"]
    + 2 * daily_analysis["rolling_std"]
)

daily_analysis["lower_limit"] = (
    daily_analysis["rolling_mean"]
    - 2 * daily_analysis["rolling_std"]
)

# Identify anomalies
daily_analysis["is_anomaly"] = (
    (daily_analysis["Sales"] > daily_analysis["upper_limit"])
    | (daily_analysis["Sales"] < daily_analysis["lower_limit"])
)

anomalies = daily_analysis[
    daily_analysis["is_anomaly"]
].copy()

if anomalies.empty:

    st.success(
        "✅ No significant sales anomalies detected."
    )

else:

    st.warning(
        f"⚠️ {len(anomalies)} unusual sales days detected."
    )

    display_anomalies = anomalies[
        ["Date", "Sales", "rolling_mean"]
    ].copy()

    display_anomalies["Difference (%)"] = (
        (
            display_anomalies["Sales"]
            - display_anomalies["rolling_mean"]
        )
        / display_anomalies["rolling_mean"]
        * 100
    )

    display_anomalies["Date"] = (
        display_anomalies["Date"]
        .dt.strftime("%d %b %Y")
    )

    display_anomalies["Sales"] = (
        display_anomalies["Sales"]
        .round(0)
    )

    display_anomalies["rolling_mean"] = (
        display_anomalies["rolling_mean"]
        .round(0)
    )

    display_anomalies["Difference (%)"] = (
        display_anomalies["Difference (%)"]
        .round(1)
    )

    display_anomalies = display_anomalies.sort_values(
        "Difference (%)",
        key=lambda x: x.abs(),
        ascending=False
    )

    st.dataframe(
        display_anomalies.head(10),
        use_container_width=True,
        hide_index=True
    )
    # --------------------------------------------------
# INVENTORY RECOMMENDATIONS
# --------------------------------------------------

st.divider()

st.subheader("📦 Inventory Recommendations")

# Calculate overall product demand
product_demand = (
    df.groupby(["Product", "Category"])
    .agg(
        Total_Units=("Units", "sum"),
        Average_Daily_Units=("Units", "mean"),
        Total_Sales=("Sales", "sum")
    )
    .reset_index()
)

# Calculate recent 7-day demand
latest_date = df["Date"].max()

recent_7_days = df[
    df["Date"] > latest_date - pd.Timedelta(days=7)
]

recent_demand = (
    recent_7_days.groupby("Product")["Units"]
    .mean()
    .reset_index()
    .rename(columns={"Units": "Recent_7_Day_Avg"})
)

# Combine demand information
product_demand = product_demand.merge(
    recent_demand,
    on="Product",
    how="left"
)

st.dataframe(
    product_demand,
    use_container_width=True,
    hide_index=True
)
# Calculate demand trend
product_demand["Demand_Trend_%"] = (
    (
        product_demand["Recent_7_Day_Avg"]
        - product_demand["Average_Daily_Units"]
    )
    / product_demand["Average_Daily_Units"]
    * 100
)

# Estimate units needed for the next 14 days
product_demand["Estimated_14_Day_Demand"] = (
    product_demand["Recent_7_Day_Avg"] * 14
)

# Generate inventory recommendations
def inventory_recommendation(trend):
    if trend >= 15:
        return "🔴 Stock Up"
    elif trend >= 5:
        return "🟡 Monitor"
    else:
        return "🟢 Stock Level OK"

product_demand["Recommendation"] = (
    product_demand["Demand_Trend_%"]
    .apply(inventory_recommendation)
)

# Sort products by urgency
product_demand = product_demand.sort_values(
    "Demand_Trend_%",
    ascending=False
)

st.subheader("📋 Recommended Inventory Actions")

display_inventory = product_demand[
    [
        "Product",
        "Category",
        "Average_Daily_Units",
        "Recent_7_Day_Avg",
        "Demand_Trend_%",
        "Estimated_14_Day_Demand",
        "Recommendation"
    ]
].copy()

display_inventory["Average_Daily_Units"] = (
    display_inventory["Average_Daily_Units"].round(1)
)

display_inventory["Recent_7_Day_Avg"] = (
    display_inventory["Recent_7_Day_Avg"].round(1)
)

display_inventory["Demand_Trend_%"] = (
    display_inventory["Demand_Trend_%"].round(1)
)

display_inventory["Estimated_14_Day_Demand"] = (
    display_inventory["Estimated_14_Day_Demand"].round(0)
)

st.dataframe(
    display_inventory,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# FUTURE FORECASTING
# --------------------------------------------------

st.divider()
# Recommendation summary
stock_up_count = (
    product_demand["Recommendation"] == "🔴 Stock Up"
).sum()

monitor_count = (
    product_demand["Recommendation"] == "🟡 Monitor"
).sum()

ok_count = (
    product_demand["Recommendation"] == "🟢 Stock Level OK"
).sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🔴 Stock Up", stock_up_count)

with col2:
    st.metric("🟡 Monitor", monitor_count)

with col3:
    st.metric("🟢 Stock Level OK", ok_count)
# Priority products
st.subheader("🚨 Priority Products")

priority_products = product_demand[
    product_demand["Recommendation"] == "🔴 Stock Up"
].head(5)

if priority_products.empty:
    st.success("✅ No products currently require urgent restocking.")
else:
    st.dataframe(
        priority_products[
            [
                "Product",
                "Category",
                "Recent_7_Day_Avg",
                "Demand_Trend_%",
                "Estimated_14_Day_Demand",
            ]
        ].rename(
            columns={
                "Recent_7_Day_Avg": "Recent Daily Demand",
                "Demand_Trend_%": "Demand Increase (%)",
                "Estimated_14_Day_Demand": "14-Day Estimated Demand",
            }
        ),
        use_container_width=True,
        hide_index=True
    )

st.subheader("🔮 Future Sales Forecast")

st.write(
    "Predict sales for the upcoming days using historical sales patterns "
    "and the trained XGBoost model."
)

forecast_days = st.selectbox(
    "Forecast Horizon",
    [7, 30],
    format_func=lambda x: f"Next {x} Days"
)

# --------------------------------------------------
# CREATE FEATURES FOR FUTURE DATE
# --------------------------------------------------

def create_future_features(history, target_date):

    sales = history["Sales"].tolist()

    features = {
        "lag_1": sales[-1],
        "lag_7": sales[-7],
        "lag_14": sales[-14],
        "lag_30": sales[-30],
        "rolling_mean_7": sum(sales[-7:]) / 7,
        "rolling_mean_14": sum(sales[-14:]) / 14,
        "rolling_mean_30": sum(sales[-30:]) / 30,
        "day_of_week": target_date.dayofweek,
        "month": target_date.month,
        "quarter": target_date.quarter,
        "week_of_year": target_date.isocalendar().week,
        "is_weekend": int(target_date.dayofweek >= 5)
    }

    return pd.DataFrame([features])


# --------------------------------------------------
# GENERATE FUTURE FORECAST
# --------------------------------------------------

if st.button("🚀 Generate Forecast", type="primary"):

    history = daily_sales.copy()

    forecasts = []

    last_date = history["Date"].max()

    for i in range(1, forecast_days + 1):

        future_date = last_date + pd.Timedelta(days=i)

        X_future = create_future_features(
            history,
            future_date
        )

        # Make sure feature order matches training
        X_future = X_future[feature_columns]

        prediction = model.predict(X_future)[0]

        # Avoid negative sales predictions
        prediction = max(0, prediction)

        forecasts.append({
            "Date": future_date,
            "Predicted Sales": prediction
        })

        # Add prediction to history
        history = pd.concat(
            [
                history,
                pd.DataFrame({
                    "Date": [future_date],
                    "Sales": [prediction]
                })
            ],
            ignore_index=True
        )


    forecast_df = pd.DataFrame(forecasts)


    # --------------------------------------------------
    # FORECAST SUMMARY
    # --------------------------------------------------

    total_forecast = forecast_df["Predicted Sales"].sum()
    average_forecast = forecast_df["Predicted Sales"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Forecast Period",
            f"{forecast_days} Days"
        )

    with col2:
        st.metric(
            "Expected Sales",
            f"₹{total_forecast:,.0f}"
        )

    with col3:
        st.metric(
            "Average Daily Sales",
            f"₹{average_forecast:,.0f}"
        )


    # --------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------

    st.subheader("📈 Sales Forecast")

    chart_history = daily_sales.tail(60).copy()

    chart_history = chart_history.rename(
        columns={"Sales": "Historical Sales"}
    )

    chart_forecast = forecast_df.set_index("Date")[
        ["Predicted Sales"]
    ]

    chart_history = chart_history.set_index("Date")

    combined_chart = pd.concat(
        [
            chart_history,
            chart_forecast
        ],
        axis=1
    )

    st.line_chart(combined_chart)


    # --------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------

    st.subheader("📅 Forecast Details")

    display_forecast = forecast_df.copy()

    display_forecast["Date"] = (
        display_forecast["Date"].dt.strftime("%d %b %Y")
    )

    display_forecast["Predicted Sales"] = (
        display_forecast["Predicted Sales"]
        .round(2)
    )

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True
    )
# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.divider()

st.subheader("🤖 Model Comparison")

st.write(
    "Compare different machine learning models using the same "
    "historical sales data."
)

# Prepare data for model comparison
comparison_data = daily_sales.copy()

comparison_features = create_features(comparison_data)

X = comparison_features.drop(columns=["Date", "Sales"])
y = comparison_features["Sales"]

# Chronological train/test split
split_index = int(len(comparison_features) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Run model comparison
if st.button("🔍 Compare Models"):

    comparison_results = compare_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    comparison_df = pd.DataFrame(comparison_results)

    comparison_df["MAE"] = comparison_df["MAE"].round(2)
    comparison_df["RMSE"] = comparison_df["RMSE"].round(2)
    comparison_df["R²"] = comparison_df["R²"].round(4)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # Find the best model based on R²
    best_model = comparison_df.loc[
        comparison_df["R²"].idxmax(),
        "Model"
    ]

    best_r2 = comparison_df["R²"].max()

    st.success(
        f"🏆 Best performing model: **{best_model}** "
        f"with an R² score of **{best_r2:.2%}**."
    )