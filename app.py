import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


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
# PREDICTION SECTION
# --------------------------------------------------

st.divider()

st.subheader("🔮 Sales Prediction")

st.write(
    "Select a date from the available dataset to generate "
    "a machine-learning prediction."
)

selected_date = st.date_input(
    "Select Date",
    value=df["Date"].max().date(),
    min_value=df["Date"].min().date(),
    max_value=df["Date"].max().date()
)


# --------------------------------------------------
# CREATE FEATURES FOR SELECTED DATE
# --------------------------------------------------

def create_prediction_features(data, selected_date):

    data = data.sort_values("Date").copy()

    target_date = pd.Timestamp(selected_date)

    row = data[data["Date"] == target_date]

    if row.empty:
        return None

    index = row.index[0]

    features = {
        "lag_1": data.loc[index, "Sales"] if index >= 1 else None,
        "lag_7": data.loc[index - 7, "Sales"] if index >= 7 else None,
        "lag_14": data.loc[index - 14, "Sales"] if index >= 14 else None,
        "lag_30": data.loc[index - 30, "Sales"] if index >= 30 else None,
    }

    if any(value is None for value in features.values()):
        return None

    sales_series = data["Sales"]

    features["rolling_mean_7"] = (
        sales_series.iloc[index - 7:index].mean()
    )

    features["rolling_mean_14"] = (
        sales_series.iloc[index - 14:index].mean()
    )

    features["rolling_mean_30"] = (
        sales_series.iloc[index - 30:index].mean()
    )

    features["day_of_week"] = target_date.dayofweek
    features["month"] = target_date.month
    features["quarter"] = target_date.quarter
    features["week_of_year"] = target_date.isocalendar().week
    features["is_weekend"] = int(target_date.dayofweek >= 5)

    return pd.DataFrame([features])


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

if st.button("🚀 Predict Sales", type="primary"):

    X = create_prediction_features(df, selected_date)

    if X is None:

        st.error(
            "Not enough historical data available to generate "
            "features for this date."
        )

    else:

        # Make sure feature order matches training
        X = X[feature_columns]

        prediction = model.predict(X)[0]

        st.success("Prediction generated successfully!")

        st.metric(
            "Predicted Sales",
            f"₹{prediction:,.2f}"
        )