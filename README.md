# SalesSense

ML-based sales forecasting project. Predicts future sales from historical CSV data using an XGBoost regression pipeline.

## Project structure

```
SalesSense/
├── data/           # Raw and processed sales CSV files
├── ml/             # ML pipeline modules
├── requirements.txt
└── README.md
```

## Pipeline (planned)

```
Sales CSV
  → Data Cleaning
  → Feature Engineering
  → Lag Features
  → XGBoost Regression
  → Model Evaluation
  → Future Sales Forecast
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Status

Initial project scaffold only. Pipeline implementation coming next.
