"""Generate a reproducible sample sales dataset for SalesSense."""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).parent / "sample_sales.csv"

PRODUCTS = {
    "Electronics": [
        ("Laptop", 899.0, 8),
        ("Smartphone", 649.0, 15),
        ("Headphones", 129.0, 25),
        ("Tablet", 399.0, 12),
        ("Smart Watch", 249.0, 18),
    ],
    "Clothing": [
        ("T-Shirt", 24.0, 40),
        ("Jeans", 59.0, 22),
        ("Jacket", 89.0, 14),
        ("Sneakers", 79.0, 20),
        ("Dress", 69.0, 16),
    ],
    "Home & Kitchen": [
        ("Blender", 49.0, 18),
        ("Coffee Maker", 79.0, 12),
        ("Cookware Set", 129.0, 8),
        ("Vacuum Cleaner", 199.0, 6),
        ("Air Fryer", 89.0, 14),
    ],
    "Beauty": [
        ("Shampoo", 12.0, 35),
        ("Moisturizer", 28.0, 28),
        ("Lipstick", 18.0, 30),
        ("Perfume", 65.0, 10),
        ("Face Serum", 42.0, 15),
    ],
}


def _product_factor(product_name: str) -> float:
    """Deterministic per-product multiplier (stable across runs)."""
    idx = sum(ord(c) for c in product_name) % 100
    return 0.85 + (idx / 100) * 0.3


def _weekly_multiplier(category: str, day_of_week: int) -> float:
    """Day-of-week demand patterns vary by category."""
    is_weekend = day_of_week >= 5
    patterns = {
        "Electronics": 1.15 if is_weekend else 0.95,
        "Clothing": 1.35 if is_weekend else 0.88,
        "Home & Kitchen": 1.08 if is_weekend else 1.0,
        "Beauty": 1.12 if is_weekend else 0.96,
    }
    return patterns[category]


def _seasonal_multiplier(date: pd.Timestamp, category: str) -> float:
    """Monthly and seasonal variation."""
    month = date.month
    day_of_year = date.dayofyear

    # Broad annual cycle
    annual = 1.0 + 0.12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Category-specific seasonal boosts
    category_boost = 0.0
    if category == "Clothing":
        # Back-to-school (Aug-Sep) and holiday gifting (Nov-Dec)
        if month in (8, 9):
            category_boost += 0.18
        if month in (11, 12):
            category_boost += 0.25
        if month in (6, 7):
            category_boost += 0.10
    elif category == "Electronics":
        if month in (11, 12):
            category_boost += 0.30
        if month == 1:
            category_boost += 0.12  # post-holiday returns offset by sales
    elif category == "Home & Kitchen":
        if month in (11, 12):
            category_boost += 0.20
        if month in (3, 4):
            category_boost += 0.08  # spring home refresh
    elif category == "Beauty":
        if month in (2, 12):
            category_boost += 0.15  # Valentine's + holidays
        if month in (6, 7):
            category_boost += 0.10

    return annual + category_boost


def _promo_multiplier(date: pd.Timestamp) -> float:
    """Promotional and high-demand periods."""
    month, day = date.month, date.day

    # Black Friday week (last week of November)
    if month == 11 and day >= 22:
        return 1.55

    # Christmas shopping season
    if month == 12 and day <= 24:
        return 1.35

    # New Year clearance
    if month == 1 and day <= 7:
        return 1.18

    # Mid-year sale (mid June)
    if month == 6 and 10 <= day <= 20:
        return 1.22

    # Random short promo windows (deterministic via date hash)
    week_index = date.isocalendar()[1]
    if week_index % 9 == 0:
        return 1.10

    return 1.0


def generate_sales_data(start: str = "2023-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Build daily sales records for all products over the given period."""
    rng = np.random.default_rng(RANDOM_SEED)

    dates = pd.date_range(start=start, end=end, freq="D")
    rows = []

    for date in dates:
        for category, products in PRODUCTS.items():
            weekly = _weekly_multiplier(category, date.dayofweek)
            seasonal = _seasonal_multiplier(date, category)
            promo = _promo_multiplier(date)

            for product_name, unit_price, base_units in products:
                product_factor = _product_factor(product_name)

                # Log-normal noise for natural variation
                noise = rng.lognormal(mean=0.0, sigma=0.18)

                units = base_units * weekly * seasonal * promo * product_factor * noise
                units = max(1, round(units))

                # Price varies slightly (discounts, bundles)
                effective_price = unit_price * rng.uniform(0.92, 1.05)
                sales = round(units * effective_price, 2)

                rows.append(
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Product": product_name,
                        "Category": category,
                        "Sales": sales,
                        "Units": units,
                    }
                )

    df = pd.DataFrame(rows)
    df["Sales"] = df["Sales"].astype(float)
    df["Units"] = df["Units"].astype(int)
    return df


def main() -> None:
    df = generate_sales_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
