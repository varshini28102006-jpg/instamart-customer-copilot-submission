"""analysis.py -- Exploratory data analysis and pandas cross-checks."""

import pandas as pd
import numpy as np


def cross_check_q2(clean_orders_path="data/clean/orders.csv",
                   clean_customers_path="data/clean/customers.csv",
                   clean_products_path="data/clean/products.csv"):
    """Rebuilds Q2 with pd.pivot_table(..., margins=True) and verifies grand total."""
    orders = pd.read_csv(clean_orders_path)
    cust = pd.read_csv(clean_customers_path)
    prod = pd.read_csv(clean_products_path)

    completed = orders[orders["status"] == "completed"]
    merged = completed.merge(cust, on="customer_id").merge(prod, on="product_id")

    pivot = pd.pivot_table(
        merged,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        fill_value=0.0,
        margins=True,
        margins_name="Total"
    ).round(2)

    city_stats = completed.merge(cust, on="customer_id").groupby("city")["order_amount"].agg(
        sum="sum",
        mean="mean",
        count="count"
    ).round(2).reset_index()

    return pivot, city_stats


def eda_order_amount(clean_orders_path="data/clean/orders.csv"):
    """Computes descriptive statistics, quartiles, IQR fences, outliers, and correlations."""
    orders = pd.read_csv(clean_orders_path)
    completed = orders[orders["status"] == "completed"].copy()
    amt = completed["order_amount"]

    mean_val = float(amt.mean())
    median_val = float(amt.median())
    skew_val = float(amt.skew())
    q1 = float(amt.quantile(0.25))
    q3 = float(amt.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers_below = int((amt < lower_fence).sum())
    outliers_above = int((amt > upper_fence).sum())

    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_matrix = completed[corr_cols].corr()

    # Find highest correlation in absolute value (excluding diagonal)
    pairs = []
    for i in range(len(corr_cols)):
        for j in range(i + 1, len(corr_cols)):
            c1, c2 = corr_cols[i], corr_cols[j]
            val = corr_matrix.loc[c1, c2]
            pairs.append((c1, c2, val, abs(val)))
    pairs.sort(key=lambda x: x[3], reverse=True)
    top_pair = pairs[0]

    stats = {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "skewness": round(skew_val, 2),
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_fence": round(lower_fence, 2),
        "upper_fence": round(upper_fence, 2),
        "outliers_below": outliers_below,
        "outliers_above": outliers_above,
        "corr_matrix": corr_matrix.round(4),
        "top_corr_pair": f"{top_pair[0]} and {top_pair[1]}: {top_pair[2]:.4f}"
    }

    return stats


if __name__ == "__main__":
    pivot, city_stats = cross_check_q2()
    print("--- Q2 Pivot Table ---")
    print(pivot)
    print("\n--- City Stats ---")
    print(city_stats)
    print("\n--- EDA Stats ---")
    eda = eda_order_amount()
    for k, v in eda.items():
        if k != "corr_matrix":
            print(f"{k}: {v}")
    print("\nCorrelation Matrix:\n", eda["corr_matrix"])
