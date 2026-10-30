"""data_cleaning.py -- Auditing and cleaning pipeline for Instamart exports."""

import os
import numpy as np
import pandas as pd


def audit_raw_data(data_dir="data/raw"):
    """Audits raw customer, product, order, and ticket tables."""
    cust_path = os.path.join(data_dir, "customers.csv")
    prod_path = os.path.join(data_dir, "products.csv")
    ord_path = os.path.join(data_dir, "orders.csv")
    tick_path = os.path.join(data_dir, "tickets.csv")

    cust = pd.read_csv(cust_path)
    prod = pd.read_csv(prod_path)
    orders = pd.read_csv(ord_path)
    tickets = pd.read_csv(tick_path, keep_default_na=False)

    # Auditing orders
    raw_orders_rows = len(orders)
    exact_duplicates = int(orders.duplicated().sum())
    pending_rows = int((orders["status"] == "pending").sum())
    distinct_order_ids = int(orders["order_id"].nunique())
    non_numeric_amounts = int(pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum())

    deliv_numeric = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    negative_deliv = int((deliv_numeric < 0).sum())
    blank_deliv = int(orders["delivery_minutes"].isna().sum() + (orders["delivery_minutes"] == "").sum())

    # Auditing customers
    blank_age = int(cust["age"].isna().sum() + (cust["age"] == "").sum())
    cust_age_num = pd.to_numeric(cust["age"], errors="coerce")
    ages_outside_10_100 = int(((cust_age_num < 10) | (cust_age_num > 100)).sum())
    distinct_raw_cities = int(cust["city"].nunique())
    raw_genders = cust["gender"].dropna()
    raw_genders = raw_genders[raw_genders != ""]
    distinct_raw_genders = int(raw_genders.nunique())
    blank_gender = int((cust["gender"].isna() | (cust["gender"] == "")).sum())

    audit_summary = {
        "raw_orders_rows": raw_orders_rows,
        "exact_duplicates": exact_duplicates,
        "pending_rows": pending_rows,
        "distinct_order_ids": distinct_order_ids,
        "non_numeric_amounts": non_numeric_amounts,
        "negative_deliv": negative_deliv,
        "blank_deliv": blank_deliv,
        "blank_age": blank_age,
        "ages_outside_10_100": ages_outside_10_100,
        "distinct_raw_cities": distinct_raw_cities,
        "distinct_raw_genders": distinct_raw_genders,
        "blank_gender": blank_gender,
        "raw_customers_rows": len(cust),
        "raw_products_rows": len(prod),
        "raw_tickets_rows": len(tickets),
    }

    return audit_summary


def clean_customers(raw_path="data/raw/customers.csv", output_path="data/clean/customers.csv"):
    """Cleans customer records: standardizes city, gender, imputes age by city median, parses dates."""
    df = pd.read_csv(raw_path)

    # 1. Standardise city: strip and title case
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # 2. Map gender spellings
    gender_map = {
        "male": "male", "Male": "male", "M": "male", "m": "male", "MALE": "male",
        "female": "female", "Female": "female", "F": "female", "f": "female", "FEMALE": "female"
    }
    df["gender"] = df["gender"].fillna("").astype(str).str.strip().map(lambda g: gender_map.get(g, "unknown"))

    # 3. Handle impossible and missing ages
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = np.nan
    df["age"] = df.groupby("city")["age"].transform(lambda s: s.fillna(s.median()))

    # 4. Parse signup dates
    df["signup_date"] = pd.to_datetime(df["signup_date"])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Format dates as YYYY-MM-DD for clean CSV
        save_df = df.copy()
        save_df["signup_date"] = save_df["signup_date"].dt.strftime("%Y-%m-%d")
        save_df.to_csv(output_path, index=False)

    return df


def clean_products(raw_path="data/raw/products.csv", output_path="data/clean/products.csv"):
    """Cleans and standardizes product catalog."""
    df = pd.read_csv(raw_path)
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="raise")
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def clean_orders(raw_path="data/raw/orders.csv", products_df=None, output_path="data/clean/orders.csv"):
    """Cleans orders following the explicit 5-step order from the brief."""
    df = pd.read_csv(raw_path)

    if products_df is None:
        products_df = clean_products()

    # Step 1: Convert order_ts to datetime
    df["order_ts"] = pd.to_datetime(df["order_ts"])

    # Step 2: Remove exact duplicate rows
    df = df.drop_duplicates()

    # Step 3: Retain only the record with the latest order_ts per order_id
    df = df.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last")

    # Step 4: Convert order_amount to numeric, impute missing from product price * qty * (1 - disc/100)
    df = df.merge(products_df[["product_id", "category", "price_inr"]], on="product_id", how="left")
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")

    missing_amt = df["order_amount"].isna()
    df.loc[missing_amt, "order_amount"] = np.round(
        df.loc[missing_amt, "price_inr"]
        * df.loc[missing_amt, "quantity"]
        * (1.0 - df.loc[missing_amt, "discount_pct"] / 100.0),
        2
    )

    # Step 5: Handle delivery_minutes: set negatives to NaN.
    # Completed orders: impute with median delivery time of category among completed orders.
    # Cancelled orders: leave delivery_minutes missing (never delivered).
    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

    completed_mask = df["status"] == "completed"
    median_deliv = df[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    df.loc[completed_mask, "delivery_minutes"] = df.loc[completed_mask, "delivery_minutes"].fillna(median_deliv)

    # Drop temporary product join columns except product_id to keep original orders schema
    clean_cols = [
        "order_id", "customer_id", "product_id", "quantity",
        "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"
    ]
    df = df[clean_cols]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_df = df.copy()
        save_df["order_ts"] = save_df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        save_df.to_csv(output_path, index=False)

    return df


def run_pipeline():
    """Runs full data auditing and cleaning pipeline."""
    audit = audit_raw_data()
    cust = clean_customers()
    prod = clean_products()
    orders = clean_orders(products_df=prod)
    return audit, cust, prod, orders


if __name__ == "__main__":
    audit, cust, prod, orders = run_pipeline()
    print("Data cleaning completed successfully.")
    print(f"Clean customers: {len(cust)}")
    print(f"Clean products: {len(prod)}")
    print(f"Clean orders: {len(orders)}")
    completed_spend = orders[orders["status"] == "completed"]["order_amount"].sum()
    print(f"Completed orders total spend: INR {completed_spend:.2f}")
