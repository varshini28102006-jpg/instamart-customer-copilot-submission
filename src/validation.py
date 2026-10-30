"""validation.py -- Formal validation script asserting reference checkpoints for Instamart Copilot.

Must print exactly 'ALL CHECKPOINTS PASSED' when all hold.
"""

import os
import sqlite3
import pandas as pd
import numpy as np


def validate_checkpoints():
    """Asserts all core checkpoints from tasks 3 and 4."""
    cust_path = "data/clean/customers.csv"
    prod_path = "data/clean/products.csv"
    ord_path = "data/clean/orders.csv"
    db_path = "data/instamart.db"

    assert os.path.exists(cust_path), f"Missing {cust_path}"
    assert os.path.exists(prod_path), f"Missing {prod_path}"
    assert os.path.exists(ord_path), f"Missing {ord_path}"
    assert os.path.exists(db_path), f"Missing {db_path}"

    cust = pd.read_csv(cust_path)
    prod = pd.read_csv(prod_path)
    orders = pd.read_csv(ord_path)

    # 1. Customer City Distribution
    expected_cities = {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197,
        "Chennai": 191, "Hyderabad": 191, "Mumbai": 184
    }
    actual_cities = cust["city"].value_counts().to_dict()
    assert actual_cities == expected_cities, f"City counts mismatch: {actual_cities} vs {expected_cities}"

    # 2. Customer Gender Distribution
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    actual_genders = cust["gender"].value_counts().to_dict()
    assert actual_genders == expected_genders, f"Gender counts mismatch: {actual_genders} vs {expected_genders}"

    # 3. Customer Age Quality (0 missing, between 18 and 60)
    assert cust["age"].isna().sum() == 0, "Missing ages found in cleaned customers"
    assert cust["age"].min() >= 18 and cust["age"].max() <= 60, f"Age out of bounds: min={cust['age'].min()}, max={cust['age'].max()}"

    # 4. Clean Orders Count (12,309 distinct order_ids)
    assert len(orders) == 12309, f"Orders count mismatch: {len(orders)} != 12309"
    assert orders["order_id"].nunique() == 12309, "Duplicate order_ids exist in cleaned orders"

    # 5. Order Status Breakdown (11,110 completed, 1,199 cancelled, 0 pending)
    status_counts = orders["status"].value_counts().to_dict()
    assert status_counts.get("completed") == 11110, f"Completed count mismatch: {status_counts.get('completed')}"
    assert status_counts.get("cancelled") == 1199, f"Cancelled count mismatch: {status_counts.get('cancelled')}"
    assert "pending" not in status_counts or status_counts["pending"] == 0, "Pending orders found in cleaned data"

    # 6. Order Amounts (0 missing, numeric)
    assert orders["order_amount"].isna().sum() == 0, "Missing order amounts found"

    # 7. Delivery Minutes Imputation & Preservation
    completed_orders = orders[orders["status"] == "completed"]
    cancelled_orders = orders[orders["status"] == "cancelled"]
    assert completed_orders["delivery_minutes"].isna().sum() == 0, "Missing delivery minutes in completed orders"
    assert cancelled_orders["delivery_minutes"].isna().sum() == 1199, "Imputed delivery minutes into cancelled orders"

    # 8. Completed Orders Revenue Sum (INR 2,998,668.80)
    total_revenue = round(float(completed_orders["order_amount"].sum()), 2)
    expected_revenue = 2998668.80
    assert abs(total_revenue - expected_revenue) < 0.01, f"Total revenue mismatch: {total_revenue} != {expected_revenue}"

    # 9. SQLite Database Tables & Row Counts
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM customers")
    assert cur.fetchone()[0] == 1200, "Database customers row count mismatch"
    cur.execute("SELECT count(*) FROM products")
    assert cur.fetchone()[0] == 40, "Database products row count mismatch"
    cur.execute("SELECT count(*) FROM orders")
    assert cur.fetchone()[0] == 12309, "Database orders row count mismatch"
    conn.close()

    # If all hold, print exact required string
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    validate_checkpoints()
