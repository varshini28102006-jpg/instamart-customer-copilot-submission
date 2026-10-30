"""database.py -- SQLite storage and query runner for Instamart Copilot."""

import os
import sqlite3
import pandas as pd


DB_PATH = os.path.join("data", "instamart.db")


def get_connection(db_path=DB_PATH):
    """Returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


def populate_database(db_path=DB_PATH, clean_dir="data/clean"):
    """Populates SQLite database from cleaned CSV files."""
    cust_path = os.path.join(clean_dir, "customers.csv")
    prod_path = os.path.join(clean_dir, "products.csv")
    ord_path = os.path.join(clean_dir, "orders.csv")

    cust = pd.read_csv(cust_path)
    prod = pd.read_csv(prod_path)
    orders = pd.read_csv(ord_path)

    conn = get_connection(db_path)
    try:
        cust.to_sql("customers", conn, if_exists="replace", index=False)
        prod.to_sql("products", conn, if_exists="replace", index=False)
        orders.to_sql("orders", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def load_sql_queries(sql_file="sql/queries.sql"):
    """Parses sql/queries.sql and returns a dictionary of query name to query string."""
    with open(sql_file, "r", encoding="utf-8") as f:
        content = f.read()

    queries = {}
    current_name = None
    current_sql = []

    for line in content.splitlines():
        if line.strip().startswith("-- @name:"):
            if current_name and current_sql:
                queries[current_name] = "\n".join(current_sql).strip()
            current_name = line.strip().split(":", 1)[1].strip()
            current_sql = []
        else:
            current_sql.append(line)

    if current_name and current_sql:
        queries[current_name] = "\n".join(current_sql).strip()

    return queries


def execute_query(query_sql, db_path=DB_PATH):
    """Executes a SQL query string against the database and returns a pandas DataFrame."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(query_sql, conn)
        return df
    finally:
        conn.close()


def run_all_sql_queries(sql_file="sql/queries.sql", db_path=DB_PATH):
    """Executes all 7 queries and returns their results."""
    queries = load_sql_queries(sql_file)
    results = {}
    for name, sql in queries.items():
        results[name] = execute_query(sql, db_path)
    return results


if __name__ == "__main__":
    populate_database()
    print("Database populated at:", DB_PATH)
    res = run_all_sql_queries()
    for qname, df in res.items():
        print(f"--- {qname} ({len(df)} rows) ---")
        print(df.head(2))
