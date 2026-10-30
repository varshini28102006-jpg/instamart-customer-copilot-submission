"""run_all.py -- End-to-end execution of all 4 parts of the Instamart Capstone Project.

Reproduces all clean data, analytical database, segmentations, models, triage
outputs, RAG evaluations, and agent execution ledgers.
"""

import sys
import os
import time

print("=" * 70)
print("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
print("Executing Full End-to-End Pipeline (Parts A, B, C, D)")
print("=" * 70)

# Step 1: Ensure raw data is generated
if not os.path.exists("data/raw/orders.csv"):
    print("\n[Data Generation] Running generate_data.py...")
    import generate_data

# Step 2: Part A - Data Foundation
print("\n" + "=" * 50)
print("PART A: Data Cleaning, SQLite Storage & Validation")
print("=" * 50)
from src.data_cleaning import run_pipeline as run_cleaning
audit, cust, prod, orders = run_cleaning()
print(f"Data cleaned: {len(cust)} customers, {len(prod)} products, {len(orders)} orders.")

from src.database import populate_database, run_all_sql_queries
populate_database()
print("Populated SQLite database at data/instamart.db.")
sql_results = run_all_sql_queries()
print(f"Executed all {len(sql_results)} SQL queries successfully.")

from src.analysis import cross_check_q2, eda_order_amount
pivot, city_stats = cross_check_q2()
eda = eda_order_amount()
print("Completed EDA and pandas cross-checks.")

from src.validation import validate_checkpoints
validate_checkpoints()

# Step 3: Part B - Customer Segments & Lapse Prediction
print("\n" + "=" * 50)
print("PART B: Customer Segments and Lapse Prediction")
print("=" * 50)
from src.segmentation import compute_rfm, evaluate_kmeans_clusters, build_customer_segments
rfm = compute_rfm()
k_eval, scaler, X_scaled = evaluate_kmeans_clusters(rfm)
segments_df, profiles, pca_info = build_customer_segments(rfm)
print(f"Segmentation complete: {len(segments_df)} customers clustered into 4 segments.")
print("Saved outputs/customer_segments.csv.")

from src.lapse_prediction import build_cohort_and_features, train_and_evaluate_models
cohort_df = build_cohort_and_features()
model_eval = train_and_evaluate_models(cohort_df)
print(f"Lapse modeling complete: Selected {model_eval['selected_model']}.")
print(f"Test F1: {model_eval['random_forest']['test_f1']} (RF), {model_eval['logistic_regression']['test_f1']} (LR).")
print("Saved outputs/lapse_model.joblib and outputs/retention_list.csv.")

# Step 4: Part C - Support Ticket Triage with Guardrails
print("\n" + "=" * 50)
print("PART C: Support Ticket Triage with Guardrails")
print("=" * 50)
from src.ticket_triage import generate_outputs as run_triage
b_df, f_df = run_triage()
print(f"Triage complete: Processed {len(f_df)} tickets.")
print("Saved outputs/triage_budget_demo.csv and outputs/triage_results.csv.")

# Step 5: Part D - Policy RAG Assistant & Refund Agent
print("\n" + "=" * 50)
print("PART D: Policy RAG Assistant and Refund Agent")
print("=" * 50)
from src.policy_rag import evaluate_policy_rag
s10 = evaluate_policy_rag(overlap=10)
s0 = evaluate_policy_rag(overlap=0)
print(f"RAG Evaluation (overlap 10): Doc Retrieval={s10['doc_retrieval_acc']}, Fact Match={s10['expected_fact_acc']}, Fallback={s10['fallback_acc']}")
print(f"RAG Evaluation (overlap 0): Fact Match={s0['expected_fact_acc']} (Question 1 breaks as expected)")

from src.refund_agent import run_all_scenarios, demonstrate_crash_and_resume
sc_res = run_all_scenarios()
print(f"Executed all {len(sc_res)} refund scenarios. Saved outputs/agent_runs.jsonl and outputs/refund_ledger.jsonl.")

# Safety demonstrations
from src.refund_agent import run_agent
scenarios_df = pd.read_csv("data/raw/agent_scenarios.csv") if "pd" in locals() else None
if scenarios_df is None:
    import pandas as pd
    scenarios_df = pd.read_csv("data/raw/agent_scenarios.csv")
s1_msg = scenarios_df.iloc[0]["message"]
token_demo = run_agent(s1_msg, "S1_token_demo", token_limit=200)
timeout_demo = run_agent(s1_msg, "S1_timeout_demo", time_limit_seconds=0)
print(f"Safety Demo: Token limit stop -> {token_demo['status']}, Timeout stop -> {timeout_demo['status']}.")

# Crash & Resumption demonstration
tools_before, tools_total, final_res = demonstrate_crash_and_resume()
print(f"Crash & Resumption Demo: Tools before crash: {tools_before}; All tools executed: {tools_total}.")

print("\n" + "=" * 70)
print("ALL CAPSTONE TASKS AND DELIVERABLES SUCCESSFULLY EXECUTED AND SAVED!")
print("=" * 70)
