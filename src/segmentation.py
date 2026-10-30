"""segmentation.py -- Customer RFM calculation, KMeans clustering, and PCA profiling."""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")


def compute_rfm(clean_orders_path="data/clean/orders.csv", snapshot=SNAPSHOT_DATE):
    """Computes Recency, Frequency, and Monetary metrics for active customers."""
    orders = pd.read_csv(clean_orders_path)
    completed = orders[orders["status"] == "completed"].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"])

    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()

    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)

    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
    return rfm


def evaluate_kmeans_clusters(rfm_df, k_range=range(2, 9), random_state=42):
    """Evaluates KMeans clustering for K=2 to 8, recording inertia and silhouette score."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_df[["recency", "frequency", "monetary"]])

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X_scaled)
        sil = silhouette_score(X_scaled, km.labels_)
        results.append({
            "k": k,
            "inertia": round(km.inertia_, 1),
            "silhouette": round(sil, 3)
        })

    return results, scaler, X_scaled


def build_customer_segments(rfm_df, n_clusters=4, random_state=42, output_path="outputs/customer_segments.csv"):
    """Clusters customers into K=4 segments, profiles them, tags premium, and writes output CSV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_df[["recency", "frequency", "monetary"]])

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_scaled)

    df = rfm_df.copy()
    df["segment_id"] = labels

    # Profile segments by mean metrics
    profiles = df.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()

    # Identify premium segment: highest mean monetary value
    premium_seg_id = int(profiles.loc[profiles["mean_monetary"].idxmax(), "segment_id"])

    # Business naming based on profiles
    segment_names = {}
    actions = {}

    for _, row in profiles.iterrows():
        sid = int(row["segment_id"])
        if sid == premium_seg_id:
            segment_names[sid] = "High-Value Champions (Premium)"
            actions[sid] = "VIP dedicated concierge desk, early catalogue access, and zero-fee express slots."
        elif row["mean_frequency"] > 12:
            segment_names[sid] = "Loyal Daily Regulars"
            actions[sid] = "Monthly grocery subscription pass, tiered cashbacks, and quick reorder widgets."
        elif row["mean_recency"] > 40:
            segment_names[sid] = "At-Risk / Dormant Shoppers"
            actions[sid] = "Targeted win-back discounts, SMS reactivation vouchers, and feedback outreach."
        else:
            segment_names[sid] = "Steady Occasional Buyers"
            actions[sid] = "Cross-category discovery promotions and weekend bulk-purchase nudges."

    df["segment_name"] = df["segment_id"].map(segment_names)
    df["is_premium"] = df["segment_id"] == premium_seg_id

    # PCA projection
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_1"] = np.round(pca_coords[:, 0], 4)
    df["pca_2"] = np.round(pca_coords[:, 1], 4)

    # Save output CSV with exact required columns
    out_cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df[out_cols].to_csv(output_path, index=False)

    pca_variance = {
        "pca_1_ratio": round(float(pca.explained_variance_ratio_[0]), 3),
        "pca_2_ratio": round(float(pca.explained_variance_ratio_[1]), 3),
        "total_explained_variance": round(float(sum(pca.explained_variance_ratio_)), 3)
    }

    return df, profiles, pca_variance


if __name__ == "__main__":
    rfm = compute_rfm()
    print("--- RFM Summary ---")
    print(rfm.describe())
    cluster_eval, scaler, X_scaled = evaluate_kmeans_clusters(rfm)
    print("\n--- KMeans Evaluation ---")
    for row in cluster_eval:
        print(f"K={row['k']}: Inertia={row['inertia']}, Silhouette={row['silhouette']}")
    segments_df, profiles, pca_info = build_customer_segments(rfm)
    print("\n--- Segment Profiles ---")
    print(profiles)
    print("\n--- PCA Info ---")
    print(pca_info)
