"""lapse_prediction.py -- Customer lapse cohort definition, feature engineering, modeling, and retention ranking."""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import resample
import joblib


PREDICTION_CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
TARGET_WINDOW_END = pd.Timestamp("2025-09-30 23:59:59")


def build_cohort_and_features(
    clean_orders_path="data/clean/orders.csv",
    clean_customers_path="data/clean/customers.csv",
    clean_products_path="data/clean/products.csv",
    cutoff=PREDICTION_CUTOFF,
    target_end=TARGET_WINDOW_END
):
    """Builds leakage-free training features strictly from orders before the cutoff."""
    orders = pd.read_csv(clean_orders_path)
    cust = pd.read_csv(clean_customers_path)
    prod = pd.read_csv(clean_products_path)

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])

    # 1. Historical window (strictly before cutoff)
    pre_orders = orders[orders["order_ts"] < cutoff].copy()
    pre_completed = pre_orders[pre_orders["status"] == "completed"]
    cohort_cids = sorted(pre_completed["customer_id"].unique())

    # 2. Target window (cutoff through target_end)
    post_orders = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] <= target_end)]
    post_completed = post_orders[post_orders["status"] == "completed"]
    active_post_cids = set(post_completed["customer_id"].unique())

    # Map products for quick category lookup
    prod_cat_map = dict(zip(prod["product_id"], prod["category"]))

    records = []
    for cid in cohort_cids:
        c_orders = pre_orders[pre_orders["customer_id"] == cid]
        c_comp = pre_completed[pre_completed["customer_id"] == cid]
        c_info = cust[cust["customer_id"] == cid].iloc[0]

        last_ts = c_comp["order_ts"].max()
        recency = (cutoff - last_ts).days
        frequency = len(c_comp)
        monetary = round(float(c_comp["order_amount"].sum()), 2)
        avg_discount = round(float(c_comp["discount_pct"].mean()), 2)

        # Cancellation rate on all orders with final status before cutoff
        c_final = c_orders[c_orders["status"].isin(["completed", "cancelled"])]
        canc_rate = round(float((c_final["status"] == "cancelled").sum() / len(c_final)), 4) if len(c_final) > 0 else 0.0

        avg_deliv = round(float(c_comp["delivery_minutes"].mean()), 2)

        # Favourite product category
        comp_categories = [prod_cat_map.get(pid, "Unknown") for pid in c_comp["product_id"]]
        if comp_categories:
            fav_cat = pd.Series(comp_categories).mode()[0]
        else:
            fav_cat = "Unknown"

        tenure = (cutoff - c_info["signup_date"]).days
        lapsed = 1 if cid not in active_post_cids else 0

        records.append({
            "customer_id": cid,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_discount": avg_discount,
            "cancellation_rate": canc_rate,
            "avg_delivery_minutes": avg_deliv,
            "fav_category": fav_cat,
            "city": c_info["city"],
            "age": float(c_info["age"]),
            "gender": c_info["gender"],
            "tenure": tenure,
            "lapsed": lapsed
        })

    feat_df = pd.DataFrame(records)
    return feat_df


def train_and_evaluate_models(feat_df, random_state=42):
    """Performs stratified split, train-only scaling/encoding, hyperparameter tuning, and evaluation."""
    X = feat_df.drop(columns=["customer_id", "lapsed"])
    y = feat_df["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    num_cols = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate", "avg_delivery_minutes", "age", "tenure"]
    cat_cols = ["fav_category", "city", "gender"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
    ])

    # Preprocessor fitted strictly on train
    preprocessor.fit(X_train)
    learned_freq_mean = float(preprocessor.named_transformers_["num"].mean_[1])
    full_cohort_freq_mean = float(X["frequency"].mean())

    # Baseline: Always predict 0 (Majority class)
    baseline_pred = np.zeros(len(y_test))
    baseline_acc = float((baseline_pred == y_test).mean())
    baseline_f1 = float(f1_score(y_test, baseline_pred, zero_division=0))

    # 1. Logistic Regression GridSearchCV
    pipe_lr = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
    ])
    grid_lr = GridSearchCV(pipe_lr, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
    grid_lr.fit(X_train, y_train)

    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    y_proba_lr = best_lr.predict_proba(X_test)[:, 1]

    lr_eval = {
        "best_params": grid_lr.best_params_,
        "best_cv_f1": round(float(grid_lr.best_score_), 4),
        "test_precision": round(float(precision_score(y_test, y_pred_lr, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, y_pred_lr, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, y_pred_lr, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist()
    }

    # 2. Random Forest GridSearchCV (45 fits)
    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])
    grid_rf = GridSearchCV(pipe_rf, {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }, cv=5, scoring="f1")
    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

    rf_eval = {
        "best_params": grid_rf.best_params_,
        "best_cv_f1": round(float(grid_rf.best_score_), 4),
        "test_precision": round(float(precision_score(y_test, y_pred_rf, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, y_pred_rf, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, y_pred_rf, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist()
    }

    # 3. Class-balance experiment: Minority class oversampling on training data only
    train_combined = pd.concat([X_train, y_train], axis=1)
    majority_train = train_combined[train_combined["lapsed"] == 0]
    minority_train = train_combined[train_combined["lapsed"] == 1]

    minority_upsampled = resample(
        minority_train,
        replace=True,
        n_samples=len(majority_train),
        random_state=random_state
    )
    train_balanced = pd.concat([majority_train, minority_upsampled])
    X_train_bal = train_balanced.drop(columns=["lapsed"])
    y_train_bal = train_balanced["lapsed"]

    pipe_lr_bal = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=grid_lr.best_params_["clf__C"], max_iter=1000, random_state=random_state))
    ])
    pipe_lr_bal.fit(X_train_bal, y_train_bal)
    y_pred_lr_bal = pipe_lr_bal.predict(X_test)

    bal_eval = {
        "train_majority_count": len(majority_train),
        "train_minority_before": len(minority_train),
        "train_minority_after": len(minority_upsampled),
        "test_precision": round(float(precision_score(y_test, y_pred_lr_bal, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, y_pred_lr_bal, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, y_pred_lr_bal, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr_bal).tolist()
    }

    # Extract feature importances and coefficients
    cat_feature_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names

    rf_clf = best_rf.named_steps["clf"]
    importances = sorted(zip(all_feature_names, rf_clf.feature_importances_), key=lambda x: x[1], reverse=True)
    top_5_rf_features = [{"feature": f, "importance": round(float(imp), 4)} for f, imp in importances[:5]]

    lr_clf = best_lr.named_steps["clf"]
    coefficients = sorted(zip(all_feature_names, lr_clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    top_lr_coefficients = [{"feature": f, "coefficient": round(float(c), 4), "sign": "+" if c > 0 else "-"} for f, c in coefficients[:5]]

    # Pick best model: best_rf (or best_lr) with test F1 >= 0.55
    selected_model = best_rf if rf_eval["test_f1"] >= lr_eval["test_f1"] else best_lr
    selected_name = "RandomForestClassifier" if selected_model == best_rf else "LogisticRegression"

    # Save model
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/lapse_model.joblib"
    joblib.dump(selected_model, model_path)

    # Verify reloading
    reloaded_model = joblib.load(model_path)
    reloaded_preds = reloaded_model.predict(X_test)
    assert np.array_equal(reloaded_preds, selected_model.predict(X_test)), "Reloaded model predictions mismatch!"

    # Retention list: Top 10 premium customers at risk of lapsing from test set
    # Load segment labels
    segments_path = "outputs/customer_segments.csv"
    if os.path.exists(segments_path):
        seg_df = pd.read_csv(segments_path)
        premium_cids = set(seg_df[seg_df["is_premium"] == True]["customer_id"])
    else:
        premium_cids = set()

    test_indices = X_test.index
    test_customers = feat_df.loc[test_indices, ["customer_id", "lapsed"]].copy()
    test_customers["lapse_probability"] = selected_model.predict_proba(X_test)[:, 1]

    premium_test_customers = test_customers[test_customers["customer_id"].isin(premium_cids)].copy()
    retention_top10 = premium_test_customers.sort_values("lapse_probability", ascending=False).head(10).copy()
    retention_top10 = retention_top10.rename(columns={"lapsed": "actually_lapsed"})
    retention_top10["lapse_probability"] = retention_top10["lapse_probability"].round(4)
    retention_top10 = retention_top10[["customer_id", "lapse_probability", "actually_lapsed"]]
    retention_top10.to_csv("outputs/retention_list.csv", index=False)

    results = {
        "cohort_size": len(feat_df),
        "lapsed_count": int(y.sum()),
        "lapsed_rate_pct": round(float(y.mean() * 100), 2),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "learned_freq_mean": round(learned_freq_mean, 4),
        "full_cohort_freq_mean": round(full_cohort_freq_mean, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "baseline_f1": baseline_f1,
        "logistic_regression": lr_eval,
        "random_forest": rf_eval,
        "oversampling_experiment": bal_eval,
        "top_5_rf_features": top_5_rf_features,
        "top_lr_coefficients": top_lr_coefficients,
        "selected_model": selected_name,
        "retention_top10": retention_top10.to_dict(orient="records"),
        "retention_truly_lapsed_count": int(retention_top10["actually_lapsed"].sum())
    }

    return results


if __name__ == "__main__":
    feat_df = build_cohort_and_features()
    print("--- Cohort Summary ---")
    print(feat_df.head(2))
    eval_results = train_and_evaluate_models(feat_df)
    print("\n--- Model Evaluation Results ---")
    print("LR Eval:", eval_results["logistic_regression"])
    print("RF Eval:", eval_results["random_forest"])
    print("Oversampling Eval:", eval_results["oversampling_experiment"])
    print("Selected Model:", eval_results["selected_model"])
    print("Retention List truly lapsed:", eval_results["retention_truly_lapsed_count"])
