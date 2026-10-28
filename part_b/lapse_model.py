"""Part B, Tasks 4-8 -- lapse prediction without leakage.

Run from the repository root:  python part_b/lapse_model.py
Cutoff 2025-09-01: features use ONLY orders dated before the cutoff (plus the clean customer
table); the label is whether the customer has a completed order in 2025-09-01..2025-09-30.
Segment labels from segments.py are NOT used as features (they include September).
"""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLEAN = os.path.join("data", "clean")
CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")      # label window: 2025-09-01 .. 2025-09-30 inclusive

NUMERIC = ["recency", "frequency", "monetary", "avg_discount_pct", "cancel_rate",
           "avg_delivery_minutes", "age", "tenure_days"]
NOMINAL = ["city", "gender", "fav_category"]


def build_dataset():
    """Return (X, y) for the cohort, one row per customer, sorted by customer_id."""
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
    customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"), parse_dates=["signup_date"])
    products = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

    before = orders[orders["order_ts"] < CUTOFF]                       # the ONLY rows features may use
    done_before = before[before["status"] == "completed"]
    in_window = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END)
                       & (orders["status"] == "completed")]

    # Leakage guard: no feature row may be dated on or after the cutoff.
    assert before["order_ts"].max() < CUTOFF

    # Cohort: at least one completed order before the cutoff.
    f = done_before.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"))
    f["recency"] = (CUTOFF - f["last_order"]).dt.days

    # Cancellation rate over ALL orders with a final status before the cutoff.
    status = before.groupby("customer_id")["status"]
    f["cancel_rate"] = status.apply(lambda s: (s == "cancelled").mean()).reindex(f.index)

    # Favourite category among completed orders; ties broken alphabetically (deterministic).
    counts = done_before.groupby(["customer_id", "category"]).size().reset_index(name="n")
    fav = (counts.sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
           .drop_duplicates("customer_id").set_index("customer_id")["category"])
    f["fav_category"] = fav

    f = f.join(customers.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
    f["tenure_days"] = (CUTOFF - f["signup_date"]).dt.days

    f["lapsed"] = (~f.index.isin(in_window["customer_id"])).astype(int)
    f = f.sort_index()
    return f[NUMERIC + NOMINAL], f["lapsed"]


def make_preprocessor():
    """k-1 one-hot columns for nominal features, standardised numeric features."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL),
    ])


def main():
    # ---------------- Task 4: define the problem ----------------
    X, y = build_dataset()
    print("## Task 4: lapse problem\n")
    print(f"cohort: {len(X):,} customers; lapsed: {y.sum()} ({y.mean():.2%})")
    assert len(X) == 1156 and y.sum() == 443
    assert X.isna().sum().sum() == 0, X.isna().sum()[lambda s: s > 0]
    print("no missing values in any feature column")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"train {len(X_train)} rows, test {len(X_test)} rows")
    assert (len(X_train), len(X_test)) == (924, 232)

    baseline = np.zeros(len(y_test), dtype=int)            # always predict "not lapsed"
    acc = accuracy_score(y_test, baseline)
    f1 = f1_score(y_test, baseline, zero_division=0)
    print(f"baseline (always 'not lapsed'): accuracy {acc:.1%}, lapsed-class F1 {f1:.2f}")
    assert round(acc, 3) == 0.616 and f1 == 0

    # ---------------- Task 5: preprocess without leakage ----------------
    print("\n## Task 5: preprocessing fitted on training rows only\n")
    pre = make_preprocessor().fit(X_train)                  # fit: TRAIN ONLY
    Xt_train, Xt_test = pre.transform(X_train), pre.transform(X_test)   # apply to both
    scaler = pre.named_transformers_["num"]
    i = NUMERIC.index("frequency")
    print(f"frequency mean learned by the scaler (train rows): {scaler.mean_[i]:.4f}")
    print(f"frequency mean over the full cohort:               {X['frequency'].mean():.4f}")
    names = pre.get_feature_names_out()
    print(f"{Xt_train.shape[1]} model columns: {len(NUMERIC)} numeric + "
          f"{Xt_train.shape[1] - len(NUMERIC)} one-hot (k-1 per nominal variable)")
    assert not np.isnan(Xt_train).any() and not np.isnan(Xt_test).any()
    print("no missing values after preprocessing")
    print("\nTASKS 4-5 MATCH REFERENCE")


if __name__ == "__main__":
    main()
