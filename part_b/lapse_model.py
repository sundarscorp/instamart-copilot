"""Part B, Tasks 4-8 -- lapse prediction without leakage.

Run from the repository root:  python part_b/lapse_model.py
Cutoff 2025-09-01: features use ONLY orders dated before the cutoff (plus the clean customer
table); the label is whether the customer has a completed order in 2025-09-01..2025-09-30.
Segment labels from segments.py are NOT used as features (they include September).
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLEAN, OUT = os.path.join("data", "clean"), "outputs"
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

    # ---------------- Task 6: tune on train, then evaluate once on test ----------------
    print("\n## Task 6: tuning (5-fold CV on training rows, scoring = F1 of the lapsed class)\n")
    print(f"baseline on test: accuracy {acc:.1%}, precision 0.00, recall 0.00, F1 0.00")
    searches = {
        "logistic regression": GridSearchCV(
            Pipeline([("pre", make_preprocessor()), ("model", LogisticRegression(max_iter=1000))]),
            {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1"),
        "random forest": GridSearchCV(
            Pipeline([("pre", make_preprocessor()),
                      ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
            {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
            cv=5, scoring="f1"),
    }
    for name, gs in searches.items():
        gs.fit(X_train, y_train)                            # tuning sees TRAIN ROWS ONLY
        n_fits = len(gs.cv_results_["params"]) * gs.n_splits_
        best = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        print(f"{name}: best {best}, best CV F1 {gs.best_score_:.3f} ({n_fits} fits)")

    # The logistic-regression grid is nearly flat: show every C so the tie is visible.
    cv = searches["logistic regression"].cv_results_
    print("\nlogistic regression CV F1 by C (mean +/- std over 5 folds):")
    for c, mu, sd in zip(cv["param_model__C"], cv["mean_test_score"], cv["std_test_score"]):
        print(f"  C={c:<6} {mu:.3f} +/- {sd:.3f}")

    print("\n### Test-set evaluation (lapsed = positive class)\n")
    print("| model | TN | FP | FN | TP | precision | recall | F1 |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    test_f1 = {}
    for name, gs in searches.items():
        pred = gs.predict(X_test)                           # first and only look at the test set
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        p = precision_score(y_test, pred, zero_division=0)
        r = recall_score(y_test, pred, zero_division=0)
        test_f1[name] = f1_score(y_test, pred, zero_division=0)
        print(f"| {name} | {tn} | {fp} | {fn} | {tp} | {p:.3f} | {r:.3f} | {test_f1[name]:.3f} |")

    # Reference: LR best C=1 (test F1 ~0.62), RF depth 8 / leaf 10 (test F1 ~0.65). The brief
    # allows different numbers "of this order"; our LR grid is a statistical tie (see above).
    rf = searches["random forest"]
    assert rf.best_params_ == {"model__max_depth": 8, "model__min_samples_leaf": 10}
    assert all(0.55 <= v <= 0.75 for v in test_f1.values()), test_f1
    print("\nTASKS 4-6 CHECKED (both grids run; test F1 of the reference order)")

    # ---------------- Task 7: over-sample the minority class (TRAINING rows only) ----------------
    print("\n## Task 7: class-balance experiment\n")
    train = X_train.assign(lapsed=y_train)
    minority, majority = train[train["lapsed"] == 1], train[train["lapsed"] == 0]
    extra = minority.sample(n=len(majority) - len(minority), replace=True, random_state=42)
    balanced = pd.concat([train, extra])                    # every original row + duplicates
    print(f"training rows before: {len(majority)} not lapsed / {len(minority)} lapsed; "
          f"after: {(balanced['lapsed'] == 0).sum()} / {(balanced['lapsed'] == 1).sum()} "
          f"({len(extra)} duplicated lapsed rows added)")
    assert (balanced["lapsed"] == 0).sum() == (balanced["lapsed"] == 1).sum() == 570

    best_lr = searches["logistic regression"].best_estimator_
    balanced_lr = clone(best_lr).fit(balanced.drop(columns="lapsed"), balanced["lapsed"])   # same settings, new data
    print("\n| logistic regression | precision | recall | F1 |\n|---|---:|---:|---:|")
    rows = {}
    for label, model in [("un-balanced training", best_lr), ("balanced training", balanced_lr)]:
        pred = model.predict(X_test)                        # same untouched test set
        rows[label] = (precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred))
        print(f"| {label} | {rows[label][0]:.3f} | {rows[label][1]:.3f} | {rows[label][2]:.3f} |")
    (p0, r0, _), (p1, r1, _) = rows["un-balanced training"], rows["balanced training"]
    assert r1 > r0 + 0.1 and p1 <= p0 + 0.02            # recall jumps; precision does not rise
    for label, model in [("un-balanced", best_lr), ("balanced", balanced_lr)]:
        tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
        print(f"{label}: flags {tp + fp} customers -> {tp} real lapsers caught, {fp} loyal customers targeted")

    # ---------------- Task 8: interpret, persist and use ----------------
    print("\n## Task 8: interpretation, persistence, retention list\n")
    rf_best = searches["random forest"].best_estimator_
    names = [n.split("__", 1)[1] for n in rf_best.named_steps["pre"].get_feature_names_out()]
    importances = pd.Series(rf_best.named_steps["model"].feature_importances_, index=names)
    print("random forest: top 5 feature importances")
    for feat, imp in importances.sort_values(ascending=False).head(5).items():
        print(f"  {feat:<24} {imp:.3f}")

    coefs = pd.Series(best_lr.named_steps["model"].coef_[0], index=names)
    print("\nlogistic regression coefficients (standardised features; + means MORE likely to lapse)")
    for feat in ["recency", "frequency", "cancel_rate", "monetary", "tenure_days"]:
        print(f"  {feat:<24} {coefs[feat]:+.3f}")

    # (b) persist the SELECTED model (random forest pipeline: preprocessing + classifier)
    path = os.path.join(OUT, "lapse_model.joblib")
    os.makedirs(OUT, exist_ok=True)
    joblib.dump(rf_best, path)
    reloaded = joblib.load(path)
    same = (reloaded.predict(X_test) == rf_best.predict(X_test)).all()
    print(f"\nsaved {path}; reloaded model reproduces all {len(X_test)} test predictions exactly: {same}")
    assert same

    # (c) Premium customers in the TEST set, ranked by predicted lapse probability
    segments = pd.read_csv(os.path.join(OUT, "customer_segments.csv")).set_index("customer_id")
    premium_test = [cid for cid in X_test.index if bool(segments.loc[cid, "is_premium"])]
    proba = pd.Series(reloaded.predict_proba(X_test.loc[premium_test])[:, 1], index=premium_test)
    top = (pd.DataFrame({"customer_id": proba.index, "lapse_probability": proba.round(4).values,
                         "actually_lapsed": y_test.loc[proba.index].values})
           .sort_values("lapse_probability", ascending=False).head(10))
    top.to_csv(os.path.join(OUT, "retention_list.csv"), index=False)
    print(f"\n{len(premium_test)} Premium customers in the test set; top {len(top)} by lapse probability:")
    print(top.to_string(index=False))
    print(f"\n{int(top['actually_lapsed'].sum())} of {len(top)} truly lapsed. Note: the Premium flag comes from "
          "whole-window RFM (including September), so this list is illustrative, not a leakage-free evaluation.")
    print("wrote outputs/retention_list.csv")
    print("\nPART B LAPSE MODEL COMPLETE")


if __name__ == "__main__":
    main()
