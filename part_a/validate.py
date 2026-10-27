"""Part A, Task 5 (validate) -- check the saved outputs against the brief's reference values.

Run from the repository root:  python part_a/validate.py
Reads ONLY what is on disk (data/clean/*.csv and data/instamart.db), so it tests what was
actually saved. Prints exactly 'ALL CHECKPOINTS PASSED' when every check holds; otherwise
stops with an AssertionError naming the first failed check.
"""
import os
import re
import sqlite3
import pandas as pd

CLEAN = os.path.join("data", "clean")
DB = os.path.join("data", "instamart.db")


def main():
    cust = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
    completed = orders["status"] == "completed"
    con = sqlite3.connect(DB)
    sql = lambda q: con.execute(q).fetchone()[0]

    checks = {
        # ---- Task 3: customers ----
        "6 cities with the reference counts": cust["city"].value_counts().to_dict() == {
            "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
        "genders female 610 / male 560 / unknown 30":
            cust["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
        "0 missing ages": cust["age"].isna().sum() == 0,
        "every age between 18 and 60": cust["age"].between(18, 60).all(),
        # ---- Task 4: orders ----
        "12,309 rows, one per order_id": len(orders) == 12309 and orders["order_id"].is_unique,
        "11,110 completed and 1,199 cancelled":
            orders["status"].value_counts().to_dict() == {"completed": 11110, "cancelled": 1199},
        "no pending rows": (orders["status"] == "pending").sum() == 0,
        "0 missing order_amount": orders["order_amount"].isna().sum() == 0,
        "0 missing delivery_minutes among completed": orders.loc[completed, "delivery_minutes"].isna().sum() == 0,
        "1,199 missing delivery_minutes among cancelled": orders.loc[~completed, "delivery_minutes"].isna().sum() == 1199,
        "completed revenue INR 2,998,668.80 (CSV)": round(orders.loc[completed, "order_amount"].sum(), 2) == 2998668.80,
        # ---- Task 5: the database matches ----
        "database has tables customers, products, orders": {
            r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")} >= {
            "customers", "products", "orders"},
        "database row counts 1,200 / 40 / 12,309": (sql("SELECT COUNT(*) FROM customers"),
            sql("SELECT COUNT(*) FROM products"), sql("SELECT COUNT(*) FROM orders")) == (1200, 40, 12309),
        "completed revenue INR 2,998,668.80 (SQL)":
            sql("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status = 'completed'") == 2998668.80,
        "order_ts stored as text YYYY-MM-DD HH:MM:SS": all(
            isinstance(ts, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", ts)
            for (ts,) in con.execute("SELECT order_ts FROM orders")),
    }
    con.close()

    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        assert ok, f"checkpoint failed: {name}"
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
