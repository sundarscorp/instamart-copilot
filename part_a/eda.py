"""Part A, Task 7 -- pandas cross-check of Q2 and exploratory statistics.

Run from the repository root:  python part_a/eda.py
(a) rebuilds Q2 with pd.pivot_table(margins=True) and checks its grand total against SQL;
    per-city sum / mean / count with groupby + named aggregation.
(b) distribution, outliers (1.5 x IQR) and correlations of completed order_amount.
Every figure is checked against the brief's reference value. Amounts are in INR.
"""
import os
import sqlite3
import pandas as pd

CLEAN = os.path.join("data", "clean")
DB = os.path.join("data", "instamart.db")
results = []   # (label, ok) for the final summary


def check(label, ok):
    results.append((label, bool(ok)))
    print(f"- [{'PASS' if ok else 'FAIL'}] {label}")


def main():
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
    customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
    products = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    done = (orders[orders["status"] == "completed"]
            .merge(customers[["customer_id", "city"]], on="customer_id")
            .merge(products[["product_id", "category"]], on="product_id"))

    # ---------------- (a) cross-check Q2 ----------------
    print("## (a) pandas cross-check of Q2\n")
    pivot = pd.pivot_table(done, values="order_amount", index="city", columns="category",
                           aggfunc="sum", margins=True, margins_name="Total").round(2)
    print(pivot.to_string(), "\n")
    with sqlite3.connect(DB) as con:
        sql_total = con.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders "
                                "WHERE status = 'completed'").fetchone()[0]
    grand = round(pivot.loc["Total", "Total"], 2)
    print(f"pivot grand total {grand:,.2f} vs SQL total {sql_total:,.2f}")
    check("pivot grand total equals SQL total (2,998,668.80)", grand == sql_total == 2998668.80)

    by_city = done.groupby("city").agg(total=("order_amount", "sum"),
                                       mean=("order_amount", "mean"),
                                       orders=("order_amount", "count")).round(2)
    print("\n| city | sum | mean | count |\n|---|---:|---:|---:|")
    for city, r in by_city.iterrows():
        print(f"| {city} | {r['total']:,.2f} | {r['mean']:,.2f} | {int(r['orders']):,} |")
    check("Bengaluru 523,643.40 / 276.47 / 1,894",
          tuple(by_city.loc["Bengaluru"]) == (523643.40, 276.47, 1894))
    check("Mumbai 479,058.50 / 297.00 / 1,613",
          tuple(by_city.loc["Mumbai"]) == (479058.50, 297.00, 1613))

    # ---------------- (b) distribution and outliers ----------------
    print("\n## (b) completed order_amount: distribution and outliers\n")
    amt = done["order_amount"]
    q1, q3 = amt.quantile(0.25), amt.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    stats = {"mean": amt.mean(), "median": amt.median(), "skewness": amt.skew(),
             "Q1 (25%)": q1, "Q3 (75%)": q3, "IQR": iqr,
             "lower fence (Q1 - 1.5 IQR)": lower, "upper fence (Q3 + 1.5 IQR)": upper,
             "outliers below lower fence": (amt < lower).sum(),
             "outliers above upper fence": (amt > upper).sum()}
    print("| statistic | value |\n|---|---:|")
    for k, v in stats.items():
        print(f"| {k} | {v:,.2f} |" if isinstance(v, float) else f"| {k} | {v:,} |")
    print()
    check("mean 269.91, median 135.00", (round(stats["mean"], 2), stats["median"]) == (269.91, 135.00))
    check("skewness about 4.61", abs(stats["skewness"] - 4.61) < 0.01)
    check("Q1 64.80, Q3 297.00", (round(q1, 2), round(q3, 2)) == (64.80, 297.00))
    check("upper fence 645.30 with 897 outliers above, none below",
          (round(upper, 2), stats["outliers above upper fence"], stats["outliers below lower fence"]) == (645.30, 897, 0))

    # ---------------- (b) correlations ----------------
    print("\n## (b) correlation matrix (completed orders)\n")
    cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr = done[cols].corr().round(3)
    print(corr.to_string(), "\n")
    pairs = corr.where(~pd.DataFrame(
        [[i <= j for j in range(len(cols))] for i in range(len(cols))], index=cols, columns=cols)).stack()
    strongest = pairs.abs().idxmax()
    print(f"largest |correlation|: {strongest[0]} vs {strongest[1]} = {pairs[strongest]:+.3f}")
    check("largest |r| is order_amount vs discount_pct, about -0.095",
          set(strongest) == {"order_amount", "discount_pct"} and abs(pairs[strongest] + 0.095) < 0.005)
    check("order_amount vs quantity about +0.087",
          abs(corr.loc["order_amount", "quantity"] - 0.087) < 0.005)

    failed = [label for label, ok in results if not ok]
    print("\nEDA MATCHES REFERENCE" if not failed else f"\nMISMATCHES: {failed}")


if __name__ == "__main__":
    main()
