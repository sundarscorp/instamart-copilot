"""Part A, Task 6 -- run the seven queries in sql/queries.sql against data/instamart.db.

Run from the repository root:  python part_a/sql_analysis.py
Each query is run with pd.read_sql_query, checked against the brief's reference result,
and summarised as a markdown table (paste-ready for README.md). Amounts are in INR.
"""
import os
import re
import sqlite3
import pandas as pd

DB = os.path.join("data", "instamart.db")
QUERIES = os.path.join("sql", "queries.sql")


def load_queries(path):
    """Split queries.sql at label lines of the form '-- Q1: ...' -> {'Q1': (title, sql)}."""
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- (Q\d+): (.*)$", text, flags=re.M)
    # parts = [header, 'Q1', title1, sql1, 'Q2', title2, sql2, ...]
    return {parts[i]: (parts[i + 1].strip(), parts[i + 2].strip())
            for i in range(1, len(parts), 3)}


def md(df, max_rows=None):
    """Render a DataFrame as a markdown table (no extra library needed)."""
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    fmt = lambda v: f"{v:,.2f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def check_q1(r):
    return [("30 customers", len(r) == 30)], md(r, 5) + f"\n\n(top 5 of {len(r)} rows)"


def check_q2(r):
    top = r.iloc[0]
    checks = [("30 rows", len(r) == 30),
              ("revenues add up to 2,998,668.80", round(r["revenue"].sum(), 2) == 2998668.80),
              ("largest cell Bengaluru / Electronics 233,909.45",
               (top["city"], top["category"], top["revenue"]) == ("Bengaluru", "Electronics", 233909.45))]
    return checks, md(r, 5) + f"\n\n(top 5 of {len(r)} rows; total {r['revenue'].sum():,.2f})"


def check_q3(r):
    return [("11 products", len(r) == 11)], md(r)


def check_q4(r):
    return [("25 customers", len(r) == 25)], f"{len(r)} customers: " + ", ".join(r["customer_id"])


def check_q5(r):
    avg = r["avg_spend"].iloc[0]
    checks = [("550 customers", len(r) == 550), ("average 2,552.06", avg == 2552.06)]
    return checks, f"{len(r)} customers above the average customer spend of INR {avg:,.2f}"


def check_q6(r):
    leaders = r[r["row_num"] == 1][["city", "customer_id", "total_spend"]].reset_index(drop=True)
    expected = {"Bengaluru": ("C0082", 12856.20), "Chennai": ("C0930", 10151.40),
                "Delhi": ("C0291", 13742.60), "Hyderabad": ("C0743", 12730.05),
                "Mumbai": ("C0693", 14766.55), "Pune": ("C0426", 10377.10)}
    got = {row.city: (row.customer_id, row.total_spend) for row in leaders.itertuples()}
    ties = r[r["rnk"] != r["row_num"]]
    checks = [("top customer per city", got == expected),
              ("exactly 8 rows with RANK() != ROW_NUMBER()", len(ties) == 8)]
    report = (md(leaders) + f"\n\nRows where RANK() differs from ROW_NUMBER(): {len(ties)}\n\n"
              + md(ties[["city", "customer_id", "total_spend", "rnk", "dense_rnk", "row_num"]]))
    return checks, report


def check_q7(r):
    last = r["cumulative_revenue"].iloc[-1]
    checks = [("183 daily rows", len(r) == 183), ("last cumulative 2,998,668.80", last == 2998668.80)]
    return checks, md(pd.concat([r.head(3), r.tail(3)])) + f"\n\n(first and last 3 of {len(r)} rows)"


CHECKS = {"Q1": check_q1, "Q2": check_q2, "Q3": check_q3, "Q4": check_q4,
          "Q5": check_q5, "Q6": check_q6, "Q7": check_q7}


def main():
    queries = load_queries(QUERIES)
    assert set(queries) == set(CHECKS), f"expected Q1-Q7 in {QUERIES}, found {sorted(queries)}"
    failed = []
    with sqlite3.connect(DB) as con:
        for name, (title, sql) in queries.items():
            result = pd.read_sql_query(sql, con)
            checks, report = CHECKS[name](result)
            print(f"\n### {name}: {title}\n\n{report}\n")
            for label, ok in checks:
                print(f"- [{'PASS' if ok else 'FAIL'}] {label}")
                if not ok:
                    failed.append(f"{name}: {label}")
    print("\nALL QUERIES MATCH REFERENCE" if not failed else f"\nMISMATCHES: {failed}")


if __name__ == "__main__":
    main()
