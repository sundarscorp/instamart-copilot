"""Part A, Task 2 -- audit the raw exports.

Run from the repository root:  python part_a/audit.py
Prints markdown tables (paste-ready for README.md):
  1. a profile of each raw table: shape, dtype and missing values per column
  2. the specific audit figures, each checked against the brief's reference value

Reading rule used everywhere: ONLY an empty cell counts as missing.
pandas' default would also turn text such as "N/A" into NaN, which hides
53 of the 120 unparseable order amounts -- so default NA handling is switched off.
"""
import os
import pandas as pd

RAW = os.path.join("data", "raw")


def load(name):
    """Read a raw CSV where only truly empty cells become missing."""
    return pd.read_csv(os.path.join(RAW, name), keep_default_na=False, na_values=[""])


def profile(df, name):
    """Print shape, then dtype / missing count / missing % for every column."""
    print(f"\n### {name}  (shape: {df.shape[0]:,} rows x {df.shape[1]} columns)\n")
    print("| column | dtype | missing | missing % |")
    print("|---|---|---:|---:|")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        print(f"| {col} | {df[col].dtype} | {n_miss:,} | {n_miss / len(df):.2%} |")


def main():
    orders = load("orders.csv")
    customers = load("customers.csv")
    products = load("products.csv")
    # tickets are not audited here, but the brief's rule for reading them later is:
    # pd.read_csv(..., keep_default_na=False)  so empty ticket texts stay "".

    print("## Raw table profiles")
    for df, name in [(orders, "orders.csv"), (customers, "customers.csv"), (products, "products.csv")]:
        profile(df, name)

    # ---- order figures ----
    amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    unparseable = amount_num.isna() & orders["order_amount"].notna()   # present but not a number
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")

    # ---- customer figures ----
    age = pd.to_numeric(customers["age"], errors="coerce")

    # (label, measured value, reference value from the brief)
    checks = [
        ("orders.csv rows", len(orders), 13084),
        ("exact duplicate rows", int(orders.duplicated().sum()), 119),
        ("rows with status = pending", int((orders["status"] == "pending").sum()), 656),
        ("distinct order_id", orders["order_id"].nunique(), 12309),
        ("non-numeric order_amount", int(unparseable.sum()), 120),
        ("negative delivery_minutes", int((delivery < 0).sum()), 106),
        ("blank delivery_minutes", int(orders["delivery_minutes"].isna().sum()), 2541),
        ("blank age", int(customers["age"].isna().sum()), 61),
        ("age outside 10-100", int(((age < 10) | (age > 100)).sum()), 6),
        ("distinct raw city spellings", customers["city"].nunique(), 24),
        ("distinct non-blank gender spellings", customers["gender"].nunique(), 10),  # nunique ignores NaN
        ("blank gender", int(customers["gender"].isna().sum()), 30),
    ]

    print("\n## Audit figures\n")
    print("| check | measured | reference | result |")
    print("|---|---:|---:|---|")
    for label, got, ref in checks:
        print(f"| {label} | {got:,} | {ref:,} | {'PASS' if got == ref else 'FAIL'} |")

    # Show WHAT the unparseable amounts are -- useful evidence for the cleaning rules.
    print("\nUnparseable order_amount values:",
          orders.loc[unparseable, "order_amount"].value_counts().to_dict())

    failed = [label for label, got, ref in checks if got != ref]
    print("\nAUDIT MATCHES REFERENCE" if not failed else f"\nMISMATCHES: {failed}")


if __name__ == "__main__":
    main()
