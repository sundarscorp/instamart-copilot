"""Part A, Tasks 3-4 -- clean the raw exports and write data/clean/*.csv.

Run from the repository root:  python part_a/clean.py
Every rule prints how many values it changed, so the README can quote the numbers.
"""
import os
import pandas as pd

RAW, CLEAN = os.path.join("data", "raw"), os.path.join("data", "clean")
GENDER_MAP = {"male": "male", "m": "male", "female": "female", "f": "female"}


def load(name):
    """Same reading rule as audit.py: only empty cells count as missing."""
    return pd.read_csv(os.path.join(RAW, name), keep_default_na=False, na_values=[""])


def clean_customers(c):
    c = c.copy()
    print("\n## Cleaning customers")

    # (a) city: strip spaces + title case -> exactly six cities.
    #     Done FIRST, because step (c) groups by city.
    before = c["city"].copy()
    c["city"] = c["city"].str.strip().str.title()
    print(f"(a) city standardised: {int((before != c['city']).sum())} values changed, "
          f"{before.nunique()} spellings -> {c['city'].nunique()} cities")

    # (b) gender: every spelling -> male / female; blank or anything else -> unknown.
    before = c["gender"].copy()
    c["gender"] = c["gender"].str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    print(f"(b) gender mapped: {int((before != c['gender']).sum())} values changed "
          f"(including {int(before.isna().sum())} blanks -> unknown)")

    # (c) age: impossible values (outside 10-100) become missing, then every missing
    #     age is filled with the median age of the customer's (already clean) city.
    c["age"] = pd.to_numeric(c["age"], errors="coerce")
    impossible = ~c["age"].between(10, 100) & c["age"].notna()
    c.loc[impossible, "age"] = pd.NA
    n_missing = int(c["age"].isna().sum())
    city_median = c.groupby("city")["age"].transform("median")
    c["age"] = c["age"].fillna(city_median)
    print(f"(c) age: {int(impossible.sum())} impossible values set to missing; "
          f"{n_missing} missing ages filled with the city median")

    # (d) signup_date -> datetime.
    c["signup_date"] = pd.to_datetime(c["signup_date"], format="%Y-%m-%d")
    print("(d) signup_date converted to datetime")
    return c


def check_customers(c):
    """Assert the brief's reference values for cleaned customers."""
    assert c["city"].value_counts().to_dict() == {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
    assert c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}
    assert c["age"].isna().sum() == 0
    assert c["age"].between(18, 60).all()
    print("customers match the reference values")


def clean_orders(o, products):
    o = o.copy()
    print("\n## Cleaning orders")
    n0 = len(o)

    # 1. order_ts -> datetime
    o["order_ts"] = pd.to_datetime(o["order_ts"], format="%Y-%m-%d %H:%M:%S")
    print(f"1. order_ts converted to datetime ({n0:,} rows)")

    # 2. remove exact duplicate rows
    o = o.drop_duplicates()
    print(f"2. exact duplicates removed: {n0 - len(o)} rows -> {len(o):,} rows")

    # 3. one row per order_id: keep the record with the LATEST order_ts
    #    (the earlier 'pending' record is superseded by the final status).
    n1 = len(o)
    o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last")
    print(f"3. superseded records removed (kept latest per order_id): {n1 - len(o)} rows -> {len(o):,} rows")

    # 4. order_amount -> numeric; unparseable ("N/A", "error") -> missing,
    #    then rebuilt from the catalogue price.
    o = o.merge(products[["product_id", "price_inr", "category"]], on="product_id", how="left")
    o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
    missing_amt = o["order_amount"].isna()
    rebuilt = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
    o.loc[missing_amt, "order_amount"] = rebuilt[missing_amt]
    print(f"4. order_amount: {int(missing_amt.sum())} unparseable amounts rebuilt from price x quantity x discount")

    # 5. delivery_minutes: negatives -> missing; completed orders filled with the
    #    median of their product category (computed over completed orders only);
    #    cancelled orders stay missing, because they were never delivered.
    o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
    negative = o["delivery_minutes"] < 0
    o.loc[negative, "delivery_minutes"] = pd.NA
    completed = o["status"] == "completed"
    cat_median = o[completed].groupby("category")["delivery_minutes"].transform("median")
    to_fill = completed & o["delivery_minutes"].isna()
    o.loc[to_fill, "delivery_minutes"] = cat_median[to_fill]
    print(f"5. delivery_minutes: {int(negative.sum())} negative values set to missing; "
          f"{int(to_fill.sum())} completed orders filled with the category median; "
          f"{int((~completed).sum())} cancelled orders left missing")
    print("   category medians used:",
          o[completed].groupby("category")["delivery_minutes"].median().to_dict())

    cols = ["order_id", "customer_id", "product_id", "quantity", "order_ts", "status",
            "discount_pct", "order_amount", "delivery_minutes"]
    return o[cols].sort_values("order_id").reset_index(drop=True)


def check_orders(o):
    """Assert the brief's reference values for cleaned orders."""
    completed = o["status"] == "completed"
    assert len(o) == 12309 and o["order_id"].is_unique
    assert o["status"].value_counts().to_dict() == {"completed": 11110, "cancelled": 1199}
    assert o["order_amount"].isna().sum() == 0
    assert o.loc[completed, "delivery_minutes"].isna().sum() == 0
    assert o.loc[~completed, "delivery_minutes"].isna().sum() == 1199
    assert round(o.loc[completed, "order_amount"].sum(), 2) == 2998668.80
    print("orders match the reference values")


def main():
    os.makedirs(CLEAN, exist_ok=True)
    customers = clean_customers(load("customers.csv"))
    check_customers(customers)
    print("\ncity medians used for filling:",
          customers.groupby("city")["age"].median().to_dict())

    products = load("products.csv")
    orders = clean_orders(load("orders.csv"), products)
    check_orders(orders)

    customers.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
    products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
    orders.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)
    print("\nwrote data/clean/customers.csv, products.csv, orders.csv")


if __name__ == "__main__":
    main()
