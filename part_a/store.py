"""Part A, Task 5 (store) -- load data/clean/*.csv into the SQLite database data/instamart.db.

Run from the repository root:  python part_a/store.py
The tables are created from sql/schema.sql (types, primary keys, foreign keys, CHECK
constraints), then filled with DataFrame.to_sql(if_exists="append") so the schema is kept.
"""
import os
import sqlite3
import pandas as pd

CLEAN = os.path.join("data", "clean")
DB = os.path.join("data", "instamart.db")
SCHEMA = os.path.join("sql", "schema.sql")


def main():
    # Dates are read as plain text so they are stored exactly as
    # 'YYYY-MM-DD' / 'YYYY-MM-DD HH:MM:SS', as the brief requires.
    customers = pd.read_csv(os.path.join(CLEAN, "customers.csv"), dtype={"signup_date": str})
    products = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), dtype={"order_ts": str})

    with sqlite3.connect(DB) as con:
        con.execute("PRAGMA foreign_keys = ON")
        with open(SCHEMA, encoding="utf-8") as f:
            con.executescript(f.read())
        # parents first, so every foreign key already has a matching row
        for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
            df.to_sql(name, con, if_exists="append", index=False)
            print(f"loaded {name}: {len(df):,} rows")
    print(f"wrote {DB} (SQLite {sqlite3.sqlite_version})")


if __name__ == "__main__":
    main()
