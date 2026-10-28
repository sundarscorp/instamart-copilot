# Instamart Customer Intelligence & Support Copilot

## Overview

_TODO: one paragraph on what the Copilot does and how Parts A to D connect._

## Reproduce

```bash
1. python -m venv .venv
2. .venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
   # Windows PowerShell: if scripts are blocked, first run
   #   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
3. pip install -r requirements.txt
4. python generate_data.py           # expect: generated: customers=1200 products=40 orders_raw=13084 tickets=164
5. python part_a/audit.py            # raw-data audit; expect: AUDIT MATCHES REFERENCE
6. python part_a/clean.py            # writes data/clean/customers.csv, products.csv, orders.csv
7. python part_a/store.py            # builds data/instamart.db from sql/schema.sql
8. python part_a/validate.py         # expect: ALL CHECKPOINTS PASSED
9. python part_a/sql_analysis.py     # runs sql/queries.sql Q1-Q7; expect: ALL QUERIES MATCH REFERENCE
# 10. ... add each Part's script here as you build it
```

## Part A

_TODO: audit table, cleaning rules with rows changed, SQL results, pandas cross-check, EDA answers._

## Part B

## Part C

## Part D

## Limitations and Next Steps
