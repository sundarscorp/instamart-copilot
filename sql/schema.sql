-- schema.sql -- tables of data/instamart.db (Part A, Task 5)
-- Dates and timestamps are stored as TEXT: 'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM:SS'.
-- Amounts are in INR.

DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;

CREATE TABLE customers (
    customer_id  TEXT PRIMARY KEY,
    city         TEXT NOT NULL,
    signup_date  TEXT NOT NULL,
    age          REAL NOT NULL CHECK (age BETWEEN 10 AND 100),
    gender       TEXT NOT NULL CHECK (gender IN ('male', 'female', 'unknown'))
);

CREATE TABLE products (
    product_id    TEXT PRIMARY KEY,
    product_name  TEXT NOT NULL,
    category      TEXT NOT NULL,
    price_inr     REAL NOT NULL CHECK (price_inr > 0)
);

CREATE TABLE orders (
    order_id          TEXT PRIMARY KEY,
    customer_id       TEXT NOT NULL REFERENCES customers (customer_id),
    product_id        TEXT NOT NULL REFERENCES products (product_id),
    quantity          INTEGER NOT NULL CHECK (quantity > 0),
    order_ts          TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('completed', 'cancelled')),
    discount_pct      INTEGER NOT NULL,
    order_amount      REAL NOT NULL CHECK (order_amount >= 0),
    delivery_minutes  REAL CHECK (delivery_minutes >= 0)   -- NULL for cancelled orders
);
