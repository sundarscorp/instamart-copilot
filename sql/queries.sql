-- queries.sql -- Part A, Task 6: business questions over data/instamart.db
-- Each query starts with a label line "-- Qn: ..." so part_a/sql_analysis.py can run them one by one.
-- Unless stated otherwise, only completed orders count as revenue or spend. Amounts are in INR.

-- Q1: WHERE then HAVING -- customers whose completed orders over INR 500 total more than INR 8000
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total DESC;

-- Q2: three-table join -- completed revenue by city and product category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: scalar subquery -- products priced above the average catalogue price
SELECT product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: LEFT JOIN -- customers who have no completed order
-- (the status filter sits in ON, not WHERE, so customers with only cancelled orders still appear)
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.status IS NULL
ORDER BY c.customer_id;

-- Q5: common table expressions -- customers whose total completed spend is above the average customer total
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders o
    WHERE o.status = 'completed'
    GROUP BY customer_id
),
avg_total AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_totals
)
SELECT ct.customer_id,
       ROUND(ct.total_spend, 2) AS total_spend,
       ROUND(a.avg_spend, 2)    AS avg_spend
FROM customer_totals ct, avg_total a
WHERE ct.total_spend > a.avg_spend
ORDER BY ct.total_spend DESC;
-- Q6: ranking window functions -- each customer's completed spend ranked within their city
WITH customer_totals AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city,
       ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM customer_totals
ORDER BY city, row_num;
-- Q7: running total -- daily completed revenue with a cumulative revenue column
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day,
           SUM(order_amount) AS revenue
    FROM orders
    WHERE orders.status = 'completed'
    GROUP BY day
)
SELECT day,
       ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
