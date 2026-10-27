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
