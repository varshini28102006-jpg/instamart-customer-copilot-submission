-- =====================================================================
-- Instamart Customer Intelligence & Support Copilot
-- Analytical SQL Queries (SQLite 3.25+)
-- =====================================================================

-- ---------------------------------------------------------------------
-- Q1: WHERE then HAVING
-- Filters completed orders with order_amount > 500, groups by customer,
-- and retains customers whose total spend across those orders exceeds 8,000.
-- Reference: 30 customers.
-- ---------------------------------------------------------------------
-- @name: q1_high_value_customers
SELECT 
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_spend,
    COUNT(order_id) AS qualifying_order_count
FROM orders
WHERE status = 'completed' 
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- ---------------------------------------------------------------------
-- Q2: Three-Table Join
-- Aggregates completed order revenue across city and product category.
-- Reference: 30 rows (6 cities x 5 categories); sum = INR 2,998,668.80;
-- Largest cell: Bengaluru / Electronics at INR 233,909.45.
-- ---------------------------------------------------------------------
-- @name: q2_revenue_by_city_category
SELECT 
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue,
    COUNT(o.order_id) AS order_count
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city ASC, revenue DESC;


-- ---------------------------------------------------------------------
-- Q3: Scalar Subquery
-- Identifies catalogue products whose price is strictly above the overall
-- average catalogue price across all products.
-- Reference: 11 products.
-- ---------------------------------------------------------------------
-- @name: q3_above_average_products
SELECT 
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr) 
    FROM products
)
ORDER BY price_inr DESC, product_id ASC;


-- ---------------------------------------------------------------------
-- Q4: LEFT JOIN
-- Identifies registered customers who have never placed a completed order
-- (either placed only cancelled orders or placed zero orders).
-- Reference: 25 customers.
-- ---------------------------------------------------------------------
-- @name: q4_customers_no_completed_orders
SELECT 
    c.customer_id,
    c.city,
    c.signup_date,
    c.gender
FROM customers c
LEFT JOIN orders o 
    ON c.customer_id = o.customer_id 
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id ASC;


-- ---------------------------------------------------------------------
-- Q5: Common Table Expressions (CTEs)
-- First CTE calculates each customer's total completed spend;
-- Second CTE calculates the overall average completed spend across active customers;
-- Main query returns all customers with spend strictly greater than the average.
-- Reference: 550 customers; average = INR 2,552.06.
-- ---------------------------------------------------------------------
-- @name: q5_above_average_spenders
WITH customer_spend AS (
    SELECT 
        customer_id,
        ROUND(SUM(order_amount), 2) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
overall_average AS (
    SELECT 
        AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT 
    cs.customer_id,
    cs.total_spend,
    ROUND(oa.avg_spend, 2) AS benchmark_avg_spend
FROM customer_spend cs
CROSS JOIN overall_average oa
WHERE cs.total_spend > oa.avg_spend
ORDER BY cs.total_spend DESC;


-- ---------------------------------------------------------------------
-- Q6: Window Ranking Functions
-- Computes customer total spend and calculates RANK(), DENSE_RANK(), and
-- ROW_NUMBER() partitioned by city and ordered by total spend descending.
-- Reference: Top per city:
--   Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40),
--   Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05),
--   Mumbai C0693 (14,766.55), Pune C0426 (10,377.10).
-- Exactly 8 rows have RANK() != ROW_NUMBER() due to spend ties.
-- ---------------------------------------------------------------------
-- @name: q6_city_customer_rankings
WITH cust_city_spend AS (
    SELECT 
        c.city,
        c.customer_id,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT 
    city,
    customer_id,
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_dense_rank,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM cust_city_spend
ORDER BY city ASC, spend_rank ASC;


-- ---------------------------------------------------------------------
-- Q7: Running Total / Cumulative Revenue Window Function
-- Calculates daily completed revenue and cumulative running total over time.
-- Reference: 183 daily rows; final cumulative spend = INR 2,998,668.80.
-- ---------------------------------------------------------------------
-- @name: q7_daily_running_revenue
WITH daily_revenue AS (
    SELECT 
        SUBSTR(order_ts, 1, 10) AS order_date,
        ROUND(SUM(order_amount), 2) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT 
    order_date,
    daily_revenue,
    ROUND(SUM(daily_revenue) OVER (ORDER BY order_date ASC), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_date ASC;
