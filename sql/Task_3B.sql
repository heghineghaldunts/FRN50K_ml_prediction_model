CREATE VIEW store_performance_dashboard_new AS
WITH expanded AS (
    SELECT *,
           unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::INT AS hour_flag
    FROM public.raw_sales_data
)
SELECT
    city_id,
    store_id,
    management_group_id,
    COUNT(DISTINCT dt) AS active_days,
    COUNT(*) AS total_store_hours,
    SUM(sale_amount::NUMERIC) AS total_sales,
    AVG(sale_amount::NUMERIC) AS avg_sales_per_hour,
    SUM(CASE WHEN sale_amount::NUMERIC > 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) AS sales_activity_rate,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END) AS stockout_hours,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) AS stockout_rate,
    SUM(CASE WHEN hour_flag = 0 THEN sale_amount::NUMERIC ELSE 0 END) AS sales_during_stockout,
    SUM(CASE WHEN sale_amount::NUMERIC > 0 THEN 1 ELSE 0 END) AS hours_with_sales,
    AVG(CASE WHEN discount::NUMERIC > 0 THEN sale_amount::NUMERIC END) AS avg_sales_with_discount,
    AVG(CASE WHEN discount::NUMERIC = 0 THEN sale_amount::NUMERIC END) AS avg_sales_without_discount,
    STDDEV(sale_amount::NUMERIC) AS sales_volatility
FROM expanded
GROUP BY city_id, store_id, management_group_id;

-- Test your view: Write a query using your view to identify the top 5 stores by sales performance
-- and bottom 5 stores by stockout management.

SELECT *
FROM store_performance_dashboard_new
ORDER BY total_sales DESC
LIMIT 5;

SELECT *
FROM store_performance_dashboard_new
ORDER BY stockout_rate DESC
LIMIT 5;

