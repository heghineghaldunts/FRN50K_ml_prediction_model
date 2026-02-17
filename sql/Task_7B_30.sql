-- Question 30: Create a "store health score" combining sales performance and operational efficiency.

WITH ranked_stores AS (
    SELECT
        store_id,
        city_id,
        total_sales,
        stockout_rate::NUMERIC AS stockout_rate,
        sales_volatility::NUMERIC AS sales_volatility,
        RANK() OVER (ORDER BY total_sales DESC) AS sales_rank
    FROM store_performance_dashboard_new
)
SELECT
    store_id,
    city_id,
    sales_rank,
    stockout_rate,
    sales_volatility,
    
    (1.0 / sales_rank) * 0.5 +
    (1 - stockout_rate) * 0.3 +
    (1 / (1 + sales_volatility)) * 0.2 AS store_health_score
FROM ranked_stores
ORDER BY store_health_score DESC;
