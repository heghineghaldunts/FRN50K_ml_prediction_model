-- Question 18: Using your store_performance view, identify stores with high sales but a

SELECT
    store_id,
    city_id,
    total_sales,
    stockout_rate
FROM store_performance_dashboard_new
WHERE total_sales > (
        SELECT AVG(total_sales)
        FROM store_performance_dashboard
    )
  AND stockout_rate > 0.2
ORDER BY total_sales DESC;
