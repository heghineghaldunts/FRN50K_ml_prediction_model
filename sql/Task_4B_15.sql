-- Question 15: Which 5 stores have the worst stockout rates?

SELECT
    store_id,
    city_id,
    COUNT(*) AS total_hours,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END) AS stockout_hours,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END)::NUMERIC
        / COUNT(*) * 100.0 AS stockout_rate
FROM (
    SELECT store_id,
           city_id,
           unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::NUMERIC AS hour_flag
    FROM public.raw_sales_data
) AS expanded
GROUP BY store_id, city_id
HAVING COUNT(*) > 100
ORDER BY stockout_rate DESC
LIMIT 5;

