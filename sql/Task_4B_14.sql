-- Question 14: Do stockouts occur more at certain hours? Show stockout rate by hour

SELECT
    hour_of_day,
    COUNT(*) AS total_hours,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END) AS stockout_hours,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END)::NUMERIC
        / COUNT(*) * 100.0 AS stockout_rate
FROM (
    SELECT generate_series(0, array_length(string_to_array(trim(both '[]' from hours_stock_status), ' '),1)-1) AS hour_of_day,
           unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::NUMERIC AS hour_flag
    FROM public.raw_sales_data
) AS expanded
GROUP BY hour_of_day
ORDER BY hour_of_day;

