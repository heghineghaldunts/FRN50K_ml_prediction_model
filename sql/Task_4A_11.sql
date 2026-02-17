-- Question 11: What percentage of total hours experienced stockouts?

SELECT
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END)::NUMERIC
    / COUNT(*) * 100.0 AS stockout_hour_percentage
FROM (
    SELECT unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::NUMERIC AS hour_flag
    FROM public.raw_sales_data
) AS expanded;
