CREATE VIEW business_rhythm_patterns_new as

WITH expanded AS (
    SELECT *,
           unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::NUMERIC AS hour_flag,
           generate_series(0, array_length(string_to_array(trim(both '[]' from hours_stock_status), ' '),1)-1) AS hour_of_day
    FROM public.raw_sales_data
)
SELECT
    hour_of_day,
    EXTRACT(DOW FROM dt::DATE) AS day_of_week,      -- 0=Sunday, 6=Saturday
    EXTRACT(MONTH FROM dt::DATE) AS month,
    CASE WHEN EXTRACT(DOW FROM dt::DATE) IN (0,6) THEN 1 ELSE 0 END AS is_weekend,
    COUNT(*) AS total_hours,
    SUM(sale_amount) AS total_sales,
    AVG(sale_amount) AS avg_sales_per_hour,
    SUM(CASE WHEN sale_amount > 0 THEN 1 ELSE 0 END) / COUNT(*) * 1.0 AS activity_rate,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END) AS stockout_hours,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END) / COUNT(*) * 1.0 AS stockout_rate,
    SUM(sale_amount) / COUNT(*) AS sales_load_index,
    SUM(CASE WHEN sale_amount > 0 THEN 1 ELSE 0 END) AS busy_hours,
    AVG(precpt) AS avg_precipitation,
    AVG(avg_temperature) AS avg_temperature,
    AVG(CASE WHEN holiday_flag = 1 THEN 1 ELSE 0 END) AS holiday_frequency,
    AVG(CASE WHEN discount > 0 THEN 1 ELSE 0 END) AS promo_frequency,
    CASE
        WHEN AVG(sale_amount) > 1000 AND SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END) / COUNT(*) > 0.2 THEN 'High Pressure'
        WHEN AVG(sale_amount) > 1000 THEN 'High Demand'
        WHEN SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END) / COUNT(*) > 0.2 THEN 'Inventory Risk'
        ELSE 'Normal'
    END AS operational_mode
FROM expanded
GROUP BY
    hour_of_day,
    EXTRACT(DOW FROM dt::DATE),
    EXTRACT(MONTH FROM dt::DATE),
    CASE WHEN EXTRACT(DOW FROM dt::DATE) IN (0,6) THEN 1 ELSE 0 END;

-- Test your view: Write a query that identifies the time periods requiring
-- the highest operational intensity and support.

SELECT
    hour_of_day,
    day_of_week,
    month,
    avg_sales_per_hour,
    stockout_rate,
    operational_mode
FROM business_rhythm_patterns_new
WHERE operational_mode = 'High Pressure'
ORDER BY avg_sales_per_hour DESC;
