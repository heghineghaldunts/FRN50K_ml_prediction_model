-- Question 27: How do different weather conditions affect sales by product category?

SELECT
    category_id,
    temp_range,
    AVG(sale_amount) AS avg_sales
FROM (
    SELECT
        sale_amount,
        avg_temperature,
        CASE
            WHEN avg_temperature < 0 THEN 'Below 0°C'
            WHEN avg_temperature BETWEEN 0 AND 10 THEN '0–10°C'
            WHEN avg_temperature BETWEEN 11 AND 20 THEN '11–20°C'
            ELSE '21°C+'
        END AS temp_range,
        first_category_id AS category_id
    FROM public.raw_sales_data
) t
WHERE category_id IS NOT NULL
GROUP BY category_id, temp_range
ORDER BY category_id, temp_range;
