-- Question 10: Do rainy days (precipitation > 0) affect sales compared to clear days?

SELECT
    CASE
        WHEN precpt > 0 THEN 'Rainy'
        ELSE 'Clear'
    END AS weather_type,
    AVG(sale_amount) AS avg_sales
FROM public.raw_sales_data
GROUP BY
    CASE
        WHEN precpt > 0 THEN 'Rainy'
        ELSE 'Clear'
    END;
