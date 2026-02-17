-- Question 22: What discount ranges (0%, 1-10%, 11-20%, 21%+) perform best?

SELECT
    CASE
        WHEN discount = 0 THEN '0%'
        WHEN discount BETWEEN 1 AND 10 THEN '1–10%'
        WHEN discount BETWEEN 11 AND 20 THEN '11–20%'
        ELSE '21%+'
    END AS discount_range,
    COUNT(*) AS total_hours,
    AVG(sale_amount) AS avg_sales,
    SUM(sale_amount) AS total_sales
FROM public.raw_sales_data
GROUP BY
    CASE
        WHEN discount = 0 THEN '0%'
        WHEN discount BETWEEN 1 AND 10 THEN '1–10%'
        WHEN discount BETWEEN 11 AND 20 THEN '11–20%'
        ELSE '21%+'
    END
ORDER BY avg_sales DESC;
