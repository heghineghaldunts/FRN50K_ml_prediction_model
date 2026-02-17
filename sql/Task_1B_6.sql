-- Question 6: How many hours had zero sales vs non-zero sales?

SELECT
    SUM(CASE WHEN hourly_sales = 0 THEN 1 ELSE 0 END) AS zero_sales_hours,
    SUM(CASE WHEN hourly_sales > 0 THEN 1 ELSE 0 END) AS non_zero_sales_hours
FROM (
    SELECT
        dt,
        hours_sale,
        SUM(sale_amount) AS hourly_sales
    FROM public.raw_sales_data
    GROUP BY dt, hours_sale
) t;