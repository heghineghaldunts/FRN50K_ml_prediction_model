-- Question 5: What is the average sales per hour?

SELECT
    AVG(hourly_sales) AS avg_sales_per_hour
FROM (
    SELECT
        dt,
        hours_sale,
        SUM(sale_amount) AS hourly_sales
    FROM public.raw_sales_data
    GROUP BY dt, hours_sale
) t;