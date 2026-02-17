-- Question 9: Compare average sales between holiday and non-holiday periods.

SELECT
    holiday_flag,
    AVG(hourly_sales) AS avg_hourly_sales
FROM (
    SELECT
        dt,
        hours_sale,
        holiday_flag,
        SUM(sale_amount) AS hourly_sales
    FROM public.raw_sales_data
    GROUP BY dt, hours_sale, holiday_flag
) t
GROUP BY holiday_flag;

