-- Question 24: How do sales compare when activity_flag = 1 vs activity_flag = 0?

SELECT
    activity_flag,
    COUNT(*) AS total_hours,
    AVG(sale_amount) AS avg_sales,
    SUM(sale_amount) AS total_sales
FROM public.raw_sales_data
GROUP BY activity_flag;
