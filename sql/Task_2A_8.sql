-- Question 8: Show total sales by day of week. Which days generate most sales?

SELECT
    EXTRACT(DOW FROM dt) AS day_of_week,
    SUM(sale_amount) AS total_sales
FROM public.raw_sales_data
GROUP BY EXTRACT(DOW FROM dt)
ORDER BY total_sales DESC;