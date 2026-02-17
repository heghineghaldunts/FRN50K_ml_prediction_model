-- Question 20: Which individual products have the highest average hourly sales?

SELECT
    product_id,
    AVG(sale_amount) AS avg_hourly_sales
FROM public.raw_sales_data
GROUP BY product_id
HAVING COUNT(*) > 50          
ORDER BY avg_hourly_sales DESC
LIMIT 10;
