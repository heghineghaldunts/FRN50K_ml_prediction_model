-- Question 16: Rank the top 10 stores by total sales.

SELECT
    store_id,
    city_id,
    SUM(sale_amount) AS total_sales
FROM public.raw_sales_data
GROUP BY store_id, city_id
ORDER BY total_sales DESC
LIMIT 10;
