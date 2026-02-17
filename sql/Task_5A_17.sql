-- Question 17: Which cities have the highest average sales per store?

SELECT
    city_id,
    AVG(store_total_sales) AS avg_sales_per_store
FROM (
    SELECT
        city_id,
        store_id,
        SUM(sale_amount) AS store_total_sales
    FROM public.raw_sales_data
    GROUP BY city_id, store_id
) t
GROUP BY city_id
ORDER BY avg_sales_per_store DESC;
