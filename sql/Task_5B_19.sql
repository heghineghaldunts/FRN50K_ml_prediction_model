-- Question 19: Which product categories generate the most total sales?

SELECT
    category_id,
    SUM(sale_amount) AS total_sales
FROM (
    SELECT first_category_id AS category_id, sale_amount FROM public.raw_sales_data
    UNION ALL
    SELECT second_category_id, sale_amount FROM public.raw_sales_data
    UNION ALL
    SELECT third_category_id, sale_amount FROM public.raw_sales_data
) t
WHERE category_id IS NOT NULL
GROUP BY category_id
ORDER BY total_sales DESC;
