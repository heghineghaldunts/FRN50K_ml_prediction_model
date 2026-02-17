-- Question 26: Find store-product combinations with high sales potential but frequent stockouts.

SELECT
    store_id,
    product_id,
    AVG(CASE WHEN hour_flag = 1 THEN sale_amount::NUMERIC END) AS avg_sales_in_stock,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END)::NUMERIC
        / COUNT(*) * 1.0 AS stockout_rate
FROM (
    SELECT 
        store_id,
        product_id,
        sale_amount,
        unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::NUMERIC AS hour_flag
    FROM public.raw_sales_data
) AS expanded
GROUP BY store_id, product_id
HAVING
    AVG(CASE WHEN hour_flag = 1 THEN sale_amount::NUMERIC END) >
        (SELECT AVG(sale_amount::NUMERIC) FROM public.raw_sales_data)
    AND
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END)::NUMERIC
        / COUNT(*) > 0.2
ORDER BY avg_sales_in_stock DESC;

