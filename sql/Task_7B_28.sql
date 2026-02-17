-- Question 28: During peak hours (identify them first), which products are most likely to be out of stock?

WITH peak_hours AS (
    SELECT hours_sale
    FROM public.raw_sales_data
    GROUP BY hours_sale
    HAVING AVG(sale_amount::NUMERIC) > (
        SELECT AVG(sale_amount::NUMERIC) FROM public.raw_sales_data
    )
),
expanded AS (
    SELECT 
        product_id,
        hours_sale,
        unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::NUMERIC AS hour_flag
    FROM public.raw_sales_data
)
SELECT
    product_id,
    COUNT(*) AS stockout_hours
FROM expanded
WHERE hour_flag = 0
  AND hours_sale IN (SELECT hours_sale FROM peak_hours)
GROUP BY product_id
ORDER BY stockout_hours DESC
LIMIT 10;

