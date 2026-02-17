-- Question 23: Which product categories respond best to promotions?

SELECT
    category_id,
    AVG(CASE WHEN discount > 0 THEN sale_amount END)
        AS avg_sales_with_discount,
    AVG(CASE WHEN discount = 0 THEN sale_amount END)
        AS avg_sales_without_discount,
    (AVG(CASE WHEN discount > 0 THEN sale_amount END)
     - AVG(CASE WHEN discount = 0 THEN sale_amount END))
        AS promo_lift
FROM (
    SELECT first_category_id AS category_id, sale_amount, discount FROM public.raw_sales_data
    UNION ALL
    SELECT second_category_id, sale_amount, discount FROM public.raw_sales_data
    UNION ALL
    SELECT third_category_id, sale_amount, discount FROM public.raw_sales_data
) t
WHERE category_id IS NOT NULL
GROUP BY category_id
HAVING COUNT(*) > 100
ORDER BY promo_lift DESC;
