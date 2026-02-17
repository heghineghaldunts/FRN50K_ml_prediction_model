-- Question 3: How many different products and product categories exist?

SELECT
    COUNT(DISTINCT product_id) AS total_products,
    (
        SELECT COUNT(DISTINCT category_id)
        FROM (
            SELECT first_category_id  AS category_id FROM public.raw_sales_data
            UNION
            SELECT second_category_id FROM public.raw_sales_data
            UNION
            SELECT third_category_id  FROM public.raw_sales_data
        ) c
    ) AS total_categories
FROM public.raw_sales_data;