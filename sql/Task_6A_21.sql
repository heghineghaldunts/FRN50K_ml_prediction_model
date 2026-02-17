-- Question 21: Compare average sales between discounted (discount > 0) and regular price hours.

SELECT
    CASE 
        WHEN discount > 0 THEN 'Discounted'
        ELSE 'Regular Price'
    END AS price_type,
    AVG(sale_amount) AS avg_sales
FROM public.raw_sales_data
GROUP BY
    CASE 
        WHEN discount > 0 THEN 'Discounted'
        ELSE 'Regular Price'
    END;
