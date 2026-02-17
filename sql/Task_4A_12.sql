-- Question 12: Compare average sales between in-stock and stocked-out hours.

SELECT
    CASE 
        WHEN hour_flag = 0 THEN 'Stockout'
        ELSE 'In Stock'
    END AS stock_status,
    AVG(sale_amount::NUMERIC) AS avg_sales
FROM (
    SELECT sale_amount,
           unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::NUMERIC AS hour_flag
    FROM public.raw_sales_data
) AS expanded
GROUP BY
    CASE 
        WHEN hour_flag = 0 THEN 'Stockout'
        ELSE 'In Stock'
    END;