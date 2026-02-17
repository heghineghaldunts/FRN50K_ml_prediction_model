-- Question 25: Do promotional activities help during stockout situations?

SELECT
    CASE 
        WHEN discount::NUMERIC > 0 THEN 'Promotion'
        ELSE 'No Promotion'
    END AS promo_status,
    COUNT(*) AS stockout_hours,
    AVG(sale_amount::NUMERIC) AS avg_sales_during_stockout,
    SUM(sale_amount::NUMERIC) AS total_sales_during_stockout
FROM (
    SELECT 
        sale_amount,
        discount,
        unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::NUMERIC AS hour_flag
    FROM public.raw_sales_data
) AS expanded
WHERE hour_flag = 0
GROUP BY CASE 
             WHEN discount::NUMERIC > 0 THEN 'Promotion'
             ELSE 'No Promotion'
         END;

