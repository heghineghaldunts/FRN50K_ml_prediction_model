-- Question 29: If a store could eliminate all stockouts, estimate the potential sales increase.

WITH expanded AS (
    SELECT
        store_id,
        sale_amount::NUMERIC,
        unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::NUMERIC AS hour_flag
    FROM public.raw_sales_data
),
store_avg_sales AS (
    SELECT
        store_id,
        AVG(sale_amount) AS avg_in_stock_sales
    FROM expanded
    WHERE hour_flag = 1
    GROUP BY store_id
)
SELECT
    s.store_id,
    SUM(s.sale_amount) AS actual_sales,
    SUM(
        CASE
            WHEN s.hour_flag = 0 THEN a.avg_in_stock_sales
            ELSE s.sale_amount
        END
    ) AS potential_sales,
    SUM(
        CASE
            WHEN s.hour_flag = 0 THEN a.avg_in_stock_sales - s.sale_amount
            ELSE 0
        END
    ) AS potential_uplift
FROM expanded s
JOIN store_avg_sales a
  ON s.store_id = a.store_id
GROUP BY s.store_id
ORDER BY potential_uplift DESC;
