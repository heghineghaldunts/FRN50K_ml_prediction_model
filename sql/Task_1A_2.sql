-- Question 2: What date range does our data cover?

SELECT
    MIN(dt) AS start_date,
    MAX(dt) AS end_date
FROM public.raw_sales_data;