-- Question 4: What is the total sales volume across all data?

SELECT
    SUM(sale_amount) AS total_sales_volume
FROM public.raw_sales_data;