-- Question 7: Show total sales by hour of day. Which hours are busiest?

SELECT
    hours_sale AS hour_of_day,
    SUM(sale_amount) AS total_sales
FROM public.raw_sales_data
GROUP BY hours_sale
ORDER BY total_sales DESC;

