-- Question 1: How many stores and cities are in the dataset?

SELECT
    COUNT(DISTINCT store_id) AS total_stores,
    COUNT(DISTINCT city_id) AS total_cities
FROM public.raw_sales_data;