CREATE VIEW category_intelligence_new AS
WITH all_categories AS (
    SELECT
        city_id,
        store_id,
        first_category_id AS category_id,
        dt,
        sale_amount,
        hours_stock_status,
        discount,
        holiday_flag,
        precpt
    FROM  public.raw_sales_data

    UNION ALL

    SELECT
        city_id,
        store_id,
        second_category_id AS category_id,
        dt,
        sale_amount,
        hours_stock_status,
        discount,
        holiday_flag,
        precpt
    FROM  public.raw_sales_data

    UNION ALL

    SELECT
        city_id,
        store_id,
        third_category_id AS category_id,
        dt,
        sale_amount,
        hours_stock_status,
        discount,
        holiday_flag,
        precpt
    FROM  public.raw_sales_data
),

expanded AS (
    SELECT *,
           unnest(string_to_array(trim(both '[]' from hours_stock_status), ' '))::INT AS hour_flag
    FROM all_categories
)

SELECT
    category_id,

    COUNT(DISTINCT store_id) AS stores_covered,
    COUNT(*) AS total_category_hours,
    SUM(sale_amount::NUMERIC) AS total_sales,

    AVG(sale_amount::NUMERIC) AS avg_sales_per_hour,
    STDDEV(sale_amount::NUMERIC) AS sales_volatility,

    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END)
        AS stockout_hours,
    SUM(CASE WHEN hour_flag = 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 1.0 AS stockout_rate,
    SUM(CASE WHEN hour_flag = 0 THEN sale_amount::NUMERIC ELSE 0 END)
        AS lost_sales_due_to_stockout,

    AVG(CASE WHEN discount::NUMERIC > 0 THEN sale_amount::NUMERIC END)
        AS avg_sales_with_discount,
    AVG(CASE WHEN discount::NUMERIC = 0 THEN sale_amount::NUMERIC END)
        AS avg_sales_without_discount,
    (AVG(CASE WHEN discount::NUMERIC > 0 THEN sale_amount::NUMERIC END)
     - AVG(CASE WHEN discount::NUMERIC = 0 THEN sale_amount END))
        AS promo_sales_lift,

    AVG(CASE WHEN precpt::NUMERIC > 0 THEN sale_amount::NUMERIC END)
        AS avg_sales_rainy,
    AVG(CASE WHEN precpt::NUMERIC = 0 THEN sale_amount::NUMERIC END)
        AS avg_sales_clear,
    (AVG(CASE WHEN precpt::NUMERIC > 0 THEN sale_amount::NUMERIC END) - AVG(CASE WHEN precpt::NUMERIC = 0 THEN sale_amount::NUMERIC END))
        AS weather_sales_impact,

    AVG(CASE WHEN holiday_flag::INT = 1 THEN sale_amount::NUMERIC END)
        AS avg_sales_holiday,
    AVG(CASE WHEN holiday_flag::INT = 0 THEN sale_amount::NUMERIC END)
        AS avg_sales_non_holiday

FROM expanded
WHERE category_id IS NOT NULL
GROUP BY category_id;

-- Test your view: Write a query that identifies categories with high promotion responsiveness
-- but also high stockout challenges.

SELECT
    category_id,
    promo_sales_lift,
    stockout_rate,
    total_sales
FROM category_intelligence_new
WHERE promo_sales_lift > 0
  AND stockout_rate > 0.2
ORDER BY promo_sales_lift DESC;
