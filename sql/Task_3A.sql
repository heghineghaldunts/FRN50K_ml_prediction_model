-- Write your CREATE VIEW statement:

CREATE VIEW hourly_business_summary_new AS
SELECT
    -- Time dimensions
    dt,
    hours_sale                                   AS hour_of_day,
    EXTRACT(DOW FROM dt)                         AS day_of_week,
    EXTRACT(MONTH FROM dt)                       AS month,
    CASE WHEN EXTRACT(DOW FROM dt) IN (0, 6)
         THEN 1 ELSE 0 END                       AS is_weekend,
    CASE WHEN CAST(hours_sale AS INT) BETWEEN 18 AND 23
         THEN 1 ELSE 0 END                       AS is_evening,

    -- Business context
    city_id,
    store_id,
    management_group_id,
    first_category_id,
    second_category_id,
    third_category_id,
    product_id,

    -- Core metrics
    SUM(sale_amount)                             AS hourly_sales,
    COUNT(*)                                     AS transaction_count,

    -- Inventory intelligence
    MAX(stock_hour6_22_cnt)                      AS stock_hours_available,
    CASE WHEN MAX(stock_hour6_22_cnt) = 0
         THEN 1 ELSE 0 END                       AS stockout_flag,

    -- External factors
    MAX(discount)                                AS discount_rate,
    MAX(holiday_flag)                            AS holiday_flag,
    CASE WHEN MAX(precpt) > 0
         THEN 1 ELSE 0 END                       AS is_rainy,

    -- Calculated business indicators
    CASE WHEN SUM(sale_amount) > 0
         THEN 1 ELSE 0 END                       AS has_sales,
    CASE WHEN SUM(sale_amount) > 0
              AND MAX(stock_hour6_22_cnt) = 0
         THEN 1 ELSE 0 END                       AS high_sales_stockout_risk

FROM public.raw_sales_data
GROUP BY
    dt,
    hours_sale,
    city_id,
    store_id,
    management_group_id,
    first_category_id,
    second_category_id,
    third_category_id,
    product_id;

-- Test your view: Write a query that uses your view to show total weekend sales by city during promotional periods.

SELECT
    city_id,
    SUM(hourly_sales) AS weekend_promo_sales
FROM hourly_business_summary_new
WHERE
    is_weekend = 1
    AND discount_rate > 0
GROUP BY city_id
ORDER BY weekend_promo_sales DESC;
