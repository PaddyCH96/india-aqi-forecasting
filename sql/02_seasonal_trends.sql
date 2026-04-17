-- Query 2: Monthly seasonality of AQI and PM2.5 across all cities
SELECT 
    EXTRACT(MONTH FROM date) AS month,
    TO_CHAR(date, 'Mon') AS month_name,
    ROUND(AVG(aqi), 1) AS avg_aqi,
    ROUND(AVG(pm25), 1) AS avg_pm25
FROM city_day
WHERE aqi IS NOT NULL
GROUP BY EXTRACT(MONTH FROM date), TO_CHAR(date, 'Mon')
ORDER BY month;