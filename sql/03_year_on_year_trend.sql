-- Query 3: Year-on-year AQI trend (note: cities_reporting increases over time)
SELECT 
    EXTRACT(YEAR FROM date) AS year,
    ROUND(AVG(aqi), 1) AS avg_aqi,
    ROUND(AVG(pm25), 1) AS avg_pm25,
    COUNT(DISTINCT city) AS cities_reporting
FROM city_day
WHERE aqi IS NOT NULL
GROUP BY EXTRACT(YEAR FROM date)
ORDER BY year;