-- Query 1: City-level AQI summary ranked by worst air quality
SELECT 
    city,
    ROUND(AVG(aqi), 1) AS avg_aqi,
    ROUND(MIN(aqi), 1) AS min_aqi,
    ROUND(MAX(aqi), 1) AS max_aqi,
    COUNT(aqi) AS days_recorded
FROM city_day
WHERE aqi IS NOT NULL
GROUP BY city
ORDER BY avg_aqi DESC;