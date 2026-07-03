#!/usr/bin/env python3
import pandas as pd
import requests
import time
from lib.logging import setup_logger
from lib.db import get_engine, count_recent_rows, insert_city_data
from lib.aqi import pm25_to_aqi
from lib.utils import retry

logger = setup_logger("fetch-recent-aqi")

CITIES = {
    'Hyderabad': {'lat': 17.385, 'lon': 78.4867},
    'Delhi': {'lat': 28.6139, 'lon': 77.209},
    'Mumbai': {'lat': 19.076, 'lon': 72.8777},
    'Bengaluru': {'lat': 12.9716, 'lon': 77.5946},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639}
}


@retry(max_attempts=2, delay=3, backoff=1, exceptions=(requests.RequestException,))
def main():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    all_data = []

    logger.info("=== Fetching AQI 2020-2024 ===")

    for city, coords in CITIES.items():
        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'start_date': '2020-07-01',
            'end_date': '2024-12-31',
            'daily': ['pm10', 'pm2_5'],
            'timezone': 'Asia/Kolkata'
        }

        logger.info(f"Fetching {city}...")
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'daily' in data and data['daily']:
                df = pd.DataFrame({
                    'date': pd.to_datetime(data['daily']['time']),
                    'pm25': data['daily']['pm2_5'],
                    'pm10': data['daily']['pm10'],
                    'city': city
                })

                df['aqi'] = df['pm25'].apply(pm25_to_aqi).round(0)
                all_data.append(df)
                logger.info(f"  ✓ {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")
            else:
                logger.warning("No data returned")

        except Exception as e:
            logger.error(f"Error: {str(e)[:100]}")

        time.sleep(1.5)

    if not all_data:
        logger.error("No data fetched")
        return

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total: {len(combined)} rows")

    output = 'data/processed/aqi_2020_2024_fetched.csv'
    combined.to_csv(output, index=False)
    logger.info(f"✓ Saved: {output}")

    logger.info("=== Database ===")
    try:
        engine = get_engine()
        existing = count_recent_rows(engine)
        logger.info(f"Existing rows: {existing}")

        if existing == 0:
            insert_city_data(engine, combined)
            logger.info(f"✓ Inserted {len(combined)} rows")
        else:
            logger.warning("Data exists, skipping")
    except Exception as e:
        logger.error(f"DB error: {e}")


if __name__ == '__main__':
    main()
