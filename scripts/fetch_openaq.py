#!/usr/bin/env python3
"""
Fetch AQI from OpenAQ API (openaq.org)
Has real Indian government station data.
All inserted data carries provenance flags.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
import time
from lib.logging import setup_logger
from lib.db import get_engine, get_data_freshness, insert_city_data
from lib.aqi import pm25_to_aqi, generate_synthetic_aqi
from lib.config import BASE_AQI, CITIES
from lib.utils import retry

BASE_URL = "https://api.openaq.org/v2"
logger = setup_logger("fetch-openaq")


@retry(max_attempts=3, delay=2, backoff=2, exceptions=(requests.RequestException,))
def fetch_city_aq(city, start_date, end_date):
    logger.info(f"  Fetching {city}...")

    try:
        params = {
            'city': city,
            'parameter': 'pm25',
            'date_from': start_date,
            'date_to': end_date,
            'limit': 10000
        }

        response = requests.get(f"{BASE_URL}/measurements", params=params, timeout=30)
        data = response.json()

        if 'results' not in data or not data['results']:
            logger.warning(f"No PM2.5 data for {city}")
            return None

        records = []
        for r in data['results']:
            records.append({
                'date': r['date']['local'][:10],
                'pm25': r['value'],
                'city': city,
                'location': r.get('location', 'unknown')
            })

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])

        df_daily = df.groupby('date')['pm25'].mean().reset_index()
        df_daily['city'] = city
        df_daily['aqi'] = df_daily['pm25'].apply(pm25_to_aqi).round(0)

        logger.info(f"  Got {len(df_daily)} days")
        return df_daily[['date', 'city', 'aqi', 'pm25']]

    except Exception as e:
        logger.error(f"Error: {str(e)[:100]}")
        return None


def generate_synthetic_data():
    logger.info("=== Generating Synthetic Data (Realistic Pattern) ===")

    dates = pd.date_range('2020-07-01', '2024-12-31', freq='D')
    all_data = []

    for city in CITIES:
        base = BASE_AQI[city]
        aqi = generate_synthetic_aqi(city, dates, base)

        df = pd.DataFrame({
            'date': dates,
            'city': city,
            'aqi': aqi.round(0),
            'pm25': (aqi / 2.5).round(1),
        })

        all_data.append(df)
        logger.info(f"  Generated {len(df)} synthetic days for {city}")

    combined = pd.concat(all_data, ignore_index=True)
    save_data(combined, synthetic=True)


def save_data(df, synthetic=False):
    logger.info("=== Summary ===")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"Cities: {df['city'].nunique()}")

    suffix = "_synthetic" if synthetic else "_fetched"
    output = f'data/processed/aqi_2020_2024{suffix}.csv'
    df.to_csv(output, index=False)
    logger.info(f"Saved: {output}")

    logger.info("=== Database ===")
    try:
        engine = get_engine()
        freshness = get_data_freshness(engine)
        logger.info(f"Existing data: {freshness['total_rows']:,} rows")

        data_source = "synthetic_realistic" if synthetic else "OpenAQ"
        insert_city_data(engine, df, data_source=data_source, is_synthetic=synthetic)
        logger.info(f"Inserted {len(df)} rows (source={data_source})")
    except Exception as e:
        logger.error(f"DB: {e}")


def main():
    start = '2020-07-01'
    end = '2024-12-31'

    logger.info(f"=== OpenAQ Fetch: {start} to {end} ===")

    all_data = []
    for city in CITIES:
        df = fetch_city_aq(city, start, end)
        if df is not None:
            all_data.append(df)
        time.sleep(1)

    if not all_data:
        logger.warning("No data from OpenAQ API. Falling back to synthetic...")
        generate_synthetic_data()
        return

    combined = pd.concat(all_data, ignore_index=True)
    save_data(combined)


if __name__ == '__main__':
    main()
