#!/usr/bin/env python3
"""Initialize the database with provenance schema and seed from real CSVs only."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import text as sa_text
from lib.db import get_engine
from lib.logging import setup_logger
from lib.config import CITIES, BASE_AQI
from lib.aqi import generate_synthetic_aqi

logger = setup_logger("init-db")

REAL_SOURCES = {
    'city_day': 'CPCB CPCB_Daily_2020',
    'city_hour': 'CPCB CPCB_Hourly_2020',
    'station_day': 'CPCB CPCB_Station_Daily_2020',
    'station_hour': 'CPCB CPCB_Station_Hourly_2020',
}

def schema_exists(engine) -> bool:
    try:
        with engine.connect() as conn:
            result = conn.execute(
                sa_text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'city_measurements')")
            ).scalar()
            return bool(result)
    except Exception:
        return False


def create_provenance_schema(engine):
    logger.info("Creating city_measurements table with provenance...")
    with engine.connect() as conn:
        conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS city_measurements (
                id SERIAL PRIMARY KEY,
                city VARCHAR(100) NOT NULL,
                date DATE NOT NULL,
                pm2_5 DOUBLE PRECISION,
                pm10 DOUBLE PRECISION,
                no DOUBLE PRECISION,
                no2 DOUBLE PRECISION,
                nox DOUBLE PRECISION,
                nh3 DOUBLE PRECISION,
                co DOUBLE PRECISION,
                so2 DOUBLE PRECISION,
                o3 DOUBLE PRECISION,
                benzene DOUBLE PRECISION,
                toluene DOUBLE PRECISION,
                xylene DOUBLE PRECISION,
                aqi DOUBLE PRECISION,
                aqi_bucket VARCHAR(50),
                is_synthetic BOOLEAN NOT NULL DEFAULT FALSE,
                data_source VARCHAR(100) NOT NULL DEFAULT 'CPCB',
                ingested_at TIMESTAMP NOT NULL DEFAULT NOW(),
                UNIQUE(city, date, data_source)
            )
        """))
        conn.execute(sa_text("""
            CREATE INDEX IF NOT EXISTS idx_city_measurements_city_date
            ON city_measurements (city, date)
        """))
        conn.execute(sa_text("""
            CREATE INDEX IF NOT EXISTS idx_city_measurements_synthetic
            ON city_measurements (is_synthetic)
        """))
        conn.commit()
    logger.info("Schema created.")


def migrate_real_data(engine):
    """Load all 26 cities from city_day.csv with is_synthetic=FALSE."""
    csv_path = "data/raw/city_day.csv"
    if not os.path.exists(csv_path):
        logger.warning(f"Not found: {csv_path}")
        return 0

    logger.info(f"Loading real data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.rename(columns={
        'Date': 'date',
        'City': 'city',
        'PM2.5': 'pm2_5',
        'PM10': 'pm10',
        'NO': 'no',
        'NO2': 'no2',
        'NOx': 'nox',
        'NH3': 'nh3',
        'CO': 'co',
        'SO2': 'so2',
        'O3': 'o3',
        'Benzene': 'benzene',
        'Toluene': 'toluene',
        'Xylene': 'xylene',
        'AQI': 'aqi',
        'AQI_Bucket': 'aqi_bucket',
    })
    df['is_synthetic'] = False
    df['data_source'] = 'CPCB'
    df['ingested_at'] = datetime.now(timezone.utc)

    df.to_sql('city_measurements', engine, if_exists='append', index=False, method='multi')
    logger.info(f"Inserted {len(df)} real rows from CPCB CSV.")
    return len(df)


def add_synthetic_data(engine):
    """Add synthetic 2020-2024 AQI data for the 6 configured cities."""
    dates = pd.date_range("2020-07-01", "2024-12-31", freq="D")
    all_rows = []
    for city in CITIES:
        aqi = generate_synthetic_aqi(city, dates, BASE_AQI[city])
        for i, d in enumerate(dates):
            all_rows.append({
                'city': city,
                'date': d,
                'aqi': round(aqi[i], 0),
                'pm2_5': round(aqi[i] / 2.5, 1),
                'is_synthetic': True,
                'data_source': 'synthetic',
                'ingested_at': datetime.now(timezone.utc),
            })

    df = pd.DataFrame(all_rows)
    df.to_sql('city_measurements', engine, if_exists='append', index=False, method='multi')
    logger.info(f"Inserted {len(df)} synthetic rows across {len(CITIES)} cities.")
    return len(df)


def create_legacy_view(engine):
    """Create city_day view for backward compatibility."""
    with engine.connect() as conn:
        conn.execute(sa_text("DROP TABLE IF EXISTS city_day CASCADE"))
        conn.execute(sa_text("""
            CREATE VIEW city_day AS
            SELECT
                city,
                date,
                pm2_5 AS pm25,
                pm10,
                no,
                no2,
                nox,
                nh3,
                co,
                so2,
                o3,
                benzene,
                toluene,
                xylene,
                aqi,
                aqi_bucket,
                is_synthetic,
                data_source,
                ingested_at
            FROM city_measurements
        """))
        conn.commit()
    logger.info("Created city_day view for backward compatibility.")


def create_stations_table(engine):
    """Load stations.csv into DB."""
    csv_path = "data/raw/stations.csv"
    if not os.path.exists(csv_path):
        logger.warning(f"Not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    with engine.connect() as conn:
        conn.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS stations (
                station_id VARCHAR(20) PRIMARY KEY,
                station_name VARCHAR(200),
                city VARCHAR(100),
                state VARCHAR(100),
                status VARCHAR(20)
            )
        """))
        conn.commit()

    df = df.rename(columns={
        'StationId': 'station_id',
        'StationName': 'station_name',
        'City': 'city',
        'State': 'state',
        'Status': 'status',
    })
    df.to_sql('stations', engine, if_exists='replace', index=False)
    logger.info(f"Inserted {len(df)} stations.")


def print_summary(engine):
    with engine.connect() as conn:
        total = conn.execute(sa_text("SELECT COUNT(*) FROM city_measurements")).scalar()
        real = conn.execute(sa_text("SELECT COUNT(*) FROM city_measurements WHERE NOT is_synthetic")).scalar()
        synthetic = conn.execute(sa_text("SELECT COUNT(*) FROM city_measurements WHERE is_synthetic")).scalar()
        cities = conn.execute(sa_text("SELECT COUNT(DISTINCT city) FROM city_measurements")).scalar()
        date_range = conn.execute(sa_text("SELECT MIN(date), MAX(date) FROM city_measurements")).fetchone()
        sources = conn.execute(sa_text("SELECT data_source, COUNT(*) FROM city_measurements GROUP BY data_source")).fetchall()

    logger.info("=" * 50)
    logger.info("DATABASE INITIALIZATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Total rows:    {total:,}")
    logger.info(f"  Real rows:     {real:,}")
    logger.info(f"  Synthetic rows: {synthetic:,}")
    logger.info(f"  Cities:        {cities}")
    logger.info(f"  Date range:    {date_range[0]} to {date_range[1]}")
    for src, cnt in sources:
        logger.info(f"  Source '{src}': {cnt:,} rows")
    logger.info("=" * 50)


def main():
    engine = get_engine()

    if schema_exists(engine):
        logger.info("city_measurements table already exists. Dropping and recreating...")
        with engine.connect() as conn:
            conn.execute(sa_text("DROP VIEW IF EXISTS city_day"))
            conn.execute(sa_text("DROP TABLE IF EXISTS city_measurements CASCADE"))
            conn.commit()

    create_provenance_schema(engine)
    migrate_real_data(engine)
    add_synthetic_data(engine)
    create_legacy_view(engine)
    create_stations_table(engine)
    print_summary(engine)
    engine.dispose()


if __name__ == '__main__':
    main()
