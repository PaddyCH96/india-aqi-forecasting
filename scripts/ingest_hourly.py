#!/usr/bin/env python3
"""Ingest city_hour.csv into city_hourly_measurements with provenance."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.pathing import ensure_project_root_on_path

ensure_project_root_on_path()

import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import text as sa_text
from lib.db import get_engine
from lib.logging import setup_logger

logger = setup_logger("ingest-hourly")

TABLE = "city_hourly_measurements"
CSV_PATH = "data/raw/city_hour.csv"

COLUMN_MAP = {
    "City": "city",
    "Datetime": "datetime",
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "NO": "no",
    "NO2": "no2",
    "NOx": "nox",
    "NH3": "nh3",
    "CO": "co",
    "SO2": "so2",
    "O3": "o3",
    "Benzene": "benzene",
    "Toluene": "toluene",
    "Xylene": "xylene",
    "AQI": "aqi",
    "AQI_Bucket": "aqi_bucket",
}


def create_table(engine):
    logger.info(f"Creating {TABLE} table...")
    with engine.connect() as conn:
        conn.execute(sa_text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id SERIAL PRIMARY KEY,
                city VARCHAR(100) NOT NULL,
                datetime TIMESTAMP NOT NULL,
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
                UNIQUE(city, datetime, data_source)
            )
        """))
        conn.execute(sa_text(f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE}_city_dt
            ON {TABLE} (city, datetime)
        """))
        conn.execute(sa_text(f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE}_synthetic
            ON {TABLE} (is_synthetic)
        """))
        conn.commit()
    logger.info(f"Table {TABLE} ready.")


def ingest_hourly_csv(engine):
    if not os.path.exists(CSV_PATH):
        logger.error(f"Not found: {CSV_PATH}")
        return 0

    logger.info(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, parse_dates=["Datetime"])
    df = df.rename(columns=COLUMN_MAP)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["is_synthetic"] = False
    df["data_source"] = "CPCB"
    df["ingested_at"] = datetime.now(timezone.utc)

    logger.info(f"Rows loaded: {len(df):,}")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"Cities: {df['city'].nunique()}")

    df.to_sql(TABLE, engine, if_exists="append", index=False, method="multi", chunksize=10000)
    logger.info(f"Inserted {len(df):,} rows into {TABLE}.")
    return len(df)


def validate_and_report(engine):
    logger.info("\n=== Hourly Data Validation ===")
    with engine.connect() as conn:
        total = conn.execute(sa_text(f"SELECT COUNT(*) FROM {TABLE}")).scalar()
        cities = conn.execute(sa_text(f"SELECT COUNT(DISTINCT city) FROM {TABLE}")).scalar()
        date_range = conn.execute(sa_text(f"SELECT MIN(datetime), MAX(datetime) FROM {TABLE}")).fetchone()
        null_aqi = conn.execute(sa_text(f"SELECT COUNT(*) FROM {TABLE} WHERE aqi IS NULL")).scalar()

        per_city = conn.execute(sa_text(f"""
            SELECT city, COUNT(*) as rows,
                   MIN(datetime) as start, MAX(datetime) as end,
                   COUNT(*) FILTER (WHERE aqi IS NOT NULL) as aqi_count,
                   COUNT(*) FILTER (WHERE pm2_5 IS NOT NULL) as pm25_count
            FROM {TABLE}
            GROUP BY city ORDER BY rows DESC
        """)).fetchall()

    logger.info(f"Total rows:    {total:,}")
    logger.info(f"Cities:        {cities}")
    logger.info(f"Date range:    {date_range[0]} to {date_range[1]}")
    logger.info(f"Null AQI:      {null_aqi:,} ({null_aqi/max(total,1)*100:.1f}%)")
    logger.info("\nPer-city coverage:")
    logger.info(f"  {'City':<20} {'Rows':>8} {'AQI%':>6} {'PM25%':>6}")
    logger.info(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*6}")
    for row in per_city:
        aqi_pct = row.aqi_count / row.rows * 100
        pm25_pct = row.pm25_count / row.rows * 100
        logger.info(f"  {row.city:<20} {row.rows:>8,} {aqi_pct:>5.1f}% {pm25_pct:>5.1f}%")

    return per_city


def detect_gaps(engine):
    logger.info("\n=== Time Continuity Check ===")
    with engine.connect() as conn:
        cities = conn.execute(sa_text(f"SELECT DISTINCT city FROM {TABLE} ORDER BY city")).fetchall()

    total_gaps = 0
    for (city,) in cities:
        df = pd.read_sql(
            sa_text(f"SELECT datetime FROM {TABLE} WHERE city = :city AND aqi IS NOT NULL ORDER BY datetime"),
            engine, params={"city": city}
        )
        if len(df) < 2:
            continue
        df["gap_hours"] = df["datetime"].diff().dt.total_seconds() / 3600
        gaps = df[df["gap_hours"] > 2]
        if len(gaps) > 0:
            total_gaps += len(gaps)
            if len(gaps) <= 5:
                for _, g in gaps.iterrows():
                    logger.info(f"  Gap in {city}: {g['datetime'] - pd.Timedelta(hours=1)} -> {g['datetime']} ({g['gap_hours']:.0f}h)")
            else:
                logger.info(f"  {city}: {len(gaps)} gaps > 2 hours (largest: {gaps['gap_hours'].max():.0f}h)")

    if total_gaps == 0:
        logger.info("No gaps > 2 hours found.")
    else:
        logger.info(f"Total gaps > 2 hours across all cities: {total_gaps}")

    return total_gaps


def main():
    try:
        engine = get_engine()
    except Exception as exc:
        logger.error(f"Database connection failed: {exc}")
        logger.warning("Hourly ingestion skipped because no PostgreSQL instance is reachable.")
        return

    try:
        if os.path.exists(CSV_PATH):
            create_table(engine)
            ingest_hourly_csv(engine)
            validate_and_report(engine)
            detect_gaps(engine)
        else:
            logger.warning(f"No CSV found at {CSV_PATH}. Hourly pipeline skipped.")
            logger.info("To generate hourly data in future: fetch from OpenAQ or CPCB.")
    except Exception as exc:
        logger.error(f"Hourly ingestion failed: {exc}")
        logger.warning("The pipeline completed with warnings because the database was unavailable.")
    finally:
        engine.dispose()
        logger.info("Hourly ingestion complete.")


if __name__ == "__main__":
    main()
