#!/usr/bin/env python3
"""Seed the database from available CSV files or generate synthetic data."""

import os
import pandas as pd
from lib.db import get_engine
from lib.logging import setup_logger
from lib.config import CITIES, BASE_AQI
from lib.aqi import generate_synthetic_aqi

logger = setup_logger("seed-data")


def db_has_data(engine) -> bool:
    try:
        with engine.connect() as conn:
            result = conn.execute(
                __import__("sqlalchemy").text("SELECT COUNT(*) FROM city_day")
            ).scalar()
            return (result or 0) > 0
    except Exception:
        return False


def seed_from_processed_csv(engine) -> bool:
    csv_path = "data/processed/aqi_2020_2024_synthetic.csv"
    if not os.path.exists(csv_path):
        logger.warning(f"Not found: {csv_path}")
        return False
    df = pd.read_csv(csv_path)
    if "pm2_5" in df.columns and "pm25" not in df.columns:
        pass
    elif "pm25" in df.columns and "pm2_5" not in df.columns:
        df = df.rename(columns={"pm25": "pm2_5"})
    elif "pm25" in df.columns and "pm2_5" in df.columns:
        pass
    df.to_sql("city_day", engine, if_exists="append", index=False)
    logger.info(f"Inserted {len(df)} rows from {csv_path}")
    return True


def seed_from_raw_csv(engine) -> bool:
    csv_path = "data/raw/city_day.csv"
    if not os.path.exists(csv_path):
        logger.warning(f"Not found: {csv_path}")
        return False
    df = pd.read_csv(csv_path)
    df.to_sql("city_day", engine, if_exists="append", index=False)
    logger.info(f"Inserted {len(df)} rows from {csv_path}")
    return True


def seed_synthetic(engine):
    dates = pd.date_range("2020-07-01", "2024-12-31", freq="D")
    all_data = []
    for city in CITIES:
        aqi = generate_synthetic_aqi(city, dates, BASE_AQI[city])
        df = pd.DataFrame({
            "date": dates,
            "city": city,
            "aqi": aqi.round(0),
            "pm2_5": (aqi / 2.5).round(1),
        })
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_sql("city_day", engine, if_exists="append", index=False)
    logger.info(f"Inserted {len(combined)} synthetic rows across {len(CITIES)} cities.")


def main():
    engine = get_engine()
    if db_has_data(engine):
        logger.info("Database already has data. Skipping seed.")
        return

    logger.info("Seeding database...")
    if not seed_from_processed_csv(engine):
        if not seed_from_raw_csv(engine):
            seed_synthetic(engine)

    engine.dispose()
    logger.info("Seed complete.")


if __name__ == "__main__":
    main()
