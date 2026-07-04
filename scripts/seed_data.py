#!/usr/bin/env python3
"""Seed the database from available CSV files or generate synthetic data.

All inserts carry provenance flags (is_synthetic, data_source, ingested_at).

Priority: processed CSV > raw CPCB CSV > synthetic generation.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.pathing import ensure_project_root_on_path

ensure_project_root_on_path()
import pandas as pd
from datetime import datetime, timezone
from lib.db import get_engine, get_data_freshness
from lib.logging import setup_logger
from lib.config import CITIES, BASE_AQI
from lib.aqi import generate_synthetic_aqi

logger = setup_logger("seed-data")


def db_has_data(engine) -> bool:
    try:
        with engine.connect() as conn:
            result = conn.execute(
                __import__("sqlalchemy").text("SELECT COUNT(*) FROM city_measurements")
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
    df["data_source"] = "synthetic"
    df["is_synthetic"] = True
    df["ingested_at"] = datetime.now(timezone.utc)
    col_map = {"pm25": "pm2_5"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df.to_sql("city_measurements", engine, if_exists="append", index=False)
    logger.info(f"Inserted {len(df)} synthetic rows from {csv_path}")
    return True


def seed_from_raw_csv(engine) -> bool:
    csv_path = "data/raw/city_day.csv"
    if not os.path.exists(csv_path):
        logger.warning(f"Not found: {csv_path}")
        return False
    df = pd.read_csv(csv_path)
    col_map = {
        "City": "city", "Date": "date",
        "PM2.5": "pm2_5", "PM10": "pm10",
        "NO": "no", "NO2": "no2", "NOx": "nox", "NH3": "nh3",
        "CO": "co", "SO2": "so2", "O3": "o3",
        "Benzene": "benzene", "Toluene": "toluene", "Xylene": "xylene",
        "AQI": "aqi", "AQI_Bucket": "aqi_bucket",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["data_source"] = "CPCB"
    df["is_synthetic"] = False
    df["ingested_at"] = datetime.now(timezone.utc)
    df.to_sql("city_measurements", engine, if_exists="append", index=False)
    logger.info(f"Inserted {len(df)} real rows from {csv_path}")
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
            "data_source": "synthetic",
            "is_synthetic": True,
            "ingested_at": datetime.now(timezone.utc),
        })
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_sql("city_measurements", engine, if_exists="append", index=False)
    logger.info(f"Inserted {len(combined)} synthetic rows across {len(CITIES)} cities.")
    return combined


def main():
    try:
        engine = get_engine()
    except Exception as exc:
        logger.error(f"Database connection failed: {exc}")
        logger.warning("Seeding skipped because no PostgreSQL instance is reachable.")
        return

    try:
        if db_has_data(engine):
            freshness = get_data_freshness(engine)
            logger.info(
                f"city_measurements already has {freshness['total_rows']:,} rows "
                f"({freshness['real_rows']:,} real, {freshness['synthetic_rows']:,} synthetic). "
                "Skipping seed."
            )
            return

        logger.info("Seeding database...")
        if not seed_from_processed_csv(engine):
            if not seed_from_raw_csv(engine):
                seed_synthetic(engine)
        logger.info("Seed complete.")
    except Exception as exc:
        logger.error(f"Seeding failed: {exc}")
        logger.warning("The script completed with warnings because the database was unavailable.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
