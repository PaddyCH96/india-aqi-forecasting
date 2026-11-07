#!/usr/bin/env python3
"""Ingest ERA5 daily weather for the forecast cities into city_weather."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd
from sqlalchemy import text as sa_text

from lib.config import CITIES, CITY_COORDS
from lib.db import get_engine
from lib.logging import setup_logger
from lib.providers.open_meteo import fetch_era5_archive, parse_era5_daily

logger = setup_logger("ingest-weather")

TABLE = "city_weather"
DATA_SOURCE = "open_meteo_era5"
CACHE_DIR = str(Path(REPO_ROOT) / "data" / "cache" / "weather")


def _dialect(engine):
    """SERIAL/NOW() are PostgreSQL spellings; SQLite uses AUTOINCREMENT/CURRENT_TIMESTAMP."""
    if engine.dialect.name == "sqlite":
        return "INTEGER PRIMARY KEY AUTOINCREMENT", "CURRENT_TIMESTAMP"
    return "SERIAL PRIMARY KEY", "NOW()"


def create_table(engine):
    logger.info(f"Creating {TABLE} table...")
    pk_type, now_fn = _dialect(engine)
    with engine.connect() as conn:
        conn.execute(
            sa_text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id {pk_type},
                city VARCHAR(100) NOT NULL,
                date TIMESTAMP NOT NULL,
                temp_mean_c DOUBLE PRECISION,
                temp_max_c DOUBLE PRECISION,
                temp_min_c DOUBLE PRECISION,
                precipitation_mm DOUBLE PRECISION,
                wind_speed_mean_kmh DOUBLE PRECISION,
                wind_speed_max_kmh DOUBLE PRECISION,
                wind_direction_deg DOUBLE PRECISION,
                relative_humidity_pct DOUBLE PRECISION,
                surface_pressure_hpa DOUBLE PRECISION,
                blh_mean_m DOUBLE PRECISION,
                blh_max_m DOUBLE PRECISION,
                blh_min_m DOUBLE PRECISION,
                grid_latitude DOUBLE PRECISION,
                grid_longitude DOUBLE PRECISION,
                grid_elevation_m DOUBLE PRECISION,
                is_synthetic BOOLEAN NOT NULL DEFAULT FALSE,
                data_source VARCHAR(100) NOT NULL DEFAULT 'open_meteo_era5',
                ingested_at TIMESTAMP NOT NULL DEFAULT {now_fn},
                UNIQUE(city, date, data_source)
            )
        """)
        )
        conn.execute(
            sa_text(f"""
            CREATE INDEX IF NOT EXISTS idx_city_weather_city_date
            ON {TABLE} (city, date)
        """)
        )
        conn.commit()
    logger.info(f"Table {TABLE} ready.")


def _validate_payload(payload, city):
    offset = payload.get("utc_offset_seconds")
    tz = payload.get("timezone")
    expected = CITY_COORDS[city]
    expected_offset = expected["utc_offset_seconds"]
    expected_tz = expected["timezone"]
    if offset != expected_offset or tz != expected_tz:
        raise ValueError(
            f"Refusing to ingest a payload for {city} outside its expected "
            f"timezone: utc_offset_seconds={offset!r} (expected {expected_offset}), "
            f"timezone={tz!r} (expected {expected_tz!r}). Daily aggregates would "
            "be shifted into the wrong local window."
        )


def load_or_fetch_payload(city, start_date=None, end_date=None):
    cache_path = Path(CACHE_DIR) / f"{city}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        _validate_payload(payload, city)
        logger.info(f"Using cached ERA5 payload for {city} ({cache_path})")
        return payload

    coords = CITY_COORDS[city]
    payload = fetch_era5_archive(coords["lat"], coords["lon"], start_date, end_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    logger.info(f"Cached ERA5 payload for {city} at {cache_path}")
    return payload


def ingest_city(engine, city, df):
    if df.empty:
        logger.warning(f"No weather rows for {city}; nothing ingested.")
        return 0

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    out["is_synthetic"] = out["is_synthetic"].astype(bool)
    if "ingested_at" not in out or out["ingested_at"].isna().all():
        out["ingested_at"] = datetime.now(timezone.utc)

    start, end = out["date"].min(), out["date"].max()
    with engine.connect() as conn:
        conn.execute(
            sa_text(
                f"DELETE FROM {TABLE} WHERE city = :city AND data_source = :src "
                "AND date >= :start AND date <= :end"
            ),
            {"city": city, "src": DATA_SOURCE, "start": start, "end": end},
        )
        conn.commit()

    out.to_sql(TABLE, engine, if_exists="append", index=False)
    logger.info(f"Ingested {len(out):,} weather rows for {city} ({start}..{end}).")
    return len(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD")
    parser.add_argument(
        "--cities", nargs="*", default=CITIES, choices=list(CITY_COORDS)
    )
    args = parser.parse_args()

    engine = get_engine()
    try:
        create_table(engine)
        for city in args.cities:
            try:
                payload = load_or_fetch_payload(city, args.start, args.end)
                df = parse_era5_daily(payload, city)
                ingest_city(engine, city, df)
            except Exception as exc:
                logger.error(f"Weather ingest failed for {city}: {exc}")
                raise
    finally:
        engine.dispose()
    logger.info("Weather ingestion complete.")


if __name__ == "__main__":
    main()
