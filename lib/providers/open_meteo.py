"""Open-Meteo ERA5 archive provider.

Fetches daily reanalysis weather plus hourly boundary layer height for a single
point and reshapes it into the `city_weather` schema. The archive API is keyless;
`models=era5` is pinned so the returned values really are ERA5 and not the
seamless ERA5/ERA5-Land blend the default resolves to at a different grid point.
"""

from datetime import datetime, timezone

import pandas as pd
import requests

from lib.logging import setup_logger
from lib.utils import retry

logger = setup_logger("open-meteo")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
IST = "Asia/Kolkata"
IST_OFFSET_SECONDS = 19800

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
]

# There is no daily aggregate for the boundary layer: the API rejects a daily
# request for it outright, so it is pulled hourly and aggregated here.
HOURLY_VARS = ["boundary_layer_height"]

COLUMN_MAP = {
    "temperature_2m_mean": "temp_mean_c",
    "temperature_2m_max": "temp_max_c",
    "temperature_2m_min": "temp_min_c",
    "precipitation_sum": "precipitation_mm",
    "wind_speed_10m_mean": "wind_speed_mean_kmh",
    "wind_speed_10m_max": "wind_speed_max_kmh",
    "wind_direction_10m_dominant": "wind_direction_deg",
    "relative_humidity_2m_mean": "relative_humidity_pct",
    "surface_pressure_mean": "surface_pressure_hpa",
}


@retry(max_attempts=3, delay=2, backoff=2, exceptions=(requests.RequestException,))
def fetch_era5_archive(latitude, longitude, start_date, end_date):
    """Return the raw ERA5 archive payload for one point and date range.

    Unlike `fetch_city_data` in scripts/fetch_recent_aqi.py, which swallows every
    exception and returns None, this deliberately raises. A silently skipped or
    silently time-shifted city is worse than a loud failure.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": IST,
        "models": "era5",
    }

    logger.info(
        "Fetching ERA5 %s..%s at %s,%s", start_date, end_date, latitude, longitude
    )
    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    offset = payload.get("utc_offset_seconds")
    if offset != IST_OFFSET_SECONDS:
        raise ValueError(
            "Refusing to ingest a non-IST payload: utc_offset_seconds="
            f"{offset!r} (expected {IST_OFFSET_SECONDS}), timezone="
            f"{payload.get('timezone')!r}. Daily aggregates would be shifted "
            "into the wrong local window."
        )

    return payload


def parse_era5_daily(payload, city):
    """Reshape an ERA5 archive payload into one `city_weather` row per IST day."""
    daily = pd.DataFrame(payload["daily"])
    daily["date"] = pd.to_datetime(daily["time"])
    daily = daily.drop(columns=["time"])

    # The API has already localised these stamps to IST; slice the date off the
    # string rather than converting timezones client-side.
    hourly = pd.DataFrame(payload["hourly"])
    hourly["date"] = pd.to_datetime(hourly["time"].str[:10])
    blh = (
        hourly.groupby("date")["boundary_layer_height"]
        .agg(blh_mean_m="mean", blh_max_m="max", blh_min_m="min")
        .reset_index()
    )

    df = daily.merge(blh, on="date", how="left")

    df["city"] = city
    df["grid_latitude"] = payload["latitude"]
    df["grid_longitude"] = payload["longitude"]
    df["grid_elevation_m"] = payload["elevation"]
    df["data_source"] = "open_meteo_era5"
    df["is_synthetic"] = False
    df["ingested_at"] = datetime.now(timezone.utc)

    return df.rename(columns=COLUMN_MAP)
