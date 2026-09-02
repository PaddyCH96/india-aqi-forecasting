from sqlalchemy import create_engine, text as sa_text
import pandas as pd

from lib.config import DB_URL


def get_engine(url: str | None = None):
    return create_engine(url or DB_URL)


def _synthetic_filter(use_synthetic: bool = False) -> str:
    return "" if use_synthetic else " AND NOT is_synthetic"


def load_city_data(
    engine,
    city: str,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    syn_filter = _synthetic_filter(use_synthetic)
    query = sa_text(f"""
        SELECT date as ds, aqi as y
        FROM city_measurements
        WHERE city = :city AND aqi IS NOT NULL{syn_filter}
        ORDER BY date
    """)
    df = pd.read_sql(query, engine, params={"city": city})
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def load_city_pollutants(
    engine,
    city: str,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    syn_filter = _synthetic_filter(use_synthetic)
    query = sa_text(f"""
        SELECT city, date, pm2_5, pm10, no, no2, nox, nh3, co, so2, o3,
               benzene, toluene, xylene, aqi, aqi_bucket,
               is_synthetic, data_source
        FROM city_measurements
        WHERE city = :city{syn_filter}
        ORDER BY date
    """)
    df = pd.read_sql(query, engine, params={"city": city})
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_city_weather(engine, city: str) -> pd.DataFrame:
    """Load daily ERA5 weather for one city, keyed on (city, date).

    Deliberately mirrors ``load_city_pollutants``' shape so callers can merge the
    two frames in pandas on ``["city", "date"]``. The join belongs in pandas, not
    in SQL: ``city_measurements.date`` is stored as TEXT in SQLite
    ('2015-01-01 00:00:00.000000') but as a real DATE in PostgreSQL, and the
    ``date()`` normalisation needed to reconcile them is not portable between the
    two dialects. Returning real Timestamps from both helpers sidesteps that
    entirely -- an equality join on ISO date strings silently returns zero rows
    in SQLite and every row in PostgreSQL.
    """
    query = sa_text("""
        SELECT city, date, temp_mean_c, temp_max_c, temp_min_c,
               precipitation_mm, wind_speed_mean_kmh, wind_speed_max_kmh,
               wind_direction_deg, relative_humidity_pct, surface_pressure_hpa,
               blh_mean_m, blh_max_m, blh_min_m, data_source
        FROM city_weather
        WHERE city = :city AND data_source = 'open_meteo_era5'
        ORDER BY date
    """)
    df = pd.read_sql(query, engine, params={"city": city})
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_cities_with_recent_data(
    engine,
    use_synthetic: bool = False,
) -> list[str]:
    syn_filter = _synthetic_filter(use_synthetic)
    cutoff = "'2024-01-01'" if use_synthetic else "'2020-01-01'"
    query = sa_text(f"""
        SELECT DISTINCT city FROM city_measurements
        WHERE aqi IS NOT NULL AND date >= {cutoff}{syn_filter}
        ORDER BY city
    """)
    cities_df = pd.read_sql(query, engine)
    return cities_df["city"].tolist()


def get_cities_with_data_summary(
    engine,
    min_days: int = 1000,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    syn_filter = _synthetic_filter(use_synthetic)
    query = sa_text(f"""
        SELECT city, COUNT(*) as days,
               MIN(date) as start_date,
               MAX(date) as end_date
        FROM city_measurements
        WHERE aqi IS NOT NULL{syn_filter}
        GROUP BY city
        HAVING COUNT(*) >= :min_days
        ORDER BY COUNT(*) DESC
    """)
    return pd.read_sql(query, engine, params={"min_days": min_days})


def get_eligible_cities(
    engine,
    min_days: int = 1000,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    syn_filter = _synthetic_filter(use_synthetic)
    cutoff = "'2024-01-01'" if use_synthetic else "'2020-01-01'"
    query = sa_text(f"""
        SELECT city, COUNT(*) as total_days, MAX(date) as end_date
        FROM city_measurements
        WHERE aqi IS NOT NULL{syn_filter}
        GROUP BY city
        HAVING COUNT(*) >= :min_days AND MAX(date) >= {cutoff}
        ORDER BY total_days DESC
    """)
    return pd.read_sql(query, engine, params={"min_days": min_days})


def count_recent_rows(
    engine,
    use_synthetic: bool = False,
) -> int:
    syn_filter = _synthetic_filter(use_synthetic)
    with engine.connect() as conn:
        return conn.execute(
            sa_text(f"SELECT COUNT(*) FROM city_measurements WHERE date >= '2020-07-01'{syn_filter}")
        ).scalar() or 0


def insert_city_data(
    engine,
    df: pd.DataFrame,
    data_source: str = "unknown",
    is_synthetic: bool = False,
):
    df_to_insert = df.rename(columns={"pm25": "pm2_5"})
    df_to_insert["data_source"] = data_source
    df_to_insert["is_synthetic"] = is_synthetic
    df_to_insert.to_sql("city_measurements", engine, if_exists="append", index=False)


def get_data_freshness(engine) -> dict:
    with engine.connect() as conn:
        total = conn.execute(sa_text("SELECT COUNT(*) FROM city_measurements")).scalar()
        real = conn.execute(sa_text("SELECT COUNT(*) FROM city_measurements WHERE NOT is_synthetic")).scalar()
        synthetic = conn.execute(sa_text("SELECT COUNT(*) FROM city_measurements WHERE is_synthetic")).scalar()
        cities = conn.execute(sa_text("SELECT COUNT(DISTINCT city) FROM city_measurements")).scalar()
        last_ingest = conn.execute(
            sa_text("SELECT MAX(ingested_at) FROM city_measurements")
        ).scalar()
        latest_date = conn.execute(
            sa_text("SELECT MAX(date) FROM city_measurements WHERE NOT is_synthetic")
        ).scalar()
        sources = conn.execute(
            sa_text("SELECT data_source, COUNT(*) as cnt FROM city_measurements GROUP BY data_source ORDER BY cnt DESC")
        ).fetchall()
    return {
        "total_rows": total,
        "real_rows": real,
        "synthetic_rows": synthetic,
        "cities": cities,
        "last_ingested_at": str(last_ingest) if last_ingest else None,
        "latest_real_date": str(latest_date) if latest_date else None,
        "data_sources": {src: cnt for src, cnt in sources},
    }
