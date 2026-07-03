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
        SELECT date, pm2_5, pm10, no, no2, nox, nh3, co, so2, o3,
               benzene, toluene, xylene, aqi, aqi_bucket,
               is_synthetic, data_source
        FROM city_measurements
        WHERE city = :city{syn_filter}
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
    data_source: str = "OpenAQ",
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
