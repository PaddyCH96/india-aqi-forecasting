from sqlalchemy import create_engine, text
import pandas as pd

from lib.config import DB_URL


def get_engine(url: str | None = None):
    return create_engine(url or DB_URL)


def load_city_data(engine, city: str) -> pd.DataFrame:
    query = text("""
        SELECT date as ds, aqi as y
        FROM city_day
        WHERE city = :city AND aqi IS NOT NULL
        ORDER BY date
    """)
    df = pd.read_sql(query, engine, params={"city": city})
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def get_cities_with_recent_data(engine) -> list[str]:
    query = text("""
        SELECT DISTINCT city FROM city_day
        WHERE aqi IS NOT NULL AND date >= '2024-01-01'
        ORDER BY city
    """)
    cities_df = pd.read_sql(query, engine)
    return cities_df["city"].tolist()


def get_cities_with_data_summary(engine, min_days: int = 1000) -> pd.DataFrame:
    query = text("""
        SELECT city, COUNT(*) as days,
               MIN(date) as start_date,
               MAX(date) as end_date
        FROM city_day
        WHERE aqi IS NOT NULL
        GROUP BY city
        HAVING COUNT(*) >= :min_days
        ORDER BY COUNT(*) DESC
    """)
    return pd.read_sql(query, engine, params={"min_days": min_days})


def get_eligible_cities(engine, min_days: int = 1000) -> pd.DataFrame:
    query = text("""
        SELECT city, COUNT(*) as total_days, MAX(date) as end_date
        FROM city_day
        WHERE aqi IS NOT NULL
        GROUP BY city
        HAVING COUNT(*) >= :min_days AND MAX(date) >= '2024-01-01'
        ORDER BY total_days DESC
    """)
    return pd.read_sql(query, engine, params={"min_days": min_days})


def count_recent_rows(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT COUNT(*) FROM city_day WHERE date >= '2020-07-01'")
        ).scalar() or 0


def insert_city_data(engine, df: pd.DataFrame):
    df_to_insert = df.rename(columns={"pm25": "pm2_5"})
    df_to_insert.to_sql("city_day", engine, if_exists="append", index=False)
