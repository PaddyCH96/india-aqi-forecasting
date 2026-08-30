import os

def _default_db_url() -> str:
    """Pick a sensible database without configuration.

    Order: AQI_DB_URL if set (Docker, Render, local .env) -> the bundled
    read-only SQLite demo database if present (Streamlit Community Cloud,
    fresh clones) -> local PostgreSQL.
    """
    explicit = os.getenv("AQI_DB_URL")
    if explicit:
        return explicit

    demo = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "aqi_demo.db",
    )
    if os.path.exists(demo):
        return f"sqlite:///{demo}"

    return "postgresql://postgres@localhost:5432/india_air_quality"


DB_URL = _default_db_url()

PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "changepoint_prior_scale": 0.05,
}

TRAIN_CUTOFF = "2023-01-01"
FORECAST_YEARS = 6
FORECAST_PERIODS = 365 * FORECAST_YEARS

AQI_THRESHOLDS = {
    "moderate": 100,
    "poor": 200,
}

CITIES = ["Hyderabad", "Delhi", "Mumbai", "Bengaluru", "Chennai", "Kolkata"]

BASE_AQI = {
    "Delhi": 180,
    "Hyderabad": 95,
    "Mumbai": 110,
    "Bengaluru": 85,
    "Chennai": 90,
    "Kolkata": 140,
}

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
