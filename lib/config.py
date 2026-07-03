import os

DB_URL = os.getenv(
    "AQI_DB_URL",
    "postgresql://postgres@localhost:5432/india_air_quality"
)

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
