"""FastAPI REST API for India AQI Forecasting.

Endpoints support a `use_synthetic` query parameter:
- False (default): real CPCB data only (2015-2020)
- True: includes synthetic 2020-2024 data
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import logging

from lib.db import get_engine, get_cities_with_recent_data, load_city_data, get_data_freshness
from lib.models import train_and_forecast, train_and_validate
from lib.metrics import evaluate_forecast, classify_model_quality
from lib.config import TRAIN_CUTOFF

logger = logging.getLogger(__name__)

app = FastAPI(title="India AQI Forecasting API", version="2.0.0")
engine = get_engine()


class CityItem(BaseModel):
    city: str


class CitiesResponse(BaseModel):
    cities: list[str]
    count: int


class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    city: str
    data_points: int
    current_avg: float | None
    pred_2030: float | None
    trend: str
    synthetic_data_used: bool
    historical: list[ForecastPoint]
    forecast: list[ForecastPoint]


class ValidationResponse(BaseModel):
    city: str
    mape: float
    rmse: float
    mae: float
    quality: str
    train_days: int
    test_days: int
    synthetic_data_used: bool


class HealthResponse(BaseModel):
    status: str
    total_rows: int
    real_rows: int
    synthetic_rows: int
    cities: int
    latest_real_date: str | None
    data_sources: dict


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint with data freshness info."""
    freshness = get_data_freshness(engine)
    return {
        "status": "ok",
        **freshness,
    }


@app.get("/cities", response_model=CitiesResponse)
def list_cities(use_synthetic: bool = Query(False)):
    """List cities with recent data available."""
    cities = get_cities_with_recent_data(engine, use_synthetic=use_synthetic)
    return {"cities": cities, "count": len(cities)}


@app.get("/forecast/{city}", response_model=ForecastResponse)
def get_forecast(city: str, use_synthetic: bool = Query(False)):
    """Generate AQI forecast for a city.
    
    Raises:
        404: City not found or no data available
        500: Forecasting failed
    """
    df = load_city_data(engine, city, use_synthetic=use_synthetic)
    if df.empty:
        raise HTTPException(404, f"No data found for city: {city}")

    # Attempt forecasting with error handling
    try:
        _, forecast = train_and_forecast(df)
    except Exception as e:
        logger.error(f"Forecasting failed for {city}: {str(e)}")
        raise HTTPException(500, f"Forecasting failed for {city}: {str(e)}")

    cutoff_year = 2024 if use_synthetic else 2020
    max_date = df["ds"].max()

    # Safely compute current average
    current_data = df[df["ds"].dt.year == cutoff_year]["y"]
    current_avg = None
    if len(current_data) > 0:
        current_avg = float(current_data.mean())
        if pd.isna(current_avg):
            current_avg = None

    # Safely compute 2030 prediction
    forecast_2030 = forecast[forecast["ds"].dt.year == 2030]["yhat"]
    if len(forecast_2030) == 0:
        raise HTTPException(500, f"No 2030 forecast available for {city}")
    
    pred_2030 = float(forecast_2030.mean())
    if pd.isna(pred_2030):
        raise HTTPException(500, f"2030 forecast is invalid (NaN) for {city}")

    # Determine trend with null safety
    if current_avg is not None:
        trend = "improving" if pred_2030 < current_avg else "worsening"
    else:
        trend = "unknown"

    forecast_out = forecast[forecast["ds"] > max_date]

    return {
        "city": city,
        "data_points": len(df),
        "current_avg": round(current_avg, 1) if current_avg else None,
        "pred_2030": round(pred_2030, 1),
        "trend": trend,
        "synthetic_data_used": use_synthetic,
        "historical": [
            {"ds": str(r["ds"].date()), "yhat": round(r["y"], 1),
             "yhat_lower": round(r["y"], 1), "yhat_upper": round(r["y"], 1)}
            for _, r in df.tail(1000).iterrows()
        ],
        "forecast": [
            {"ds": str(r["ds"].date()), "yhat": round(r["yhat"], 1),
             "yhat_lower": round(r["yhat_lower"], 1),
             "yhat_upper": round(r["yhat_upper"], 1)}
            for _, r in forecast_out.iterrows()
        ],
    }


@app.get("/validate/{city}", response_model=ValidationResponse)
def get_validation(city: str, use_synthetic: bool = Query(False)):
    """Validate model performance on historical data.
    
    Raises:
        404: City not found or no data available
        400: Insufficient data for validation
        500: Model validation failed
    """
    df = load_city_data(engine, city, use_synthetic=use_synthetic)
    if df.empty:
        raise HTTPException(404, f"No data found for city: {city}")

    train = df[df["ds"] < TRAIN_CUTOFF]
    test = df[df["ds"] >= TRAIN_CUTOFF]

    if len(test) < 30:
        raise HTTPException(400, f"Insufficient test data for {city}: {len(test)} days")

    # Attempt model training with error handling
    try:
        _, _, results = train_and_validate(train, test)
    except Exception as e:
        logger.error(f"Model validation failed for {city}: {str(e)}")
        raise HTTPException(500, f"Model validation failed for {city}: {str(e)}")

    metrics = evaluate_forecast(results)
    
    # Handle NaN metrics
    if pd.isna(metrics["mape"]):
        raise HTTPException(500, f"Invalid MAPE for {city} (all actual values may be zero)")
    
    quality, _ = classify_model_quality(metrics["mape"])

    return {
        "city": city,
        "mape": round(metrics["mape"], 2),
        "rmse": round(metrics["rmse"], 2),
        "mae": round(metrics["mae"], 2),
        "quality": quality,
        "train_days": len(train),
        "test_days": len(test),
        "synthetic_data_used": use_synthetic,
    }


@app.get("/data/freshness")
def data_freshness():
    """Get data freshness and source information."""
    return get_data_freshness(engine)
