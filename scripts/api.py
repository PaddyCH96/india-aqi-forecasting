"""FastAPI REST API for India AQI Forecasting.

Endpoints support a `use_synthetic` query parameter:
- False (default): real CPCB data only (2015-2020)
- True: includes synthetic 2020-2024 data
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from lib.db import get_engine, get_cities_with_recent_data, load_city_data, get_data_freshness
from lib.models import train_and_forecast, train_and_validate
from lib.metrics import evaluate_forecast, classify_model_quality
from lib.config import TRAIN_CUTOFF

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
    freshness = get_data_freshness(engine)
    return {
        "status": "ok",
        **freshness,
    }


@app.get("/cities", response_model=CitiesResponse)
def list_cities(use_synthetic: bool = Query(False)):
    cities = get_cities_with_recent_data(engine, use_synthetic=use_synthetic)
    return {"cities": cities, "count": len(cities)}


@app.get("/forecast/{city}", response_model=ForecastResponse)
def get_forecast(city: str, use_synthetic: bool = Query(False)):
    df = load_city_data(engine, city, use_synthetic=use_synthetic)
    if df.empty:
        raise HTTPException(404, f"No data found for city: {city}")

    _, forecast = train_and_forecast(df)

    cutoff_year = 2024 if use_synthetic else 2020
    max_date = df["ds"].max()

    current_avg = (
        df[df["ds"].dt.year == cutoff_year]["y"].mean()
        if len(df[df["ds"].dt.year == cutoff_year]) > 0
        else None
    )
    pred_2030 = forecast[forecast["ds"].dt.year == 2030]["yhat"].mean()

    forecast_out = forecast[forecast["ds"] > max_date]

    return {
        "city": city,
        "data_points": len(df),
        "current_avg": round(current_avg, 1) if current_avg else None,
        "pred_2030": round(pred_2030, 1),
        "trend": "improving" if pred_2030 and current_avg and pred_2030 < current_avg else "worsening",
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
    df = load_city_data(engine, city, use_synthetic=use_synthetic)
    if df.empty:
        raise HTTPException(404, f"No data found for city: {city}")

    train = df[df["ds"] < TRAIN_CUTOFF]
    test = df[df["ds"] >= TRAIN_CUTOFF]

    if len(test) < 30:
        raise HTTPException(400, f"Insufficient test data for {city}: {len(test)} days")

    _, _, results = train_and_validate(train, test)
    metrics = evaluate_forecast(results)
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
    return get_data_freshness(engine)
