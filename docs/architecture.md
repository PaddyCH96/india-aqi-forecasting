# Architecture

## Overview

India Air Quality Forecasting is a data science project that ingests AQI data for Indian cities, trains Prophet forecasting models, and surfaces predictions through interactive Streamlit dashboards and a REST API. It is designed for urban planners, real estate analysts, and policy makers.

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Data Sources                       │
│  ┌────────────┐   ┌─────────────────┐               │
│  │ OpenAQ API │   │ Open-Meteo API  │               │
│  └──────┬─────┘   └────────┬────────┘               │
│         │                  │                        │
│         ▼                  ▼                        │
│  ┌──────────────────────────────────────┐           │
│  │      fetch_openaq.py /               │           │
│  │      fetch_recent_aqi.py             │           │
│  └────────────────┬─────────────────────┘           │
│                   │                                │
│                   ▼                                │
│  ┌──────────────────────────────────────┐           │
│  │         PostgreSQL (city_day)         │           │
│  └────┬─────────────────┬───────────────┘           │
│       │                 │                           │
│       ▼                 ▼                           │
│  ┌──────────┐   ┌───────────────┐                   │
│  │ Streamlit │   │    FastAPI    │                   │
│  │Dashboard  │   │    (REST)     │                   │
│  └──────────┘   └───────┬───────┘                   │
│                         │                           │
│                         ▼                           │
│  ┌──────────────────────────────────────┐           │
│  │    External consumers (CI, alerts,   │           │
│  │    frontend integrations)            │           │
│  └──────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

### Batch Processing

```
PostgreSQL ──► multi_city_pipeline.py ──► outputs/multi_city_results.csv
            ──► validate_prophet.py    ──► outputs/ (charts + model_config.json)
            ──► seed_data.py           ──► synthetic data on first run
```

## Component Details

### `lib/` — Shared Library
Single source of truth for all business logic. Separated from scripts during a prior refactor to eliminate ~70% code duplication.

| Module | Responsibility |
|--------|---------------|
| `config.py` | Centralized constants (DB URL, Prophet params, AQI thresholds, city list) |
| `db.py` | SQLAlchemy engine + parameterized SQL queries (no f-strings) |
| `models.py` | Prophet model factory: `create_model()`, `train_and_forecast()`, `train_and_validate()` |
| `metrics.py` | MAPE, RMSE, MAE computation + `classify_model_quality()` |
| `aqi.py` | PM2.5 → AQI conversion (US EPA breakpoints) + synthetic data generation |
| `charts.py` | Reusable matplotlib charts (history, forecast, validation, scatter, comparison) |
| `logging.py` | Structured logging with timestamps to stdout |
| `utils.py` | Retry decorator with exponential backoff, DB URL validation |

### `scripts/` — Applications

| Script | Type | Description |
|--------|------|-------------|
| `dashboard_final.py` | Streamlit | 3-tab dashboard (History, Forecast, Validation) |
| `dashboard_complete.py` | Streamlit | 4-tab dashboard adds Multi-City Comparison |
| `api.py` | FastAPI | REST endpoints: `/cities`, `/forecast/{city}`, `/validate/{city}`, `/health` |
| `multi_city_pipeline.py` | CLI | Batch Prophet across all eligible cities |
| `validate_prophet.py` | CLI | 3-model comparison (full, pre-COVID, skip-COVID) + cross-validation |
| `fetch_openaq.py` | CLI | OpenAQ API ingestion with synthetic data fallback |
| `fetch_recent_aqi.py` | CLI | Open-Meteo API ingestion for recent data |
| `seed_data.py` | CLI | Bootstrap database from CSV or synthetic data |

### `tests/` — Test Suite
62 unit tests across `test_metrics.py`, `test_aqi.py`, `test_models.py`. Pure computation tests require no database. Mock-based tests in `test_db.py` verify query logic.

## Data Flow

1. **Ingestion:** `fetch_openaq.py` or `fetch_recent_aqi.py` pull PM2.5 data from free APIs → CSV → PostgreSQL.
2. **Storage:** `city_day` table with columns: `date`, `city`, `aqi`, `pm2_5`, `location` (optional).
3. **Forecasting:** Models read from PostgreSQL via `lib/db.py`, train Prophet with yearly seasonality, produce 6-year forecast with 95% confidence intervals.
4. **Presentation:** Streamlit dashboards render matplotlib charts; FastAPI returns JSON for programmatic access.
5. **Fallback:** If APIs fail, `generate_synthetic_aqi()` produces realistic AQI data with seasonal/weekly/noise patterns.

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.10+ |
| Forecasting | Prophet (Meta) | 1.3+ |
| Dashboard | Streamlit | 1.56+ |
| API | FastAPI + Uvicorn | 0.110+ |
| Data | pandas, numpy | 3.0+, 2.4+ |
| Charts | matplotlib | 3.10+ |
| Database | PostgreSQL via SQLAlchemy + psycopg2 | 16, 2.0+, 2.9+ |
| Container | Docker + Docker Compose | latest |
| CI | GitHub Actions | pytest + pytest-cov |

## Deployment Options

- **Local:** `python scripts/dashboard_final.py` with local PostgreSQL
- **Docker:** `docker compose up` starts PostgreSQL, seeds data, launches dashboard
- **With API:** `docker compose --profile api up` adds REST API on port 8000
