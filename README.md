# India Air Quality Forecasting & Urban Risk Analysis

## Problem
Urban air quality in India is highly volatile and poorly forecasted at a city level. This creates downstream risks for public health, real estate valuation, and policy planning.

## Objective
Build a data-driven pipeline to:
- Analyze historical AQI trends across Indian cities
- Forecast future air quality using time-series modeling
- Surface insights relevant to urban planning and risk assessment

## What This Project Does
1. Data ingestion and preprocessing of AQI datasets
2. Exploratory analysis of seasonal and city-level trends
3. Time-series forecasting using Prophet
4. Visualization of long-term AQI trajectories (up to 2030)
5. Comparative analysis across cities

## Key Insights
- Clear seasonal AQI spikes across most Tier-1 cities
- Long-term upward AQI trend in high-growth urban zones
- Significant variance across cities → localized policy required

## Tech Stack
- Python (pandas, numpy, matplotlib)
- Prophet (time-series forecasting)
- Streamlit (interactive dashboards)
- FastAPI + Uvicorn (REST API)
- PostgreSQL (data storage)
- Docker + Docker Compose (deployment)
- GitHub Actions (CI)

## Prerequisites
- Python 3.10+
- PostgreSQL with `india_air_quality` database and `city_day` table
- Data files in `data/raw/` (see `data/raw/` structure)

## Quick Start (Docker)

```bash
docker compose up --build
# Dashboard: http://localhost:8501
```

With REST API:

```bash
docker compose --profile api up --build
# Dashboard: http://localhost:8501
# API docs: http://localhost:8000/docs
```

## Local Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure database (optional, defaults to localhost)
cp .env.example .env
# Edit .env with your database URL if needed

# Create database and seed data
createdb india_air_quality
python scripts/seed_data.py
```

## How to Run

```bash
# Recommended dashboard (3 tabs: History, Forecast, Validation)
streamlit run scripts/dashboard_final.py

# Feature-rich dashboard (4 tabs: History, Forecast, Validation, Multi-City Comparison)
streamlit run scripts/dashboard_complete.py

# REST API
uvicorn scripts.api:app --reload --port 8000

# Fetch real AQI data from OpenAQ API
python scripts/fetch_openaq.py

# Run multi-city batch forecasting pipeline
python scripts/multi_city_pipeline.py

# Run model validation with cross-validation
python scripts/validate_prophet.py

# Run tests
pytest tests/ -v --cov=lib/
```

## Project Structure
```
india-air-quality/
├── lib/                # Shared library (config, db, models, metrics, aqi)
├── scripts/            # Production-ready scripts and dashboards
├── notebooks/          # Jupyter notebooks for exploration
├── sql/                # Analytical SQL queries
├── data/raw/           # Raw CSV datasets
├── data/processed/     # Processed/generated data
├── outputs/            # Charts, forecasts, model config
├── requirements.txt
├── .env.example
└── README.md
```

## Scripts Overview
| Script | Purpose |
|--------|---------|
| `dashboard_final.py` | Recommended 3-tab Streamlit dashboard |
| `dashboard_complete.py` | Feature-rich 4-tab dashboard with multi-city comparison |
| `fetch_openaq.py` | Fetch AQI data from OpenAQ API with synthetic fallback |
| `fetch_recent_aqi.py` | Fetch AQI data from Open-Meteo API |
| `multi_city_pipeline.py` | Batch forecast processing for all eligible cities |
| `validate_prophet.py` | Model validation with 3-model comparison & cross-validation |

## Limitations
- Historical data limited (majority pre-2020)
- Forecast accuracy constrained by lack of external regressors
- No real-time data integration
- PostgreSQL dependency for dashboard functionality

## Future Improvements
- Integrate live AQI APIs
- Add weather + traffic regressors
- Build a city-level risk scoring system
- Deploy as an API/dashboard
- Add automated test suite

## Why This Matters
This project demonstrates how air quality data can be transformed into:
- Urban risk indicators
- Policy-relevant insights
- Decision support tools for real estate and infrastructure planning
- Hyderabad shows a consistent upward AQI trend post-2018, indicating increasing environmental risk in high-growth urban corridors
