# CLONE.md — New Developer Onboarding

Welcome! This guide gets you from zero to running the India Air Quality Forecasting project in under 10 minutes.

## Prerequisites

Check you have these installed:

```bash
python3 --version        # 3.11+
psql --version           # PostgreSQL client
docker --version         # (optional, for Docker deployment)
docker compose version   # (optional)
```

## 1. Clone

```bash
git clone https://github.com/PaddyCH96/india-aqi-forecasting.git
cd india-air-quality
```

## 2. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Database Setup

### Option A: Local PostgreSQL (Recommended)

```bash
# Create the database
createdb india_air_quality

# Configure connection
cp .env.example .env
# Default: postgresql://postgres:postgres@localhost:5432/india_air_quality
# Edit .env if your PostgreSQL uses different credentials
```

### Option B: Docker PostgreSQL

```bash
docker compose up db -d
# PostgreSQL at localhost:5432, user/pass: postgres/postgres
# Database india_air_quality is auto-created
```

## 4. Seed Data

```bash
python scripts/seed_data.py
```

This will:
- Look for CSV files in `data/raw/`
- If none found, generate synthetic AQI data for 6 cities (2020–2024)
- Insert everything into the `city_day` table

## 5. Verify It Works

```bash
# Run the full test suite
pytest tests/ -v

# Expected output: 144 passed
```

## 6. Run the Dashboard

```bash
streamlit run scripts/dashboard.py
# Open http://localhost:8501
```

You should see a 6-page dashboard with Executive Summary, Historical Trends,
Pollutant Drill-Down, City Deep-Dive, Data Quality, and Forecasting views.

## 7. Try the API (Optional)

```bash
uvicorn scripts.api:app --reload --port 8000
```

Then in another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/cities
curl http://localhost:8000/forecast/Delhi
curl http://localhost:8000/validate/Mumbai
```

API docs available at http://localhost:8000/docs

## 8. Run the Forecasting Pipeline (Optional)

```bash
# Multi-city batch forecast
python scripts/multi_city_pipeline.py

# Model comparison and validation
python scripts/validate_prophet.py
```

## Docker Quick Start (Skip Steps 2-4)

If you have Docker Compose:

```bash
# Start everything
docker compose --profile api up --build

# Dashboard: http://localhost:8501
# API: http://localhost:8000/docs
```

## What's Included

| Layer | Tech | Purpose |
|-------|------|---------|
| Shared library | `lib/` | 12 modules: config, db, models, metrics, aqi, charts, logging, utils, analysis, feature_engineering, ml_pipeline, model_training, model_evaluation, forecasting_service |
| Dashboards | Streamlit | Interactive AQI visualization |
| API | FastAPI | REST endpoints for forecasts and validation |
| ETL | Python scripts | Data ingestion from OpenAQ, Open-Meteo, CSV |
| Forecasting | Prophet | Time-series AQI predictions through 2030 |
| Tests | pytest | 144 tests across 9 files, 95% coverage |
| CI | GitHub Actions | Automated test run on push |
| Deployment | Docker Compose | 4-service stack with PostgreSQL |

## Project Structure Quick Reference

| Directory | Purpose |
|-----------|---------|
| `lib/` | Shared code — never modify outside this directory for core logic |
| `scripts/` | Runnable entry points (dashboards, API, pipelines) |
| `tests/` | All tests, mirrors `lib/` structure |
| `docs/` | Architecture, deployment, testing, handover docs |
| `data/raw/` | Raw CSVs — gitignored, regenerate via seed script |
| `data/processed/` | Generated data — gitignored |
| `outputs/` | Charts and forecasts — gitignored |

## Troubleshooting

**Database connection refused**
```
Ensure PostgreSQL is running. Check with: pg_isready
Verify .env DB_URL is correct.
Default: postgresql://postgres:postgres@localhost:5432/india_air_quality
```

**No module named 'lib'**
```
Run from project root. Activate venv: source venv/bin/activate
Reinstall: pip install -e .
```

**Tests fail with database errors**
```
Tests use mocks — they shouldn't need a real database.
Run: pytest tests/ -v -k "not test_db"
If test_db fails, ensure pytest-mock is installed (it's in requirements.txt).
```

**Streamlit shows no data**
```
Run seed_data.py first. Verify data with: psql -d india_air_quality -c "SELECT COUNT(*) FROM city_day;"
```

## Need Help?

- Architecture docs: `docs/architecture.md`
- Deployment guide: `docs/deployment.md`
- Testing strategy: `docs/testing.md`
- Full handover: `docs/handover.md`
- Release report: `release_report.md`
