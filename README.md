<p align="center">
  <h1 align="center">India Air Quality Forecasting</h1>
  <p align="center">
    <strong>700K+ records · 26 cities · 12 pollutants · 5.5 years</strong><br>
    <em>XGBoost forecasts at 0.8–3.2% MAPE — 10–20× better than seasonal-naive baselines</em>
  </p>
  <p align="center">
    <a href="https://india-aqi-forecasting-ujpesuvlw7zjgysapt8ff8.streamlit.app">
      <img alt="Live demo" src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white"></a>
    <a href="https://github.com/PaddyCH96/india-aqi-forecasting/actions/workflows/test.yml">
      <img alt="Tests" src="https://github.com/PaddyCH96/india-aqi-forecasting/actions/workflows/test.yml/badge.svg"></a>
    <img alt="Tests count" src="https://img.shields.io/badge/tests-145%20passing-brightgreen">
    <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white">
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-yellow"></a>
  </p>
  <p align="center">
    <a href="https://india-aqi-forecasting-ujpesuvlw7zjgysapt8ff8.streamlit.app"><strong>Open the live dashboard</strong></a> ·
    <a href="CASE_STUDY.md">Case Study</a> ·
    <a href="INSIGHTS.md">Key Insights</a> ·
    <a href="DEMO_WALKTHROUGH.md">Walkthrough</a>
  </p>
</p>

---

> **Try it first:** the [live dashboard](https://india-aqi-forecasting-ujpesuvlw7zjgysapt8ff8.streamlit.app) runs on the bundled demo database —
> 29,531 real CPCB daily records across 26 cities and 78,774 hourly readings, with
> six trained XGBoost models. No signup, no setup.

---

## Key Results

| City | XGBoost MAPE | Training Data | Data Quality |
|------|:-----------:|:------------:|:-----------:|
| Bengaluru | **0.8%** | 1,362 days | Excellent |
| Hyderabad | **0.9%** | 1,332 days | Excellent |
| Chennai | **0.9%** | 1,336 days | Very Good |
| Delhi | **1.0%** | 1,451 days | Excellent |
| Mumbai | **2.9%** | 227 days | **Critical gaps** |
| Kolkata | **3.2%** | 206 days | Limited |

**The system beats naive baselines by 10–20×** (Moving Average: 12–25% MAPE, Seasonal Naive: 31–64% MAPE).

---

## Features

- **6-page analytics dashboard** — Executive summary, trends, pollutant drill-down, city deep-dive, data quality, ML forecasting
- **XGBoost + Random Forest models** — 66 features per city (lags, rolling stats, seasonal cycles, pollutant interactions)
- **Data provenance** — Every row tagged as real/synthetic with source tracking
- **REST API** — FastAPI with `/forecast/{city}`, `/validate/{city}`, `/data/freshness`
- **145 tests** across 9 files, 95% coverage, Ruff-clean
- **Runs three ways** — hosted on Streamlit Cloud with zero setup, `docker compose up` for the full PostgreSQL stack, or a bare clone against the bundled SQLite database

---

## Architecture

```
┌──────────────┐    ┌─────────────┐    ┌──────────────────┐
│  CPCB CSVs   │    │  OpenAQ API │    │  Synthetic Data  │
│  250MB, 5 f. │    │  (real-time)│    │  (fallback)      │
└──────┬───────┘    └──────┬──────┘    └────────┬─────────┘
       │                   │                    │
       └───────────────────┼────────────────────┘
                           ▼
              ┌────────────────────────┐
              │   PostgreSQL (5 tables) │
              │  city_measurements      │
              │  city_hourly           │    ← 700k+ rows, provenance-tracked
              │  station_day, stations │
              └───┬────────────────┬───┘
                  │                │
         ┌────────▼───┐    ┌──────▼────────┐
         │  Dashboards │    │  FastAPI API  │
         │  Streamlit  │    │  /forecast    │
         │  :8501      │    │  /validate    │
         └──────┬──────┘    └──────┬────────┘
                │                  │
         ┌──────▼──────────────────▼──────────┐
         │         ML Forecasting Layer        │
         │                                     │
         │   feature_engineering (66 feats)    │
         │   → ml_pipeline (time split)        │
         │   → model_training (XGB, RF, MA)    │
         │   → forecasting_service (inference) │
         └─────────────────────────────────────┘
```

**Data flow:** Raw CSVs/APIs → `seed_data.py` → PostgreSQL → `lib/` processing → Dashboard/API/Forecast

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Models | XGBoost, scikit-learn (Random Forest), Prophet |
| Dashboard | Streamlit + matplotlib |
| API | FastAPI + uvicorn |
| Database | PostgreSQL in Docker; bundled SQLite for the hosted demo (SQLAlchemy) |
| Data | pandas, numpy |
| Infrastructure | Docker, Docker Compose |
| CI/CD | GitHub Actions (pytest, ruff) |
| Testing | pytest, pytest-cov (145 tests) |

---

## Quick Start

```bash
# Docker (easiest — no local PostgreSQL needed)
git clone https://github.com/PaddyCH96/india-aqi-forecasting.git
cd india-aqi-forecasting
docker compose up --build
# Dashboard at http://localhost:8501

# Or with API:
docker compose --profile api up --build
# API docs at http://localhost:8000/docs
```

### Data in a fresh clone

A read-only SQLite database ships with the repo at `data/aqi_demo.db` (9.8 MB):

| Rows | Source | Cities |
|------|--------|--------|
| 29,531 | CPCB daily city aggregates (real) | 26 |
| 9,870 | synthetic fallback, tagged `is_synthetic` | 6 |

With no `AQI_DB_URL` set the app uses that file, so `streamlit run scripts/dashboard.py`
works on a fresh clone with no database server. Set `AQI_DB_URL` to point at PostgreSQL
instead — Docker and the Render blueprint both do.

Real rows are shown by default; synthetic rows are excluded until you tick
*"Include synthetic data"* in the sidebar, or pass `?use_synthetic=true` to the API.
That separation is enforced in SQL, not assumed.

The larger hourly and station-level extracts (`city_hour.csv` at 707,875 rows,
`station_hour.csv` at ~4M) are not committed — they are too large for the repo. The
26-city, 700K-record figures in Key Results come from those full extracts.

**Data attribution:** air quality measurements are published by the Central Pollution
Control Board (CPCB), Government of India.

### Deploy your own (free)

**Streamlit Community Cloud** — free, no card, no expiry:

1. [share.streamlit.io](https://share.streamlit.io) -> sign in with GitHub
2. **New app** -> this repo -> branch `main`
3. Main file path: `scripts/dashboard.py`
4. **Deploy**

No database or secrets to configure — the bundled `data/aqi_demo.db` is used
automatically. Apps sleep after ~7 days idle and wake on the next visit.

**Render** — if you want PostgreSQL and the REST API, `render.yaml` is a Blueprint:
Render dashboard -> New -> Blueprint -> this repo -> Apply. Note that Render's free
PostgreSQL instances expire after 30 days.

### Local Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
createdb india_air_quality
python scripts/seed_data.py     # bootstraps from CSV or generates synthetic
streamlit run scripts/dashboard.py
```

---

## Dashboard Pages

| # | Page | What It Shows |
|---|------|-------------|
| 1 | Executive Summary | National snapshot, KPI cards, city ranking |
| 2 | Historical Trends | Multi-city trends, seasonal decomposition, monthly averages |
| 3 | Pollutant Drill-Down | Per-pollutant distributions, 9×9 correlation matrix, diurnal patterns |
| 4 | City Deep-Dive | Single-city history, year-over-year bars, pollutant summary table |
| 5 | Data Quality | Missing data heatmap, completeness warnings by city |
| 6 | Forecasting | 24h–336h XGBoost forecast with confidence bands + AQI alerts |

---

## Key Insights

- **Delhi is an extreme outlier** — mean AQI 259.5 is 2.7× higher than the next worst city
- **PM2.5 alone predicts AQI with r=0.97** — other pollutants are largely redundant for forecasting
- **Mumbai has a monitoring crisis** — 61% of daily AQI records missing, worst of 26 cities
- **Winter pollution penalty varies by geography** — 2.5× in the north, 1.3× in the south
- **Data quality determines accuracy** — not model choice. Better monitoring > better algorithms

---

## Project Structure

```
├── lib/                 # Shared library (12 modules)
│   ├── config.py        #   Constants
│   ├── db.py            #   Parameterized SQL queries
│   ├── feature_engineering.py  # 66 features per city
│   ├── model_training.py       # 5 model trainers
│   ├── forecasting_service.py  # Inference for dashboard
│   ├── charts.py, analysis.py  # Visualization + EDA
│   └── ...                     # metrics, aqi, models, utils, logging
├── scripts/             # Runnable applications
│   ├── dashboard.py     #   6-page unified dashboard
│   ├── api.py           #   FastAPI REST API
│   ├── seed_data.py     #   Database bootstrap
│   └── ingest_hourly.py #   Hourly data pipeline
├── tests/               # 145 tests
├── docs/                # Architecture, EDA, deployment, ML eval
├── models/              # Six trained XGBoost models (committed, 4.5 MB)
├── CASE_STUDY.md        # Portfolio narrative
├── INSIGHTS.md          # Top 5 findings with evidence
└── Dockerfile + docker-compose.yml
```

---

## Reading Order for Recruiters

1. **[Case Study](CASE_STUDY.md)** — Narrative overview of the project (10 min read)
2. **[Key Insights](INSIGHTS.md)** — Five defensible findings with evidence (5 min read)
3. **[Live dashboard](https://india-aqi-forecasting-ujpesuvlw7zjgysapt8ff8.streamlit.app)** — the running system, no setup required
4. **Code** — `lib/` for core logic, `tests/` for test coverage

---

*Bundled demo database: CPCB daily records 2015-01 to 2020-07 across 26 cities, hourly readings 2019-01 to 2020-07 for six cities, plus a synthetic 2020-07 to 2024-12 series tagged `is_synthetic`. Headline figures (700K+ records, 12 pollutants, 5.5 years) come from the full CPCB extract, which is not committed. Built with open data and open source.*
