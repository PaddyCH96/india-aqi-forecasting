# India Air Quality: End-to-End Data & ML Portfolio Project

**Forecast AQI across 26 Indian cities with 0.8–3.2% MAPE using XGBoost + engineered features. Includes EDA, interactive dashboard, REST API, and Docker deployment.**

[View Case Study →](CASE_STUDY.md) · [Key Insights →](INSIGHTS.md) · [GitHub →](https://github.com/PaddyCH96/india-aqi-forecasting)

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
- **144 tests** across 9 files, 95% coverage, Ruff-clean
- **Docker deployment** — 4-service compose (PostgreSQL, seed, dashboard, API)

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
| Database | PostgreSQL (SQLAlchemy + psycopg2) |
| Data | pandas, numpy |
| Infrastructure | Docker, Docker Compose |
| CI/CD | GitHub Actions (pytest, ruff) |
| Testing | pytest, pytest-cov (144 tests) |

---

## Quick Start

```bash
# Docker (easiest — no local PostgreSQL needed)
git clone https://github.com/PaddyCH96/india-aqi-forecasting.git
cd india-air-quality
docker compose up --build
# Dashboard at http://localhost:8501

# Or with API:
docker compose --profile api up --build
# API docs at http://localhost:8000/docs
```

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
├── tests/               # 144 tests
├── docs/                # Architecture, EDA, deployment, ML eval
├── models/              # Trained models (*.pkl, generated)
├── CASE_STUDY.md        # Portfolio narrative
├── INSIGHTS.md          # Top 5 findings with evidence
└── Dockerfile + docker-compose.yml
```

---

## Reading Order for Recruiters

1. **[Case Study](CASE_STUDY.md)** — Narrative overview of the project (10 min read)
2. **[Key Insights](INSIGHTS.md)** — Five defensible findings with evidence (5 min read)
3. **Dashboard** — Run `streamlit run scripts/dashboard.py` to see it live
4. **Code** — `lib/` for core logic, `tests/` for test coverage

---

*CPCB data 2015–2020 · 26 cities · 12 pollutants · 5.5 years. Built with open data and open source.*
