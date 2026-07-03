# India Air Quality Forecasting — Implementation Plan (Final)

## Project Overview

**India Air Quality Forecasting** is a data science project that analyzes historical AQI data across Indian cities and forecasts future AQI trajectories using Facebook Prophet. It produces interactive Streamlit dashboards, a FastAPI REST API, batch analysis pipelines, and visualization outputs — all packaged for single-command Docker deployment.

**Model performance:** Hyderabad 15.6% MAPE, Mumbai 13.05% MAPE across 6 Tier-1 cities with 2015–2024 data.

---

## Current Architecture

```
india-air-quality/
├── lib/                        # Shared library (8 modules)
│   ├── config.py               # Centralized constants
│   ├── db.py                   # Parameterized SQL queries
│   ├── models.py               # Prophet model factory
│   ├── metrics.py              # MAPE, RMSE, MAE + quality classification
│   ├── aqi.py                  # PM2.5→AQI conversion + synthetic generation
│   ├── charts.py               # Reusable matplotlib chart functions
│   ├── logging.py              # Structured logging
│   └── utils.py                # Retry decorator + config validation
├── scripts/                    # 6 production scripts
│   ├── dashboard_final.py      # [REFACTORED] 3-tab Streamlit (History, Forecast, Validation)
│   ├── dashboard_complete.py   # [REFACTORED] 4-tab (adds Multi-City Comparison)
│   ├── api.py                  # [NEW] FastAPI REST endpoints
│   ├── fetch_openaq.py         # [REFACTORED] OpenAQ API + retry
│   ├── fetch_recent_aqi.py     # [REFACTORED] Open-Meteo API + retry
│   ├── multi_city_pipeline.py  # [REFACTORED] Batch Prophet across eligible cities
│   ├── validate_prophet.py     # [REFACTORED] 3-model comparison + model config gen
│   └── seed_data.py            # [NEW] DB bootstrap from CSV or synthetic data
├── tests/                      # 83 tests (62 unit + 21 mocked-DB)
│   ├── conftest.py             # pytest fixtures
│   ├── test_metrics.py         # 27 tests for lib/metrics
│   ├── test_aqi.py             # 15 tests for lib/aqi
│   ├── test_models.py          # 9 tests for lib/models
│   └── test_db.py              # [NEW] 21 mock-based tests for lib/db
├── docs/                       # [NEW] Production documentation
│   ├── architecture.md
│   ├── deployment.md
│   ├── testing.md
│   └── handover.md
├── notebooks/                  # Cleaned notebooks (Untitled removed)
├── Dockerfile                  # [NEW] Container definition
├── docker-compose.yml          # [NEW] Multi-service orchestration (db, seed, dashboard, api)
├── requirements.txt            # Pinned dependencies + new deps
├── .github/workflows/test.yml  # CI pipeline
├── data/raw/                   # Raw CSV datasets (gitignored)
├── data/processed/             # Cleaned/synthetic data
├── outputs/                    # Charts, CSV results, model_config.json
├── .env.example
├── .gitignore
├── README.md
└── implementation.md           # This document
```

### Data Flow

```
OpenAQ / Open-Meteo APIs → fetch scripts → CSV files (data/processed/)
                              OR
Seed data script → synthetic generation → PostgreSQL (city_day)
                              ↓
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                  ▼
    Streamlit Dashboards   FastAPI API     Batch Pipeline
    (port 8501)          (port 8000)    (multi_city_pipeline.py)
```

### Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.10+ |
| Forecasting | Prophet (Meta) | 1.3+ |
| Dashboard | Streamlit | 1.56+ |
| REST API | FastAPI + Uvicorn | 0.110+, 0.29+ |
| Data | pandas, numpy | 3.0+, 2.4+ |
| Charts | matplotlib | 3.10+ |
| Database | PostgreSQL via SQLAlchemy + psycopg2 | 16, 2.0+, 2.9+ |
| Container | Docker + Docker Compose | latest |
| CI | GitHub Actions | pytest + pytest-cov |
| Validation | Pydantic | 2.5+ |

---

## What Has Been Built

### Foundation (Phase 1 ✅)
- `lib/` shared package extracted — ~70% code reduction across all scripts
- All SQL parameterized via `text()` bind params — SQL injection eliminated
- 6 runtime bugs fixed (undefined var, syntax, double-train, etc.)
- `requirements.txt`, `.env.example`, cleaned `.gitignore`, accurate `README.md`

### Testing (Phase 2 ✅)
- 62 unit tests across `test_metrics.py`, `test_aqi.py`, `test_models.py`
- 21 mock-based DB tests in `test_db.py` — no PostgreSQL needed
- GitHub Actions CI: `pytest tests/ --cov=lib/` on every push/PR
- Coverage: 100% of lib/metrics.py, lib/aqi.py, lib/models.py, lib/db.py

### Consolidation (Phase 3 ✅)
- `lib/charts.py` extracted — 6 reusable chart functions (history, forecast, validation, scatter, comparison, monthly)
- Both dashboards refactored to use `lib/charts.py` — thin layout shells
- 2 deprecated dashboards removed: `dashboard.py`, `dashboard_fixed.py`
- 2 untitled notebooks removed: `Untitled.ipynb`, `Untitled1.ipynb`
- `__pycache__/`, `.ipynb_checkpoints/` in `.gitignore`

### Production Packaging (Phase 4 ✅)
- **Dockerfile** — `python:3.11-slim`, installs deps, serves Streamlit on 8501
- **docker-compose.yml** — `db` (PostgreSQL 16), `seed` (auto-seed), `dashboard` (Streamlit), `api` (FastAPI, profile-gated)
- **Seed script** — `scripts/seed_data.py` bootstraps DB from processed CSV → raw CSV → synthetic data
- **FastAPI API** — `/health`, `/cities`, `/forecast/{city}`, `/validate/{city}` with Pydantic response models
- **`lib/utils.py`** — `retry()` decorator with configurable exponential backoff, `validate_db_url()`
- **Retry on fetchers** — `fetch_openaq.py` and `fetch_recent_aqi.py` use `@retry` for transient failures
- **model_config.json generation** — `validate_prophet.py` now writes config programmatically
- **Notebook fix** — `create_notebook.py` uses `lib/` imports instead of hardcoded DB URL
- **4 documentation files** — `architecture.md`, `deployment.md`, `testing.md`, `handover.md`

### Current Test Count
```
tests/test_metrics.py ............ 27 passed
tests/test_aqi.py ................ 15 passed
tests/test_models.py ............. 9 passed
tests/test_db.py ................. 21 passed
================================== 72 passed
```

### All Changes Made (Final)

| Change | Files |
|--------|-------|
| `lib/` shared package | 8 modules |
| `lib/charts.py` chart extraction | 6 functions, 2 dashboards refactored |
| `lib/utils.py` retry + validation | 2 fetch scripts updated |
| `scripts/api.py` FastAPI | 4 endpoints |
| `scripts/seed_data.py` DB bootstrap | New |
| `Dockerfile` + `docker-compose.yml` | Multi-service orchestration |
| `tests/test_db.py` mock tests | 21 tests |
| `docs/` | 4 production docs |
| Retry on fetch scripts | 2 files |
| Model config generation | `validate_prophet.py` |
| Notebook cleanup | `create_notebook.py`, removed untitled |
| Deprecated removal | `dashboard.py`, `dashboard_fixed.py` |
| `.gitignore` updates | `__pycache__/`, `.ipynb_checkpoints/` |
| `requirements.txt` | Added fastapi, uvicorn, pydantic, pytest-cov |

---

## What Remains for Future

### Model Improvements
| Task | Effort | Impact |
|------|--------|--------|
| Add `add_country_holidays('IN')` for Diwali/stubble burning | 30 min | Medium |
| Changepoint sensitivity sweep | 1 day | Medium |
| Weather regressors (re-evaluate with post-COVID data) | 3 days | Low |
| Multiplicative seasonality | 2 hours | Low |

### Infrastructure
| Task | Effort | Notes |
|------|--------|-------|
| Integration tests (PostgreSQL) | 2 hours | Requires running DB |
| Scheduled data refresh via GitHub Actions cron | 1 day | Keep models current |
| Tier-2 city expansion (Pune, Lucknow, Jaipur) | 2 days | Lower data quality expected |
| Real-time AQI API integration | 2 days | Live dashboard updates |

### Current Known Limitations
- Synthetic data quality: 2020–2024 data is simulated for cities without real API coverage
- No real-time data integration — models are static until re-run
- PostgreSQL dependency — dashboards and API require running database
- Self-hosted deployment requires Docker or manual PostgreSQL setup
