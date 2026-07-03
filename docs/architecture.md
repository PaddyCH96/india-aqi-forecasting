# System Architecture

## Data Flow Overview

```
INGESTION ──► STORAGE ──► PROCESSING ──► ANALYTICS ──► ML ──► PRESENTATION
```

---

## 1. Ingestion Layer

Three data sources converge into a unified pipeline:

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                          │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  CPCB CSVs   │  │  OpenAQ API  │  │  Synthetic Generator  │  │
│  │  (offline)   │  │  (real-time) │  │  (fallback)          │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         │         scripts/fetch_openaq.py        │              │
│         │         scripts/fetch_recent_aqi.py    │              │
│         └─────────────────┼──────────────────────┘              │
│                           ▼                                     │
│                   scripts/seed_data.py                          │
│                   (auto-detects available sources,              │
│                    prefers real CSVs, falls back to             │
│                    synthetic, sets provenance flags)             │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
```

**Data provenance:** Every ingested row receives:
- `is_synthetic: bool` — true if machine-generated
- `data_source: str` — "CPCB", "OpenAQ", or "synthetic"
- `ingested_at: timestamp` — when it entered the system

---

## 2. Storage Layer

```
┌──────────────────────────────────────────────────────────────────┐
│                       STORAGE LAYER                             │
│                     PostgreSQL (india_air_quality)               │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  city_measurements (39,401 rows)                        │   │
│  │  ─ city, date, aqi, pm2_5, pm10, no2, co, o3, ...     │   │
│  │  ─ is_synthetic, data_source, ingested_at               │   │
│  │    29,531 real CPCB (2015–2020)                         │   │
│  │    9,870 synthetic (2020–2024, AQI-only)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  city_hourly_measurements (707,875 rows)                 │   │
│  │  ─ city, datetime, pm2_5, pm10, no2, co, o3, aqi       │   │
│  │  ─ Provenance-tracked, CPCB data only                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  station_day (108,035 rows)  │  stations (230 rows)     │   │
│  │  Station-level daily data   │  Station metadata         │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
```

**Real vs. Synthetic separation:** All `lib/db.py` queries default to `is_synthetic=FALSE`. Synthetic data must be explicitly opted into via `use_synthetic=True` parameter. The dashboard shows a warning badge when synthetic data is enabled.

---

## 3. Processing & Analytics Layer

```
┌──────────────────────────────────────────────────────────────────┐
│                     PROCESSING LAYER                            │
│                     lib/ (shared modules)                       │
│                                                                │
│  lib/config.py      → Constants, DB URL, model params          │
│  lib/db.py          → Parameterized SQL queries, provenance     │
│  lib/aqi.py         → PM2.5→AQI conversion, synthetic gen       │
│  lib/utils.py       → @retry decorator, URL validation          │
│  lib/logging.py     → Structured stdout logging                 │
│                                                                │
│                     ANALYTICS LAYER                            │
│                                                                │
│  lib/analysis.py    → 15 EDA functions:                        │
│     city_ranking(), aqi_distribution(), missing_heatmap(),     │
│     correlation_matrix(), summer_winter_comparison(),          │
│     year_over_year(), monthly_trends(), diurnal patterns       │
│                                                                │
│  lib/charts.py      → 19 chart functions:                      │
│     plot_multi_city_trends, city_ranking, correlation_heatmap, │
│     diurnal_pattern, seasonal_box, history, distribution,      │
│     aqi_category_bars, missing_heatmap                         │
│                                                                │
│  lib/metrics.py     → calc_mape(), calc_rmse(), calc_mae(),    │
│                       evaluate_forecast(), classify_quality()  │
└──────────────────────────────────────────────────────────────────┘
```

**Key design decision:** `lib/` is the single source of truth. All scripts import from `lib/` rather than duplicating logic. This eliminated ~70% code duplication from earlier versions.

---

## 4. ML Layer

```
┌──────────────────────────────────────────────────────────────────┐
│                       ML FORECASTING LAYER                      │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Feature Engineering (lib/feature_engineering.py)    │   │
│  │                                                         │   │
│  │  add_lag_features()        → lags at 1/2/3/7 days       │   │
│  │  add_rolling_features()    → rolling mean/std/max       │   │
│  │                             (3/7/30 day windows)        │   │
│  │  add_seasonal_features()   → sin/cos month, dow,       │   │
│  │                             is_weekend, quarter         │   │
│  │  add_interaction_features()→ PM2.5/PM10 ratio,         │   │
│  │                             NO2×CO product, etc.       │   │
│  │  add_city_normalization()  → z-score relative to       │   │
│  │                             city mean                  │   │
│  │                                                         │   │
│  │  TOTAL: 66 features per city                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  2. Dataset Builder (lib/ml_pipeline.py)               │   │
│  │                                                         │   │
│  │  build_ml_dataset()  → Load from DB + engineer features │   │
│  │  time_based_split()  → Train: ≤2019, Test: ≥2019        │   │
│  │  prepare_ml_data()   → Drop sparse features >80% NaN,  │   │
│  │                         median impute remaining         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3. Model Training (lib/model_training.py)              │   │
│  │                                                         │   │
│  │  train_xgboost()         ← Primary model                │   │
│  │  train_random_forest()   ← Secondary (baseline)         │   │
│  │  train_moving_average()  ← Naive baseline (7-day)      │   │
│  │  train_seasonal_naive()  ← Naive baseline (365-day)    │   │
│  │  train_prophet()         ← Time-series baseline         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  4. Evaluation (lib/model_evaluation.py)                │   │
│  │                                                         │   │
│  │  evaluate_all_models()  → Side-by-side comparison       │   │
│  │  cross_city_evaluation()→ Table across all cities       │   │
│  │  error_analysis()       → Breakdown by AQI band        │   │
│  │  seasonal_error_analysis()→ Winter/summer/monsoon      │   │
│  │  model_rankings_by_metric()→ Best model per city       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                    │
│                           ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  5. Inference (lib/forecasting_service.py)              │   │
│  │                                                         │   │
│  │  train_and_save_model() → Train + pickle to disk        │   │
│  │  predict_future()       → Generate N-day forecast       │   │
│  │  get_forecast_for_dashboard()→ Dashboard-ready JSON    │   │
│  │  list_trained_models()  → Inventory saved models        │   │
│  │  load_model()           → Load pickled model            │   │
│  │                                                         │   │
│  │  Models stored in models/ directory (*.pkl)             │   │
│  │  One model per city, trained on-demand or pre-trained   │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Notable Design Decisions

- **Per-city models, not global** — Delhi's dynamics differ fundamentally from Bengaluru's. A single model across all cities would underfit.
- **NaN strategy is per-city** — features with >80% missing in Mumbai (like NH3, NOx) are auto-dropped for that city but retained for Delhi where they're well-measured.
- **Time-based split, not random** — avoids data leakage from future into past.
- **naive baselines included** — Moving Average and Seasonal Naive let us quantify the ML value-add (ML is 10–20× better).
- **Models serialized to disk** — no GPU, no cloud dependency. Single-machine inference.

---

## 5. Presentation Layer

```
┌──────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│                                                                │
│  ┌─────────────────────────┐    ┌──────────────────────────┐   │
│  │   Streamlit Dashboard   │    │    FastAPI REST API      │   │
│  │   (scripts/dashboard.py)│    │    (scripts/api.py)      │   │
│  ├─────────────────────────┤    ├──────────────────────────┤   │
│  │  Page 1: Executive      │    │  GET /health            │   │
│  │         Summary         │    │  GET /cities            │   │
│  │  Page 2: Historical     │    │  GET /forecast/{city}   │   │
│  │         Trends          │    │  GET /validate/{city}   │   │
│  │  Page 3: Pollutant      │    │  GET /data/freshness    │   │
│  │         Drill-Down      │    └──────────────────────────┘   │
│  │  Page 4: City Deep-Dive │                                   │
│  │  Page 5: Data Quality   │    Port 8000                     │
│  │  Page 6: Forecasting    │                                   │
│  ├─────────────────────────┤                                   │
│  │  Port 8501              │                                   │
│  └─────────────────────────┘                                   │
│                                                                │
│  All charts via matplotlib (lib/charts.py)                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure

```
┌──────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS                        │
│                                                                │
│  Docker (recommended)              Local (dev)                 │
│  ┌──────────────────────┐        ┌────────────────────────┐   │
│  │  docker-compose.yml  │        │  python scripts/*.py   │   │
│  │                      │        │                        │   │
│  │  postgres:16         │        │  Local PostgreSQL       │   │
│  │  seed (one-shot)     │        │  + venv + pip install  │   │
│  │  dashboard (:8501)   │        │                        │   │
│  │  api (:8000, profile)│        │  streamlit run         │   │
│  └──────────────────────┘        └────────────────────────┘   │
│                                                                │
│  CI (GitHub Actions)                                            │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  On push to main:                                     │   │
│  │  ├── pip install -r requirements.txt                 │   │
│  │  ├── ruff check lib/ tests/ scripts/                 │   │
│  │  └── pytest tests/ -v --cov=lib/ --cov-report=term   │   │
│  └────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Provenance Detail

```
Each row in city_measurements:

┌────────────┬──────────────────────────────────────┐
│ Column     │ Description                          │
├────────────┼──────────────────────────────────────┤
│ date       │ Measurement date                     │
│ city       │ City name (26 unique)                │
│ aqi        │ AQI value (0–500+, CPCB computed)    │
│ pm2_5      │ PM2.5 concentration (µg/m³)          │
│ pm10       │ PM10 concentration (µg/m³)           │
│ no2, co... │ Other pollutants                     │
│ aqi_bucket │ Category label (Good → Severe)       │
│ is_synthetic│ TRUE if machine-generated           │
│ data_source│ "CPCB", "OpenAQ", or "synthetic"     │
│ ingested_at│ Insert timestamp                     │
└────────────┴──────────────────────────────────────┘

Queries default to: WHERE is_synthetic = FALSE
```
