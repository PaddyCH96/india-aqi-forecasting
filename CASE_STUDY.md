# India Air Quality: From Sparse Data to City-Level Forecasting

*An end-to-end data engineering and ML case study across 26 Indian cities, 12 pollutants, and 5.5 years of measurements.*

---

## Problem

India is home to 22 of the world's 30 most polluted cities. Yet public air quality data is fragmented across agencies, inconsistent in quality, and almost never used for forward-looking analysis.

Urban planners, real estate developers, and policymakers need to answer:
- Which cities are getting worse — and how fast?
- When will pollution spike next season?
- Can we predict AQI accurately enough to act on it?

The data exists. But it's messy, incomplete, and locked in government CSV files.

---

## Why It Matters

Air pollution in India is not just a health crisis — it's an economic one:
- **Real estate:** Premium housing in polluted cities faces depreciation pressure
- **Workforce:** Poor AQI reduces productivity and drives talent to cleaner cities
- **Policy:** Cities spend billions on mitigation but lack tools to measure effectiveness
- **Planning:** Urban expansion decisions rarely consider air quality trajectories

A system that can ingest, clean, analyze, and forecast air quality data turns a public dataset into a decision-support tool.

---

## The Data System

**Source:** CPCB (Central Pollution Control Board) — 5 files, 250 MB compressed, publicly available

| Dataset | Rows | Coverage | Quality |
|---------|------|----------|---------|
| `city_day.csv` | 29,531 | 26 cities, 2015–2020 | Clean (0 duplicates, 0 timestamp errors) |
| `city_hour.csv` | 707,875 | 26 cities, hourly | Medium coverage, 992 gaps >2h detected |
| `station_day.csv` | 108,035 | 110 stations | High quality |
| `station_hour.csv` | ~4M | 110 stations, hourly | Largest, but 209 MB |
| `stations.csv` | 230 | Station metadata | Static reference |

**Data quality realities:**

- **Mumbai** has only 38.6% AQI coverage — worst of any city. 61% of days have no reading.
- **Xylene** is 61% missing across all cities — dropped from modeling.
- **AQI bucket labels** perfectly match computed values — zero errors.
- **PM2.5** explains 97% of AQI variance — a single sensor covers most predictive power.
- After feature engineering (lags, rolling windows), Mumbai goes from 2,009 daily rows to only 227 usable training samples.

**Data provenance:** Every row is flagged `is_synthetic` (True/False) with a `data_source` column. All queries default to real data only — synthetic must be explicitly opted into.

**Quote from the data quality audit:**
> "Mumbai has critically low coverage. Only CO is well-measured. All other pollutants have ~40% or less coverage. This is the single largest data quality issue in the system."

---

## The Analytics System

Built a 6-page interactive dashboard covering:

1. **Executive Summary** — National snapshot, KPI cards, city ranking
2. **Historical Trends** — Multi-city trends over time, seasonal decomposition
3. **Pollutant Drill-Down** — Per-pollutant distributions, correlation matrices
4. **City Deep-Dive** — Single-city analysis with year-over-year comparisons
5. **Data Quality** — Missing data heatmaps, completeness warnings
6. **Forecasting** — ML-based predictions with confidence intervals

**Key analytical findings from EDA:**

- **Delhi is an outlier** — mean AQI 259.5, which is 2.7× higher than the next city (Kolkata, 140.6). Its winter AQI averages ~350.
- **70% of days** across all cities fall in "Moderate" or worse category. Only 16.4% of days are "Satisfactory" or better.
- **Winter is 1.5–2.5× worse** than summer for northern cities, but only ~1.3× for southern cities — a geographically determined pattern.
- **Weekend AQI is 5–10% lower** across most cities, strongest in traffic-dominated Delhi and Mumbai.
- **PM2.5 ↔ AQI: r = 0.97.** PM2.5 ↔ O3: r = 0.08. These pollutants operate independently.

---

## The ML System

### Architecture

```
engineered features → time-based split → NaN handling → model training → evaluation → forecast
```

- **5 model types:** Moving Average, Seasonal Naive, XGBoost, Random Forest, Prophet
- **66 features per city:** lags (1/2/3/7d), rolling stats (3/7/30d), cyclical seasonals, pollutant interactions, city normalization
- **NaN strategy:** auto-drop features >80% missing per-city, median imputation for the rest
- **All models per city** — no cross-city transfer (Delhi ≠ Bengaluru)
- **6 models on disk**, ready for dashboard inference

### Results

| City | Training Days | XGBoost RMSE | MAPE | Best Model |
|------|:------------:|:-----------:|:----:|:----------:|
| Bengaluru | 1,362 | 0.9 | **0.8%** | Random Forest |
| Hyderabad | 1,332 | 0.9 | **0.9%** | Random Forest |
| Chennai | 1,336 | 1.6 | **0.9%** | Random Forest |
| Delhi | 1,451 | 2.7 | **1.0%** | Random Forest |
| Mumbai | 227 | 6.7 | **2.9%** | XGBoost |
| Kolkata | 206 | 6.9 | **3.2%** | XGBoost |

### What the numbers mean

**Data quality determines accuracy, not model choice.** Cities with >1,300 training samples consistently achieve sub-1% MAPE regardless of whether we use XGBoost or Random Forest. Mumbai and Kolkata — the cities with the worst monitoring — have 3–4× higher error even with the same algorithm. The single highest-ROI improvement for this system is better data collection, not a better model.

Baselines confirm the ML adds value: Moving Average achieves 12–25% MAPE, Seasonal Naive 31–64%. The feature engineering pipeline (66 features) is what separates XGBoost from naive approaches.

---

## Key Results

1. **XGBoost achieves 0.8–3.2% MAPE** across 6 cities — 10–20× better than naive baselines
2. **Delhi's mean AQI (259.5) is 2.7× higher** than the next worst major city — it's in a different pollution regime entirely
3. **Mumbai has a monitoring crisis** — 61.4% of daily AQI records are missing, making it the hardest city to model
4. **PM2.5 alone predicts AQI with r=0.97** — the other 11 pollutants add marginal value for forecasting
5. **Winter pollution penalty is geographically determined** — 2.5× in the north, 1.3× in the south — national policies miss this

---

## System Architecture

```
OpenAQ API ──┐
Open-Meteo   ─┼──► seed_data.py ──► PostgreSQL ──┬──► Dashboards (Streamlit :8501)
CPCB CSVs   ─┘                                    ├──► REST API (FastAPI :8000)
                                                   └──► ML Pipeline ──► Forecast
                                                        │
                                                    lib/feature_engineering.py
                                                    lib/ml_pipeline.py
                                                    lib/model_training.py
                                                    lib/model_evaluation.py
                                                    lib/forecasting_service.py
```

**Tech stack:** Python (+ Prophet, XGBoost, scikit-learn) · PostgreSQL · Streamlit · FastAPI · Docker · GitHub Actions · pytest

**144 tests** across 9 files, all passing. Ruff-clean. 95% code coverage.

---

## Limitations

- **No real data beyond 2020** — CPCB dataset ends July 2020. Synthetic data extends to 2024 for demo purposes but is AQI-only (no multi-pollutant).
- **Mumbai and Kolkata** have <250 usable training samples — models for these cities have wider confidence intervals.
- **Future prediction** uses last-known features rather than true multi-step forecasting — adequate for the dashboard but not production-grade.
- **No weather regressors** — temperature, wind, and humidity would improve accuracy but aren't in the current data.
- **No hyperparameter tuning** — default XGBoost parameters were used for consistency.

---

## Future Improvements

1. **OpenAQ extension** — fetch 2020–2024 real data to replace synthetic and extend test sets
2. **Hyperparameter tuning** per city (Optuna/Ray Tune)
3. **Multi-step forecasting** — replace naive `predict_future()` with proper iterated forecasting
4. **SHAP explanations** — explain individual high-AQI predictions
5. **Weather integration** — add temperature/humidity/wind from Open-Meteo
6. **Station-level models** — leverage 108K station-day records for finer-grained predictions
7. **Anomaly detection** — flag Diwali, stubble burning, and other extreme events
8. **CI/CD deployment** — auto-retrain on new data

---

*Built with CPCB open data, Prophet, XGBoost, Streamlit, and PostgreSQL.*
*All code, analysis, and documentation at [github.com/PaddyCH96/india-aqi-forecasting](https://github.com/PaddyCH96/india-aqi-forecasting)*
