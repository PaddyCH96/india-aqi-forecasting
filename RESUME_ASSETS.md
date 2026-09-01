# Resume & LinkedIn Assets

---

## Resume Bullet Points (Impact-Focused)

**Bullet 1 — System Scale**
> Built an end-to-end air quality forecasting system for 26 Indian cities processing 700K+ hourly records and 12 pollutants. Designed a PostgreSQL-backed data pipeline with provenance tracking (real vs. synthetic separation) that eliminated data quality ambiguity.

**Bullet 2 — ML Performance**
> Trained per-city XGBoost regression models with 66 engineered features (lags, rolling windows, seasonal cycles, pollutant interactions) and built a rolling-origin backtest scoring them against a persistence baseline across four horizons. Found persistence is a strong baseline for daily city AQI and the model beats it reliably only for Delhi at 1–3 days; traced an earlier sub-3% MAPE result to target leakage, fixed it, and published the negative result with a regression test guarding against recurrence.

**Bullet 3 — Full-Stack Delivery**
> Delivered a production-ready system including an interactive 6-page Streamlit dashboard, FastAPI REST API, Docker deployment (4 services), CI/CD pipeline (144 tests, 95% coverage, Ruff-clean), and comprehensive portfolio documentation.

---

## LinkedIn Post

**Headline (hook):** I built an air quality forecasting system for 26 Indian cities — and discovered Mumbai has a monitoring crisis.

**Body:**

Most people think a gradient-boosted model will beat a naive forecast for air quality. I backtested mine properly and it mostly doesn't — carrying today's AQI forward is a strong baseline, and my model beats it reliably only for Delhi at 1–3 days.

The part I'm actually proud of: my first version looked far better because two features leaked the target. I found it, fixed it, and published the worse number.

For this project, I built an end-to-end data system:

→ Ingested 5.5 years of CPCB data (700K+ hourly records, 12 pollutants)
→ Built a feature engineering pipeline generating 66 features per city
→ Backtested per-city XGBoost against a persistence baseline across 4 horizons — and reported where it loses
→ Created a 6-page analytics + forecasting dashboard
→ Packaged everything in Docker with 144 passing tests

The most surprising finding?

**Mumbai has 61% of its AQI data missing** — the worst monitoring of any major Indian city. India's financial capital is making air quality decisions with less data than any comparable city.

And Delhi's mean AQI of 259 is 2.7× higher than the next worst city. It's not in the same category — it's in a different regime entirely.

Full case study, code, and insights: https://github.com/PaddyCH96/india-aqi-forecasting

#DataEngineering #MachineLearning #AirQuality #India #PortfolioProject

---

## Short Project Summary (80 words)

Built an end-to-end air quality forecasting system for 26 Indian cities. Ingested 700K+ hourly records from CPCB, built a 66-feature engineering pipeline, and trained per-city XGBoost models scored by rolling-origin backtest against a persistence baseline. Persistence proved strong: the model beats it reliably only for Delhi at 1–3 days. An earlier sub-3% MAPE result came from target leakage — found, fixed, and now guarded by a regression test. Delivered via 6-page Streamlit dashboard, FastAPI, Docker, and CI/CD with 144 tests.

---

## GitHub Description (Short)

> **India Air Quality Forecasting** — End-to-end data + ML system for 26 cities. Per-city XGBoost backtested against a persistence baseline, with the negative result published. Streamlit dashboard, FastAPI, Docker, CI/CD. 144 tests, 95% coverage.

## GitHub About Section

> End-to-end data engineering and ML portfolio project. Forecasts AQI across 26 Indian cities using XGBoost with 66 engineered features. Features include interactive dashboards, REST API, Docker deployment, and comprehensive portfolio documentation. Forecasts are scored by rolling-origin backtest against a persistence baseline; the model beats persistence reliably only for Delhi at 1–3 days, and an earlier leaked-target result was found, fixed and test-guarded.
