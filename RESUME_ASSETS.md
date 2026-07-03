# Resume & LinkedIn Assets

---

## Resume Bullet Points (Impact-Focused)

**Bullet 1 — System Scale**
> Built an end-to-end air quality forecasting system for 26 Indian cities processing 700K+ hourly records and 12 pollutants. Designed a PostgreSQL-backed data pipeline with provenance tracking (real vs. synthetic separation) that eliminated data quality ambiguity.

**Bullet 2 — ML Performance**
> Trained per-city XGBoost regression models with 66 engineered features (lags, rolling windows, seasonal cycles, pollutant interactions) achieving 0.8–3.2% MAPE — outperforming naive baselines by 10–20×. Identified that data coverage, not model complexity, is the primary accuracy constraint.

**Bullet 3 — Full-Stack Delivery**
> Delivered a production-ready system including an interactive 6-page Streamlit dashboard, FastAPI REST API, Docker deployment (4 services), CI/CD pipeline (144 tests, 95% coverage, Ruff-clean), and comprehensive portfolio documentation.

---

## LinkedIn Post

**Headline (hook):** I built an air quality forecasting system for 26 Indian cities — and discovered Mumbai has a monitoring crisis.

**Body:**

Most people think you need complex models to forecast air quality. Turns out, a simple XGBoost with smart features hits 0.8% MAPE.

What actually matters? Data quality.

For this project, I built an end-to-end data system:

→ Ingested 5.5 years of CPCB data (700K+ hourly records, 12 pollutants)
→ Built a feature engineering pipeline generating 66 features per city
→ Trained XGBoost models achieving 0.8–3.2% MAPE across 6 major cities
→ Created a 6-page analytics + forecasting dashboard
→ Packaged everything in Docker with 144 passing tests

The most surprising finding?

**Mumbai has 61% of its AQI data missing** — the worst monitoring of any major Indian city. India's financial capital is making air quality decisions with less data than any comparable city.

And Delhi's mean AQI of 259 is 2.7× higher than the next worst city. It's not in the same category — it's in a different regime entirely.

Full case study, code, and insights: https://github.com/PaddyCH96/india-aqi-forecasting

#DataEngineering #MachineLearning #AirQuality #India #PortfolioProject

---

## Short Project Summary (80 words)

Built an end-to-end air quality forecasting system for 26 Indian cities. Ingested 700K+ hourly records from CPCB, built a 66-feature engineering pipeline, and trained per-city XGBoost models achieving 0.8–3.2% MAPE. Discovered that data quality (not model choice) is the accuracy constraint — Mumbai has 61% missing AQI data. Delivered via 6-page Streamlit dashboard, FastAPI, Docker, and CI/CD with 144 tests.

---

## GitHub Description (Short)

> **India Air Quality Forecasting** — End-to-end data + ML system for 26 cities. XGBoost achieves 0.8–3.2% MAPE. Streamlit dashboard, FastAPI, Docker, CI/CD. 144 tests, 95% coverage.

## GitHub About Section

> End-to-end data engineering and ML portfolio project. Forecasts AQI across 26 Indian cities using XGBoost with 66 engineered features. Features include interactive dashboards, REST API, Docker deployment, and comprehensive portfolio documentation. Achieves 0.8-3.2% MAPE across 6 major cities.
