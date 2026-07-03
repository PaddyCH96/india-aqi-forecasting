# 2-Minute Demo Walkthrough

A scripted flow to show the system to recruiters, interviewers, or stakeholders.

---

## Setup

```bash
# Already running? Skip this.
git clone https://github.com/PaddyCH96/india-aqi-forecasting.git
cd india-air-quality
docker compose up --build
# Wait 30s for seed + dashboard startup
open http://localhost:8501
```

---

## Flow (:00–:10) — Start → Dashboard Loads

> "This dashboard analyzes air quality across 26 Indian cities using 5.5 years of CPCB data. I built the full pipeline — from raw CSV files to ML forecasting — as an end-to-end portfolio project."

**Action:** The dashboard loads showing Page 1 (Executive Summary). Six cities are pre-selected in the sidebar. KPI cards show city count, date range, total rows, and highest/lowest averages.

---

## Flow (:15–:45) — City Comparison Insight

> "The first thing you notice: **Delhi is in a different league**. Its average AQI is 259 — that's 2.7× higher than the next worst city. Most Indian cities average in the 'Moderate' range, but Delhi sits firmly in 'Poor' territory year-round. This isn't incremental — it's a fundamentally different pollution regime."

**Action:** Scroll down to the city ranking bar chart. Point to Delhi's bar vs Kolkata/Bengaluru. Then click Page 2 (Historical Trends).

---

## Flow (:50–1:10) — Seasonal Trend

> "The seasonal pattern is striking. Northern cities like Delhi see winter AQI spike to 2.5× summer levels — temperature inversions trap pollutants. Southern cities like Bengaluru have much milder variation — about 1.3×. **Pollution in India is geographically determined, and national policies miss this.**"

**Action:** On Page 2, show the seasonal box plots for Delhi vs Bengaluru. Point to the winter/summer ratio chart.

---

## Flow (1:15–1:40) — ML Forecast

> "This is the forecasting page. I trained XGBoost models per city with 66 engineered features — lags, rolling windows, seasonal cycles, pollutant interactions. For Bengaluru, the model achieves **0.8% MAPE** — meaning it predicts AQI within 1% of actual values. Even Mumbai, which has 61% missing data, gets to 2.9% MAPE. **Data quality is the constraint, not model choice.**"

**Action:** Click Page 6 (Forecasting). Select Bengaluru. Set horizon to 72h. Click "Train Model" (or show pre-trained). Point out the confidence bands and the comparison against historical data.

---

## Flow (1:45–2:00) — Key Insight + Close

> "The most actionable insight from this project: **PM2.5 alone predicts AQI with r=0.97**. You don't need 12 pollutants — you need one good PM2.5 sensor and calendar features. And Mumbai's 61% data gap shows that India's financial capital is making policy decisions with less data than any other major city. **Better monitoring would improve forecasting more than a better algorithm would.**"

**Action:** Click Page 3 (Pollutant Drill-Down). Show the correlation heatmap — point to PM2.5 ↔ AQI at 0.97. Then stop.

---

## Summary for Interviewer

> "I built this as a complete data project: ingestion from government CSVs and APIs, data quality auditing with provenance tracking, 15-function EDA library, 5-model ML pipeline with feature engineering, an interactive dashboard, and a REST API — all Dockerized and tested (144 tests, 95% coverage). The code, case study, and insights are all on GitHub."

---

## Quick Reference — Where to Find Things

| Need This | Go Here |
|-----------|---------|
| Narrative overview | `CASE_STUDY.md` |
| Top 5 insights | `INSIGHTS.md` |
| ML performance | `docs/ml_evaluation_report.md` |
| Architecture | `docs/architecture.md` |
| Data quality audit | `docs/data_quality_report.md` |
| EDA findings | `docs/eda_report.md` |
| Run live demo | `docker compose up --build` |
