# Project Roadmap: india-air-quality

## Current Status vs Target

| Dimension | Current | Target |
|-----------|---------|--------|
| Cities | 6 (with synthetic data) | 26 (real data only, clearly labeled) |
| Pollutants | AQI only | 12+ pollutants |
| Data range | 2015-2024 (synthetic after 2020) | 2015-2020 (real, with plan to extend) |
| Modeling | Prophet only | Prophet + XGBoost + LSTM |
| Dashboards | 2 separate, limited | 1 unified, 6-page dashboard |
| Tests | 100 (good) | Maintain + add for new code |
| Documentation | Good | Improve with audit findings |

## Phases

### Phase 0: Foundation Audit (This Document)

- ✅ Data audit complete
- ✅ EDA plan defined
- ✅ Dashboard plan defined
- ✅ ML strategy defined
- ✅ Roadmap defined

### Phase 1: Data Pipeline Upgrade (1 week)

**Objective**: Clean up data handling, remove synthetic data leakage, establish real data baseline.

| Task | Details | Effort |
|------|---------|--------|
| Fork/copy real CPCB data into clean DB | Load all 26 cities, all pollutants, city_day + city_hour | 0.5 day |
| Mark synthetic data boundary | Add `is_synthetic` column to city_day | 0.5 day |
| Add OpenAQ ingestion for 2020-2024 real data | Extend `fetch_openaq.py` to cover all 26 cities | 1 day |
| Remove synthetic fallback from seed pipeline | seed_data.py should flag synthetic, not merge silently | 0.5 day |
| Add data freshness reporting | API endpoint `/data/freshness` + dashboard badge | 0.5 day |

**Deliverables:**
- Database with `is_synthetic` flag
- OpenAQ data for 2020-2024 (partial coverage expected)
- `data_freshness` module

### Phase 2: EDA Notebook Series (1 week)

**Objective**: Produce 8 Jupyter notebooks covering all EDA dimensions.

| Notebook | Contents | Depends On |
|----------|----------|------------|
| `01_distributions.ipynb` | Histograms, box plots, summary stats per city | Phase 1 |
| `02_temporal_trends.ipynb` | Trends, seasonality, YoY, day-of-week | Phase 1 |
| `03_correlations.ipynb` | Correlation matrices, scatter pairs | Phase 1 |
| `04_city_comparison.ipynb` | Rankings, clustering, comparative metrics | Phase 1 |
| `05_missing_data.ipynb` | Null patterns, gap analysis, imputation | Phase 1 |
| `06_anomaly_detection.ipynb` | Isolation Forest, event mapping | Phase 1 |
| `07_decomposition.ipynb` | STL decomposition, changepoints | Phase 1 |
| `08_station_analysis.ipynb` | Station-level spatial patterns | Phase 1 |

**Deliverables:**
- 8 notebooks with code + commentary + charts
- Key findings documented for dashboard integration

### Phase 3: Dashboard Rewrite (1.5 weeks)

**Objective**: Replace 2 existing dashboards with 1 unified 6-page dashboard.

| Task | Effort |
|------|--------|
| Page 1: Executive Summary (KPI cards, map, ranking) | 1 day |
| Page 2: Historical Trends (interactive line charts, heatmap) | 1 day |
| Page 3: Pollutant Drill-Down (distributions, correlations) | 1 day |
| Page 4: Forecast Hub (Prophet results with synthetic badge) | 1 day |
| Page 5: City Comparison (overlays, radar, PCA) | 1 day |
| Page 6: Health Risk Analysis (gauge, calendar, advisory) | 1 day |
| Sidebar global filters, state management, caching | 1 day |
| Synthetic data badge system, chart exports | 0.5 day |

**Deliverables:**
- Single `dashboard.py` with 6 pages
- All charts interactive (Plotly)
- Synthetic data clearly labeled throughout

### Phase 4: ML Pipeline (2 weeks)

**Objective**: Build production-grade ML pipeline surpassing Prophet baseline.

| Task | Effort | ML Phase |
|------|--------|----------|
| Feature engineering module (`lib/features.py`) | 1 day | 1 |
| XGBoost next-day AQI prediction | 2 days | 1 |
| AQI bucket classification (6-class) | 1 day | 1 |
| Model comparison dashboard tab | 1 day | 1 |
| LSTM hourly forecasting | 3 days | 2 |
| Multi-city ensemble model | 2 days | 2 |
| Anomaly detection pipeline | 2 days | 3 |
| Changepoint detection | 1 day | 3 |
| MLflow or ML tracking integration | 1 day | All |

**Deliverables:**
- `lib/features.py` (feature engineering)
- `lib/regressors.py` or individual model scripts
- ML comparison section in dashboard
- Anomaly detection results mapped to real events

### Phase 5: Portfolio Polish (1 week)

**Objective**: Make the repo presentation-ready.

| Task | Effort |
|------|--------|
| Project README rewrite (story-first narrative) | 1 day |
| Add screenshots/gifs of dashboard to README | 0.5 day |
| Write case study notebook (3-5 page narrative) | 1 day |
| Add slides directory (`presentation/`) | 0.5 day |
| Deploy demo (Render or Fly.io) | 1 day |
| Add CI for ML training (scheduled GitHub Action) | 0.5 day |
| Final walkthrough video script | 0.5 day |

**Deliverables:**
- Production README with screenshots
- Deployed demo URL
- Case study notebook
- Presentation materials

## Effort Summary

| Phase | Duration | Key Skills Demonstrated |
|-------|----------|------------------------|
| 0: Audit | (complete) | Data analysis, critical thinking |
| 1: Data Pipeline | 1 week | Data engineering, ETL, SQL |
| 2: EDA | 1 week | Statistics, visualization, storytelling |
| 3: Dashboard | 1.5 weeks | UI/UX, Streamlit, Plotly |
| 4: ML | 2 weeks | Feature engineering, XGBoost, LSTM, anomaly detection |
| 5: Polish | 1 week | Documentation, deployment, presentation |

**Total: ~6.5 weeks**

## Portfolio Recommendation (Strongest Version)

### Version A: "The Complete Package" (Recommended — 6.5 weeks)

A full-stack data science portfolio project demonstrating:

1. **Data Engineering** (Phase 1) — Multi-source ingestion, data quality handling, SQL
2. **Exploratory Analysis** (Phase 2) — Rigorous stats, 8 notebooks, clear findings
3. **Visualization** (Phase 3) — Professional 6-page interactive dashboard
4. **Machine Learning** (Phase 4) — Prophet → XGBoost → LSTM progression with comparisons
5. **Production Readiness** (Phase 5) — Tests, CI/CD, deployment, docs

**Portfolio Signal Strength:** Very High
**Target Roles:** Data Scientist, ML Engineer, Data Analyst

### Version B: "Analytics Focus" (3 weeks — Phases 1-3)

If ML depth isn't needed:

1. Data pipeline clean-up
2. EDA notebooks
3. Dashboard with all 12 pollutants, 26 cities, 6 pages

**Portfolio Signal Strength:** High
**Target Roles:** Data Analyst, BI Engineer, Analytics Engineer

### Version C: "ML Focus" (4 weeks — Phases 1, 2, 4)

If dashboard isn't needed:

1. Data pipeline + feature engineering
2. EDA (abbreviated)
3. Full ML pipeline: Prophet baseline → XGBoost → LSTM → Anomaly Detection

**Portfolio Signal Strength:** High
**Target Roles:** ML Engineer, Data Scientist

## Key Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| OpenAQ API has limited 2020-2024 historical coverage | Medium | Accept partial coverage; label clearly |
| Synthetic data contaminates model training | High | Add `is_synthetic` flag; train only on real data for portfolio |
| Mumbai only has 38% AQI coverage | Medium | Use station-level data to fill gaps; note limitation |
| 5.5 years insufficient for yearly seasonality | Medium | Note limitation; use weekly/daily patterns instead |
| Station-hour data is 209MB (slow to process) | Low | Use sampling or chunking for initial analysis |

## Immediate Next Step Recommendation

**Start with Phase 1, Task 1:** Load all 26 cities from `city_day.csv` into a clean database schema with all 12+ pollutant columns. This unblocks every subsequent phase and takes < 1 day.

Then proceed to **Phase 2, Notebook 1** (distributions) which will immediately surface insights worth visualizing.
