# Dashboard Plan: india-air-quality

## Current State

Two Streamlit dashboards exist:
- `dashboard_final.py` — 3 tabs (History, Forecast, Validation)
- `dashboard_complete.py` — 4 tabs (History, Forecast, Validation, Multi-City Comparison)

Both use only AQI from 6 cities. Both depend on synthetic 2020-2024 data for forecasts.

## Proposed Dashboard Architecture

### Single, Unified Dashboard with 6 Pages

```
Page 1: Executive Summary       ← New
Page 2: Historical Trends       ← Enhanced from History
Page 3: Pollutant Drill-Down    ← New
Page 4: Forecast Hub            ← Enhanced from Forecast
Page 5: City Comparison         ← Enhanced from Multi-City Comparison
Page 6: Health Risk Analysis    ← New
```

## Page Specifications

### Page 1: Executive Summary

**Purpose**: One-glance national air quality status.

| Component | Type | Data Source |
|-----------|------|-------------|
| National average AQI (today) | KPI card | Latest city_day data |
| Cities above moderate (AQI > 100) | KPI card | Current AQI bucket |
| Worst city today | KPI card | Max AQI city |
| National AQI trend (5-year) | Sparkline | Monthly averages |
| City ranking bar chart | Horizontal bar | Mean AQI per city |
| India map with city bubbles | Geo scatter | City lat/lng + AQI |
| AQI bucket distribution | Donut chart | Count per bucket |

**Interactions**: City selector → filters all widgets, Date range slider.

### Page 2: Historical Trends

**Purpose**: Explore how pollution has changed over time across pollutants.

| Component | Type | Data Source |
|-----------|------|-------------|
| AQI over time | Interactive line chart (Plotly) | city_day with date range selector |
| Pollutant selector | Dropdown | PM2.5, PM10, NO2, CO, O3, SO2 |
| Moving average overlay | Toggle | 7/30/90/365 day rolling mean |
| Year-over-year comparison | Overlay chart | Same month, multiple years |
| Seasonal decomposition | STL components | Trend, seasonal, residual |
| Month-by-year heatmap | Calendar heatmap | Mean AQI per month |

**Interactions**: City selector, date range slider, pollutant dropdown, moving average toggle.

### Page 3: Pollutant Drill-Down

**Purpose**: Deep analysis of individual pollutants and their relationships.

| Component | Type | Data Source |
|-----------|------|-------------|
| Pollutant distribution | Histogram + KDE | Selected pollutant per city |
| Correlation heatmap | Colored matrix | All 12 pollutants (selectable subset) |
| Scatter plot (2 pollutants) | Interactive scatter | Any 2 pollutant axes |
| Pollutant vs AQI scatter | Scatter with trend line | Individual pollutant vs computed AQI |
| Time-series (single pollutant) | Line chart | Selected pollutant over time |
| Missing data pattern | Heatmap | Null values by month and pollutant |

**Interactions**: City selector, pollutant X/Y selectors, date range.

### Page 4: Forecast Hub

**Purpose**: View and compare Prophet forecasts.

| Component | Type | Data Source |
|-----------|------|-------------|
| Forecast chart (current) | Line with uncertainty | Prophet output |
| Model quality badge | Colored indicator | MAPE-based (Excellent/Good/Moderate/Poor) |
| Forecast table | Data table | 30/90/365 day ahead values |
| Training data overlay | Toggle | Show actuals behind forecast |
| 2030 projection | Single KPI | Forecasted AQI for Jan 2030 |
| Confidence interval | Range display | yhat_lower → yhat_upper |
| What-if scenario | Toggle | "If trend continues" vs "if improvement" |

**Interactions**: City selector, forecast horizon slider, model selector (if multiple).

### Page 5: City Comparison

**Purpose**: Compare multiple cities side-by-side.

| Component | Type | Data Source |
|-----------|------|-------------|
| Multi-city AQI trends | Overlay line chart | All selected cities |
| Comparison bar chart | Grouped bar | Metric per city |
| Radar chart | Multi-axis | Pollution profile per city |
| City ranking table | Sortable table | All metrics |
| Cluster scatter (PCA) | 2D scatter | City positions in PCA space |
| Correlation across cities | Heatmap | AQI correlation between city pairs |

**Interactions**: Multi-city selector (checkboxes), metric selector.

### Page 6: Health Risk Analysis

**Purpose**: Translate AQI into health risk guidance.

| Component | Type | Data Source |
|-----------|------|-------------|
| Current risk level | Gauge chart | Latest AQI bucket |
| Days at each risk level | Stacked bar | Annual distribution |
| Health advisory | Conditional text | Based on AQI_Bucket |
| High-risk event calendar | Calendar plot | Days with AQI > 200 |
| Year-over-year risk change | Comparison bars | Risk level counts by year |
| Population exposure estimate | Calculated metric | (If city population data available) |

**Interactions**: City selector, year selector.

## Technical Implementation

### Stack

| Component | Technology |
|-----------|-----------|
| Framework | Streamlit (retained for simplicity) |
| Charts | Plotly (interactive) + Matplotlib (static exports) |
| Layout | `st.tabs` + `st.sidebar` for navigation |
| State | `st.session_state` for cross-page filters |
| Data layer | Pandas + SQLAlchemy (existing pattern) |
| Caching | `@st.cache_data` for expensive queries |

### Navigation

```python
PAGES = {
    "Executive Summary": page_executive_summary,
    "Historical Trends": page_historical_trends,
    "Pollutant Drill-Down": page_pollutant_drilldown,
    "Forecast Hub": page_forecast_hub,
    "City Comparison": page_city_comparison,
    "Health Risk Analysis": page_health_risk,
}
```

### Global Filters (Sidebar)

- City: multi-select (default: all 26)
- Date range: slider (2015-01-01 to 2020-07-01 for real data)
- Pollutant: dropdown (default: AQI)
- Frequency: daily / weekly / monthly resampling

### Data Layer

```python
@st.cache_data(ttl=3600)
def load_city_data(cities, date_range, pollutant):
    # Parameterized SQL query
    # Returns DataFrame
```

## Implementation Priority

| Phase | Pages | Effort | Value |
|-------|-------|--------|-------|
| 1 | Executive Summary + Historical Trends | 3 days | High |
| 2 | Forecast Hub (refactor existing) | 2 days | High |
| 3 | Pollutant Drill-Down + Health Risk | 3 days | Medium |
| 4 | City Comparison (refactor existing) | 2 days | Medium |
| 5 | Geo map, exports, polish | 2 days | Low |

## Export Features (All Pages)

- Download chart as PNG/SVG
- Download data as CSV
- Shareable URL with filter state (via query params)

## Synthetic Data Labeling

Any chart/section using synthetic data (2020-2024) must display a clear badge:
```
⚠ Synthetic Data — Values beyond July 2020 are simulated
```
