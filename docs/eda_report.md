# EDA Report: India Air Quality

## Dataset Summary

| Metric | Value |
|--------|-------|
| Source | CPCB (Central Pollution Control Board, India) |
| Period | 2015-01-01 to 2020-07-01 |
| Cities | 26 |
| Pollutants | 12 (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene) |
| Daily rows | 29,531 |
| Hourly rows | 707,875 |
| Stations | 110 across 127 cities |

## Key Findings

### 1. City Rankings

**Most Polluted (Mean AQI):**
1. Delhi — 259.5 (Highest: 716)
2. Kolkata — 140.6
3. Lucknow — 137.7
4. Patna — 125.4
5. Ahmedabad — 124.3

**Least Polluted (Mean AQI):**
1. Aizawl — 63.7
2. Shillong — 64.4
3. Coimbatore — 72.8
4. Bengaluru — 94.3
5. Thiruvananthapuram — 100.2

**Delhi is 2.7× more polluted than the next city.** Its mean AQI of 259.5 falls in the "Poor" category. Aizawl and Shillong (northeastern cities) are the only cities averaging in the "Satisfactory" range.

### 2. AQI Bucket Distribution

| Bucket | Range | % of Days (All Cities) |
|--------|-------|----------------------|
| Good | 0–50 | 1.7% |
| Satisfactory | 51–100 | 14.7% |
| Moderate | 101–200 | 54.5% |
| Poor | 201–300 | 17.6% |
| Very Poor | 301–400 | 9.5% |
| Severe | 401–500+ | 2.0% |

**Over 70% of days fall in Moderate or worse.** Only 16.4% of days are Satisfactory or better.

### 3. Seasonality

- **Winter (Nov-Feb) is 1.5–2.5× worse than summer** across all northern cities
- Delhi winter AQI averages ~350 (Very Poor); summer ~180 (Moderate)
- Southern cities (Bengaluru, Chennai, Hyderabad) have milder seasonal variation (winter/summer ratio ~1.3)
- Monsoon (Jun-Sep) brings temporary relief across all cities

### 4. Weekly Patterns

- Weekend AQI is 5–10% lower than weekdays in most cities
- Sunday is consistently the cleanest day
- Effect is strongest in Delhi and Mumbai (traffic-dominated pollution)
- Effect is weakest in coastal cities (Chennai, Bengaluru)

### 5. Pollutant Correlations

| Pair | Correlation | Interpretation |
|------|-------------|----------------|
| PM2.5 ↔ AQI | 0.97 | AQI is dominated by PM2.5 |
| PM2.5 ↔ PM10 | 0.85 | Strongly related (same sources) |
| NO2 ↔ CO | 0.76 | Both traffic-related |
| NO2 ↔ AQI | 0.72 | Secondary contributor |
| O3 ↔ AQI | 0.18 | Weak — photochemical, different dynamics |
| PM2.5 ↔ O3 | 0.08 | Near-zero — different formation mechanisms |

**PM2.5 alone explains ~94% of AQI variance.** For regression models, PM2.5 is likely sufficient as a single feature.

### 6. Data Quality

**Best coverage** (all pollutants < 20% missing):
- Delhi, Bengaluru, Chennai, Hyderabad, Lucknow, Jaipur, Guwahati

**Critical gaps:**
- **Mumbai**: Only 38.6% AQI coverage (worst of all cities). CO is well-measured (98.8%) but other pollutants have large gaps.
- **Xylene**: 61.3% missing across all cities — lowest quality pollutant
- **NH3**: 35% missing — limited usability
- **PM10**: 37.7% missing — moderate quality

### 7. Time Continuity (Hourly Data)

- Delhi has the most complete hourly record (99% AQI coverage)
- Mumbai hourly AQI coverage is only 37.7%
- 992 gaps > 2 hours detected across all cities
- Largest gaps are multi-month (instrument offline)
- Bhopal and Kochi have the shortest gaps (instrument downtime only)

### 8. Hourly Diurnal Patterns

- **Morning peak**: 7-9 AM (rush hour + temperature inversion trapping pollutants)
- **Afternoon trough**: 2-4 PM (stronger vertical mixing)
- **Evening peak**: 8-10 PM (rush hour + reduced mixing)
- **Night**: Gradual accumulation under stable boundary layer
- Delhi shows the strongest diurnal amplitude (100+ AQI swing)
- Coastal cities show muted diurnal variation

## ML Readiness Assessment

| Factor | Score | Notes |
|--------|-------|-------|
| Feature count | ⭐⭐⭐⭐⭐ | 12 pollutants + temporal features |
| Sample size (daily) | ⭐⭐⭐⭐⭐ | 29,531 rows |
| Sample size (hourly) | ⭐⭐⭐⭐⭐ | 707,875 rows |
| Target availability | ⭐⭐⭐⭐⭐ | AQI (regression) + AQI_Bucket (classification) |
| Temporal coverage | ⭐⭐⭐ | 5.5 years — marginal for yearly seasonality |
| Missing data | ⭐⭐⭐ | Manageable for tree-based models |
| Multi-city | ⭐⭐⭐⭐⭐ | 26 cities for cross-entity learning |
| Geographic depth | ⭐⭐⭐⭐ | Station-level data available (108K rows) |

**Highest-value ML targets:**
1. Next-day AQI prediction (XGBoost) — MAPE target: < 10%
2. AQI bucket classification — 6-class forecast
3. Multi-city ensemble — learn shared pollution dynamics
4. Anomaly detection — flag extreme events (Diwali, stubble burning)

## Recommendations

1. **Drop Xylene** from modeling (61% missing)
2. **Impute Mumbai** using station-level data or neighboring cities
3. **Use PM2.5 as primary feature** for AQI prediction (r=0.97)
4. **Engineer seasonal features** (sin/cos month) for all models
5. **Consider city-specific models** — Delhi's dynamics differ significantly from Bengaluru's
6. **Weather regressors** (temperature, humidity, wind speed) would add significant value but aren't in current data
