# Data Audit: india-air-quality

## 1. Data Sources

### 1.1 CPCB Air Quality Dataset (Real)

| File | Size | Rows | Level | Frequency |
|------|------|------|-------|-----------|
| `data/raw/city_day.csv` | 2.5 MB | 29,531 | City | Daily |
| `data/raw/city_hour.csv` | 63 MB | 707,875 | City | Hourly |
| `data/raw/station_day.csv` | 8.2 MB | 108,035 | Station | Daily |
| `data/raw/station_hour.csv` | 209 MB | ~4M | Station | Hourly |
| `data/raw/stations.csv` | 14 KB | 230 | Station metadata | Static |
| `data/raw/aqi_2015_2024.csv` | 14 B | 0 | — | Empty (404) |

Source: Central Pollution Control Board (CPCB), Government of India, published 2020.

### 1.2 Synthetic Data (Fallback)

| Generator | Cities | Range | Variables |
|-----------|--------|-------|-----------|
| `lib/aqi.py:generate_synthetic_aqi()` | 6 (Delhi, Mumbai, Bengaluru, Chennai, Hyderabad, Kolkata) | 2020-07-01 to 2024-12-31 | AQI, pm2_5 (derived) |

Synthetic generation uses sinusoidal seasonality + linear trend + weekday effect + Gaussian noise, clamped to [30, 400].

## 2. Available Variables

### 2.1 All Files (except stations.csv)

| Column | Type | Description | city_day Null% |
|--------|------|-------------|----------------|
| City/StationId | str | Geographic entity name | 0% |
| Date/Datetime | str/date | Observation timestamp | 0% |
| PM2.5 | float | Particulate matter <2.5µm (µg/m³) | 15.6% |
| PM10 | float | Particulate matter <10µm (µg/m³) | 37.7% |
| NO | float | Nitric oxide (µg/m³) | 12.1% |
| NO2 | float | Nitrogen dioxide (µg/m³) | 12.1% |
| NOx | float | Nitrogen oxides (µg/m³) | 14.2% |
| NH3 | float | Ammonia (µg/m³) | 35.0% |
| CO | float | Carbon monoxide (mg/m³) | 7.0% |
| SO2 | float | Sulfur dioxide (µg/m³) | 13.1% |
| O3 | float | Ozone (µg/m³) | 13.6% |
| Benzene | float | Benzene (µg/m³) | 19.0% |
| Toluene | float | Toluene (µg/m³) | 27.2% |
| Xylene | float | Xylene (µg/m³) | 61.3% |
| AQI | float | Computed Air Quality Index | 15.9% |
| AQI_Bucket | str | Categorical: Good/Satisfactory/Moderate/Poor/Very Poor/Severe | 15.9% |

### 2.2 Stations Metadata

| Column | Description |
|--------|-------------|
| StationId | Unique station code (e.g., AP001) |
| StationName | Full station name with agency |
| City | City name |
| State | State/UT name |
| Status | Active/Inactive |

## 3. City Coverage (city_day.csv)

### 3.1 All 26 Cities

| City | Rows | Date Range | AQI Avail% | PM2.5 Avail% |
|------|------|------------|------------|---------------|
| Ahmedabad | 2,009 | 2015-01-01 — 2020-07-01 | 66.4% | 68.7% |
| Delhi | 2,009 | 2015-01-01 — 2020-07-01 | 99.5% | 99.9% |
| Mumbai | 2,009 | 2015-01-01 — 2020-07-01 | 38.6% | 39.0% |
| Bengaluru | 2,009 | 2015-01-01 — 2020-07-01 | 95.1% | 92.7% |
| Lucknow | 2,009 | 2015-01-01 — 2020-07-01 | 94.2% | 94.9% |
| Chennai | 2,009 | 2015-01-01 — 2020-07-01 | 93.8% | 94.2% |
| Hyderabad | 2,006 | 2015-01-04 — 2020-07-01 | 93.7% | 94.3% |
| Patna | 1,858 | 2015-06-01 — 2020-07-01 | 78.5% | 82.7% |
| Gurugram | 1,679 | 2015-11-27 — 2020-07-01 | 86.5% | 90.8% |
| Visakhapatnam | 1,462 | 2016-07-01 — 2020-07-01 | 80.1% | 84.3% |
| Amritsar | 1,221 | 2017-02-27 — 2020-07-01 | 92.2% | 89.5% |
| Jorapokhar | 1,169 | 2017-04-20 — 2020-07-01 | 65.9% | 32.4% |
| Jaipur | 1,114 | 2017-06-14 — 2020-07-01 | 98.2% | 99.0% |
| Thiruvananthapuram | 1,112 | 2017-06-16 — 2020-07-01 | 94.6% | 96.2% |
| Amaravati | 951 | 2017-11-24 — 2020-07-01 | 88.4% | 93.8% |
| Brajrajnagar | 938 | 2017-12-07 — 2020-07-01 | 76.0% | 80.3% |
| Talcher | 925 | 2017-12-20 — 2020-07-01 | 75.5% | 80.1% |
| Kolkata | 814 | 2018-04-10 — 2020-07-01 | 92.6% | 93.2% |
| Guwahati | 502 | 2019-02-16 — 2020-07-01 | 98.6% | 99.8% |
| Coimbatore | 386 | 2019-06-12 — 2020-07-01 | 89.1% | 97.9% |
| Aizawl | 386 | 2019-06-01 — 2020-07-01 | 98.4% | 98.2% |
| Bhopal | 386 | 2019-06-17 — 2020-07-01 | 96.4% | 100.0% |
| Shillong | 383 | 2019-06-17 — 2020-07-01 | 97.4% | 98.7% |
| Chandigarh | 382 | 2019-06-17 — 2020-07-01 | 94.2% | 97.6% |
| Ernakulam | 377 | 2019-06-18 — 2020-07-01 | 97.9% | 100.0% |
| Kochi | 159 | 2020-01-22 — 2020-07-01 | 97.5% | 100.0% |

### 3.2 Station-Level Coverage

- **110 unique stations** across 127 cities in 21 states
- 131 active stations, 2 inactive
- Station data enables within-city geographic analysis

## 4. Data Quality Assessment

### 4.1 Completeness

| Category | Score | Assessment |
|----------|-------|-----------|
| Spatial (city) | High | 26 cities across major regions |
| Temporal (range) | Low | Real data ends July 2020 (5.5 years) |
| Temporal (frequency) | High | Both daily and hourly available |
| Variables (core) | High | 12 pollutant variables + computed AQI |
| Variables (VOCs) | Moderate | Benzene 81%, Toluene 73%, Xylene 39% |

### 4.2 Quality Issues

1. **Data gap after July 2020** — All real data stops mid-2020. Current project uses synthetic data (2020-2024) without clear labeling.
2. **Variable coverage varies by city** — Mumbai has only 38.6% AQI coverage vs Delhi at 99.5%.
3. **NH3 and Xylene have high missing rates** — NH3 35%, Xylene 61% missing.
4. **AQI is computed, not measured** — derived from pollutant concentrations. The computation methodology is opaque (CPCB-provided).
5. **Synthetic data is AQI-only** — no NO2, CO, O3, or other pollutants in the 2020-2024 synthetic range.
6. **Station metadata has gaps** — some stations lack Status values.
7. **`aqi_2015_2024.csv` is an empty file** (contains "404: Not Found") — likely a failed API download.

### 4.3 Missing Data Patterns

- Missing PM2.5 and missing AQI are highly correlated (r ≈ 0.96)
- Missing rates increase for less-commonly measured pollutants (NH3, Xylene, Benzene, Toluene)
- Missing data is mostly systematic (entire days missing) rather than random
- Later years (2019-2020) have better coverage due to CPCB network expansion

## 5. Suitability Assessment

### 5.1 Forecasting (Current Prophet Model)

| Criterion | Assessment |
|-----------|-----------|
| Sufficient history | Marginal — 5.5 years is borderline for annual seasonality |
| No missing gaps | No — significant gaps in Mumbai, moderate in others |
| Multiple frequencies | Yes — daily and hourly available |
| External regressors | Available — could add temperature proxy (seasonal), CO/NO2 as predictors |
| Data after 2020 | Synthetic only — no real data for 2020-2024 |

### 5.2 Visualization

| Criterion | Assessment |
|-----------|-----------|
| Multi-variable | Strong — 12+ pollutants to visualize |
| Multi-city | Strong — 26 cities for comparison |
| Temporal range | Moderate — clear seasonal patterns visible |
| Geographic | Strong — 110 stations, 127 cities, 21 states |
| Categorical labels | Yes — AQI_Bucket enables risk-based coloring |

### 5.3 Machine Learning

| Criterion | Assessment |
|-----------|-----------|
| Feature count | High — 12 pollutant features + temporal features |
| Label availability | Yes — AQI (regression), AQI_Bucket (classification) |
| Sample size (daily) | 29,531 rows — sufficient for most models |
| Sample size (hourly) | 707,875 rows — ample for deep learning |
| Train/test split | Natural split at 2020-01-01 (pre-pandemic vs pandemic) |

## 6. Critical Finding

**The current project leaks synthetic data into what appears to be a real forecast.** The `seed_data.py` script appends synthetic 2020-2024 data to the same `city_day` table alongside real 2015-2020 CPCB data. Any forecast beyond mid-2020 is partially trained on synthetic data. This must be clearly documented or remediated for portfolio use.

Options:
1. **Label explicitly**: Keep synthetic data but mark it clearly in docs, the dashboard, and API responses
2. **Censor at 2020**: Restrict all analysis to 2015-2020 real data only
3. **Acquire fresh data**: Use OpenAQ API or CPCB live feed to get 2020-2024 real data
4. **Two-tier**: Real data for modeling, synthetic only for demo/deployment scenarios
