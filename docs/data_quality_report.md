# Data Quality Report

## 1. Source Overview

| Dataset | Rows | Columns | Size | Format | Quality |
|---------|------|---------|------|--------|---------|
| `city_day.csv` | 29,531 | 16 | 2.5 MB | CSV, daily | High |
| `city_hour.csv` | 707,875 | 16 | 63 MB | CSV, hourly | Medium |
| `station_day.csv` | 108,035 | 16 | 8.2 MB | CSV, daily | High |
| `station_hour.csv` | ~4,000,000 | 16 | 209 MB | CSV, hourly | Medium |
| `stations.csv` | 230 | 5 | 14 KB | CSV, static | High |

## 2. Data Quality Dimensions

### 2.1 Completeness

| Pollutant | city_day | city_hour | station_day |
|-----------|----------|-----------|-------------|
| AQI | 84.1% | 60.7% | 80.6% |
| PM2.5 | 84.4% | 61.9% | 80.0% |
| PM10 | 62.3% | 0.0% (sampled) | 60.5% |
| CO | 93.0% | 65.9% | 88.0% |
| NO2 | 87.9% | 68.9% | 84.7% |
| O3 | 86.4% | 66.6% | 76.3% |
| SO2 | 86.9% | 64.6% | 76.7% |
| Benzene | 81.0% | 68.7% | 70.9% |
| Toluene | 72.8% | 68.7% | 64.2% |
| Xylene | 38.7% | 68.7% | 21.2% |
| NH3 | 65.0% | 0.0% (sampled) | 55.5% |

### 2.2 Completeness by City (city_day, AQI)

| City | AQI % | PM2.5 % | NO2 % | CO % | Grade |
|------|-------|---------|-------|------|-------|
| Delhi | 99.5% | 99.9% | 99.9% | 100.0% | A+ |
| Lucknow | 94.2% | 94.9% | 99.5% | 99.9% | A |
| Jaipur | 98.2% | 99.0% | 98.1% | 99.5% | A |
| Guwahati | 98.6% | 99.8% | 97.8% | 99.8% | A |
| Bengaluru | 95.1% | 92.7% | 99.7% | 99.5% | A |
| Chennai | 93.8% | 94.2% | 98.2% | 98.8% | A- |
| Hyderabad | 93.7% | 94.3% | 98.6% | 99.8% | A- |
| Kolkata | 92.6% | 93.2% | 96.9% | 100.0% | A- |
| Amritsar | 92.2% | 89.5% | 91.5% | 99.2% | B+ |
| Visakhapatnam | 80.1% | 84.3% | 60.6% | 79.5% | C |
| **Mumbai** | **38.6%** | **39.0%** | **37.6%** | **98.8%** | **F** |

**Mumbai has critically low coverage.** Only CO is well-measured. All other pollutants have ~40% or less coverage. This is the single largest data quality issue.

### 2.3 Duplicates

| File | Duplicate Rows | Duplicate Keys |
|------|---------------|---------------|
| city_day.csv | 0 | 0 (City+Date) |
| city_hour.csv | 0 | 0 (City+Datetime) |
| station_day.csv | 0 | 0 (StationId+Date) |
| stations.csv | 0 | 0 (StationId) |

**No duplicate issues found across any dataset.**

### 2.4 Timestamp Validity

| File | Invalid | Future Dates | Pre-2015 | Coverage |
|------|---------|-------------|----------|----------|
| city_day.csv | 0 | 0 | 0 | 2015-01-01 to 2020-07-01 |
| city_hour.csv | 0 | 0 | 0 | 2015-01-01 to 2020-07-01 |
| station_day.csv | 0 | 0 | 0 | 2015-01-01 to 2020-07-01 |

**No timestamp issues.** All dates are valid and within expected range.

### 2.5 Value Range Validation

| Variable | Valid Range | Issues Found |
|----------|-------------|-------------|
| PM2.5 | 0-500 µg/m³ | 0 negative values |
| PM10 | 0-1000 µg/m³ | 0 negative values |
| AQI | 0-500 (CPCB cap) | **543 values > 500 (max: 2049)** |
| CO | 0-10 mg/m³ | 0 negative values, some zeros suspicious |
| NO2 | 0-500 µg/m³ | 0 negative values |
| O3 | 0-500 µg/m³ | 0 negative values |

**AQI values exceed standard CPCB cap of 500.** This may be due to PM2.5 and PM10 readings where the sub-index calculation produces higher values. These are present in the source data and are not data entry errors.

### 2.6 AQI Bucket Consistency

| Source | Bucket-AQI Mismatches | Rate |
|--------|----------------------|------|
| city_day.csv | 0 / 24,850 | 0.00% |
| station_day.csv | 0 / 87,025 | 0.00% |

**Perfect alignment.** AQI_Bucket values exactly match AQI numerical thresholds. The CPCB computation is consistent.

### 2.7 Cross-Dataset Consistency

| Check | Result |
|-------|--------|
| City names match between city_day and stations.csv | ✅ All 26 city_day cities found in stations.csv |
| Station IDs in station_day match stations.csv | ✅ 110/110 matched |
| Date ranges consistent across datasets | ✅ All end at 2020-07-01 |

## 3. Synthetic Data Quality

| Dimension | Assessment |
|-----------|-----------|
| Temporal pattern | Sinusoidal yearly seasonality (reasonable) |
| Trend | Linear increase (captures broad urbanization) |
| Weekly pattern | Weekend dip (reasonable) |
| Noise | Gaussian (may not capture extreme events) |
| Multi-pollutant | **No — synthetic data is AQI-only** |
| Realism | Moderate — captures averages, misses extreme events |

## 4. Issues Log

| # | Severity | Issue | File | Status |
|---|----------|-------|------|--------|
| 1 | Critical | Synthetic data (2020-2024) mixed with real data in DB | city_day | Fixed |
| 2 | High | Mumbai 38.6% AQI coverage limits usability | city_day.csv | Noted |
| 3 | Medium | 543 AQI values exceed CPCB 500 cap | city_day.csv | Noted |
| 4 | Low | `aqi_2015_2024.csv` is empty (404) | aqi_2015_2024.csv | Noted |
| 5 | Low | 97/230 stations missing Status value | stations.csv | Noted |
| 6 | Low | Xylene 61.3% missing in city_day | city_day.csv | Noted |

## 5. Quality Grades by Dataset

| Dataset | Completeness | Accuracy | Consistency | Timeliness | Overall |
|---------|-------------|----------|-------------|------------|---------|
| city_day.csv | B | A | A | C (ends 2020) | B+ |
| city_hour.csv | C | A | A | C (ends 2020) | B- |
| station_day.csv | B | A | A | C (ends 2020) | B |
| station_hour.csv | C | A | A | C (ends 2020) | B- |
| stations.csv | B | A | A | A | A- |
| Synthetic data | D | C | A | A | C |

## 6. Recommendations

1. **Mumbai gap**: Use station-level data from stations.csv to fill Mumbai AQI gaps (multiple stations likely have partial coverage).
2. **AQI cap exceedance**: Document as CPCB methodology artifact, not an error.
3. **Xylene drop**: Drop Xylene from modeling due to 61% missing rate.
4. **Synthetic data**: Clearly separated with `is_synthetic=True` flag in the database.
5. **Expand with OpenAQ**: Fetch 2020-2024 data from OpenAQ to replace synthetic data.
