# EDA Plan: india-air-quality

## Overview

Systematic exploratory data analysis across 26 cities, 12+ pollutants, and 5.5 years of daily data (plus 4 years of hourly data). The goal is to surface patterns, anomalies, and relationships that inform forecasting, ML modeling, and dashboard design.

## 1. Univariate Analysis

### 1.1 Distribution Analysis (per city per pollutant)

| Analysis | Tool | Output |
|----------|------|--------|
| Histogram + KDE | `matplotlib` / `seaborn` | Distribution shape (normal, skewed, multimodal) |
| Box plots | `seaborn.boxplot` | IQR, outliers, whisker range |
| Summary stats | `pandas.describe()` | Mean, median, std, skew, kurtosis |
| AQI bucket proportions | `value_counts(normalize=True)` | % in Good/Satisfactory/Moderate/Poor/Very Poor/Severe |

**Key questions:**
- Which cities have the worst air quality (highest mean/median AQI)?
- Which pollutants show the widest variance?
- Are AQI distributions normal or skewed? (Hypothesis: right-skewed — days with extreme pollution)

### 1.2 Temporal Trend Analysis

| Analysis | Method | Output |
|----------|--------|--------|
| Year-over-year means | Group by year, plot line | Trend direction per city |
| Monthly averages | Group by month, heatmap | Seasonal pattern |
| Day-of-week patterns | Group by weekday, bar chart | Weekly cycle |
| Moving average (30, 90, 365 day) | `rolling().mean()` | Smooth trend line |

**Key questions:**
- Is Delhi improving or worsening year-over-year?
- Do weekends have lower AQI (less traffic/industry)?
- What months have peak pollution across cities?

## 2. Bivariate Analysis

### 2.1 Pollutant Correlations

| Analysis | Method | Output |
|----------|--------|--------|
| Pairwise correlation matrix | `pandas.DataFrame.corr()` | Heatmap across all 12 pollutants |
| Scatter matrix | `seaborn.pairplot()` | 12x12 scatter grid |
| Spearman rank correlation | `scipy.stats.spearmanr()` | Non-linear relationships |

**Key questions:**
- Which pollutants are most correlated with AQI? (Hypothesis: PM2.5 dominates)
- Are PM2.5 and PM10 perfectly correlated? (Should be high but not 1.0)
- Are NO2 and CO correlated (both traffic-related)?
- Which pollutants are redundant and could be dropped?

### 2.2 Pollutant vs. Time

| Analysis | Method | Output |
|----------|--------|--------|
| PM2.5 vs time | Line plot per city | Trend + seasonal |
| NO2 vs time | Line plot per city | Traffic-related trend |
| O3 vs time | Line plot per city | Photochemical pattern (summer peaks) |

**Key questions:**
- Do different pollutants peak in different seasons?
- Has NO2 decreased during COVID lockdown (2020)?
- Is the PM2.5/PM10 ratio stable across seasons? (Dust vs combustion)

## 3. Multi-City Comparative Analysis

### 3.1 City Rankings

| Analysis | Method | Output |
|----------|--------|--------|
| Mean AQI ranking | Bar chart (sorted) | Cleanest → most polluted |
| Seasonal pattern similarity | Cross-correlation | City clustering by pattern |
| Pollution volatility | Coefficient of variation | Most/least predictable city |
| Extreme event frequency | Count of days AQI > 300 per city | Severe pollution days |

**Key questions:**
- Which cities form natural clusters? (Indo-Gangetic plain vs coastal vs southern?)
- Is intra-city variance higher than inter-city variance?

### 3.2 Geographic Analysis

| Analysis | Method | Output |
|----------|--------|--------|
| State-level aggregation | Group by State from stations.csv | Regional patterns |
| Station-level variation within city | Station-day data | Within-city pollution gradients |
| City clustering by pollution profile | PCA + K-means | City typology (traffic-dominant vs industrial vs clean) |

**Key questions:**
- Are coastal cities systematically cleaner?
- Do cities in the same state have similar pollution profiles?
- Is intra-city variance significant enough for hyper-local forecasting?

## 4. Missing Data Analysis

### 4.1 Gap Analysis

| Analysis | Method | Output |
|----------|--------|--------|
| Null% by city by pollutant | Heatmap | Which cities lack which variables |
| Run-length of missing sequences | Consecutive NaN detection | Are gaps random or systemic? |
| Seasonality of missing data | Month-of-year vs null rate | Are gaps seasonal? (e.g., monsoon instrument failure) |

**Key questions:**
- Why does Mumbai have only 38% AQI coverage?
- Are missing values MCAR, MAR, or NMAR?
- Is imputation feasible? (For which cities/pollutants?)

## 5. Advanced Analysis

### 5.1 Anomaly Detection

| Method | Approach | Output |
|--------|----------|--------|
| Z-score | Per city, per pollutant | Point anomalies |
| IQR-based | Box plot whisker method | Outlier days |
| STL decomposition | `statsmodels.tsa.seasonal.STL` | Residual anomalies |
| Isolation Forest | Unsupervised anomaly detection | Multi-pollutant anomaly score |

**Key questions:**
- Do identified anomalies correspond to known events? (Diwali, COVID lockdown, stubble burning season)
- Can anomaly score be used as a risk indicator?

### 5.2 Time-Series Decomposition

| Method | Output |
|--------|--------|
| Additive decomposition (STL) | Trend + Seasonal + Residual per city |
| Multiple seasonal periods | Yearly + weekly cycles |
| Changepoint detection | Detect policy/event impact points |

**Key questions:**
- Is the trend truly linear or non-linear?
- When did COVID lockdown affect AQI (March 2020)?
- Are there structural breaks in the time series?

### 5.3 Station-Level Spatial Analysis

| Method | Output |
|--------|--------|
| Station ranking within city | Mean AQI per station for Delhi |
| Station correlation matrix | Are stations within a city highly correlated? |
| Distance vs AQI correlation | (If station coordinates available) |

**Key questions:**
- Does a single station represent a city or are there micro-climates?
- Can we interpolate between stations for unmeasured areas?

## 6. Implementation Priority

| Phase | Analyses | Effort | Value |
|-------|----------|--------|-------|
| 1 | Distributions, trends, correlations | 2 days | High |
| 2 | City rankings, seasonal heatmaps, missing data | 2 days | High |
| 3 | Anomaly detection, decomposition | 3 days | Medium |
| 4 | Geographic/station analysis, spatial patterns | 3 days | Medium |

## 7. Outputs

Each EDA phase produces:
- A Jupyter notebook in `notebooks/eda/`
- Static charts in `outputs/eda/`
- A summary section to incorporate into dashboard pages

## 8. Code Structure

```
notebooks/eda/
├── 01_distributions.ipynb       # Univariate analysis
├── 02_temporal_trends.ipynb     # Trends and seasonality
├── 03_correlations.ipynb        # Pollutant relationships
├── 04_city_comparison.ipynb     # Multi-city analysis
├── 05_missing_data.ipynb        # Data quality
├── 06_anomaly_detection.ipynb   # Outlier analysis
├── 07_decomposition.ipynb       # STL and changepoints
└── 08_station_analysis.ipynb    # Geographic patterns
```
