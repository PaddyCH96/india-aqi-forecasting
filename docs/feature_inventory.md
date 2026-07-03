# Feature Inventory

## 1. Available Features by Category

### 1.1 Pollutant Features (Direct Measurements)

| Feature | Source | Type | Missing % | ML Suitability |
|---------|--------|------|-----------|----------------|
| `pm2_5` | city_measurements | Continuous | 15.6% | ★★★★★ Excellent |
| `pm10` | city_measurements | Continuous | 37.7% | ★★★☆☆ Moderate |
| `no` | city_measurements | Continuous | 12.1% | ★★★★☆ Good |
| `no2` | city_measurements | Continuous | 12.1% | ★★★★☆ Good |
| `nox` | city_measurements | Continuous | 14.2% | ★★★★☆ Good |
| `nh3` | city_measurements | Continuous | 35.0% | ★★☆☆☆ Limited |
| `co` | city_measurements | Continuous | 7.0% | ★★★★★ Excellent |
| `so2` | city_measurements | Continuous | 13.1% | ★★★★☆ Good |
| `o3` | city_measurements | Continuous | 13.6% | ★★★★☆ Good |
| `benzene` | city_measurements | Continuous | 19.0% | ★★★☆☆ Moderate |
| `toluene` | city_measurements | Continuous | 27.2% | ★★☆☆☆ Limited |
| `xylene` | city_measurements | Continuous | 61.3% | ★☆☆☆☆ Poor |

### 1.2 Computed Features (Available in Source)

| Feature | Source | Type | Description |
|---------|--------|------|-------------|
| `aqi` | city_measurements | Continuous | Composite index (0-500+) |
| `aqi_bucket` | city_measurements | Categorical (6) | Good/Satisfactory/Moderate/Poor/Very Poor/Severe |

### 1.3 Temporal Features (Derivable)

| Feature | Derivation | Type | ML Suitability |
|---------|-----------|------|----------------|
| `year` | date.year | Discrete | ★★★★☆ Good |
| `month` | date.month | Cyclical (12) | ★★★★★ Excellent |
| `day_of_week` | date.dayofweek | Cyclical (7) | ★★★★☆ Good |
| `day_of_year` | date.dayofyear | Cyclical (365) | ★★★☆☆ Moderate |
| `quarter` | date.quarter | Discrete (4) | ★★★☆☆ Moderate |
| `is_weekend` | day_of_week ≥ 5 | Binary | ★★★☆☆ Moderate |
| `days_since_start` | (date - min_date).days | Continuous | ★★☆☆☆ Limited |
| `month_sin` | sin(2π·month/12) | Continuous | ★★★★☆ Good |
| `month_cos` | cos(2π·month/12) | Continuous | ★★★★☆ Good |
| `dow_sin` | sin(2π·dow/7) | Continuous | ★★★☆☆ Moderate |
| `dow_cos` | cos(2π·dow/7) | Continuous | ★★★☆☆ Moderate |

### 1.4 Lag Features (Derivable)

| Feature | Derivation | ML Suitability |
|---------|-----------|----------------|
| `aqi_lag1` | AQI(t-1) | ★★★★★ Excellent |
| `aqi_lag7` | AQI(t-7) | ★★★★☆ Good |
| `aqi_lag30` | AQI(t-30) | ★★★☆☆ Moderate |
| `aqi_lag365` | AQI(t-365) | ★★★☆☆ Moderate |
| `pm25_lag1` | PM2.5(t-1) | ★★★★★ Excellent |
| `no2_lag1` | NO2(t-1) | ★★★★☆ Good |
| `co_lag1` | CO(t-1) | ★★★★☆ Good |

### 1.5 Rolling Window Features (Derivable)

| Feature | Derivation | ML Suitability |
|---------|-----------|----------------|
| `aqi_roll7_mean` | 7-day rolling mean | ★★★★☆ Good |
| `aqi_roll7_std` | 7-day rolling std | ★★★☆☆ Moderate |
| `aqi_roll30_mean` | 30-day rolling mean | ★★★★☆ Good |
| `aqi_roll30_std` | 30-day rolling std | ★★★☆☆ Moderate |
| `aqi_roll7_max` | 7-day rolling max | ★★★☆☆ Moderate |
| `aqi_roll7_min` | 7-day rolling min | ★★☆☆☆ Limited |

### 1.6 Geographic Features (Derivable)

| Feature | Source | ML Suitability |
|---------|--------|----------------|
| `city` | city_measurements | ★★★★☆ Good (as categorical) |
| `state` | stations table | ★★★☆☆ Moderate |
| `region` | Manual grouping (North/South/East/West) | ★★☆☆☆ Limited |
| `station_count` | stations per city | ★★☆☆☆ Limited |

### 1.7 Target Variables

| Target | Type | Use Case | ML Suitability |
|--------|------|----------|----------------|
| `aqi` (t+1) | Regression | Next-day AQI prediction | ★★★★★ Excellent |
| `aqi` (t+7) | Regression | Weekly AQI prediction | ★★★★☆ Good |
| `aqi` (t+30) | Regression | Monthly AQI prediction | ★★★☆☆ Moderate |
| `aqi` (t+365) | Regression | Yearly AQI prediction | ★★☆☆☆ Limited |
| `aqi_bucket` (t+1) | Classification (6-class) | Next-day risk category | ★★★★★ Excellent |
| `aqi_bucket` (binary) | Classification (2-class) | Safe/Unsafe (Moderate threshold) | ★★★★☆ Good |
| `anomaly` (t) | Binary | Anomaly detection | ★★★☆☆ Moderate |

## 2. Recommended Feature Set for ML

### Tier 1: Essential (Use These First)

```python
FEATURES_TIER_1 = [
    # Temporal
    'month_sin', 'month_cos',
    'day_of_week',
    'is_weekend',
    # Pollutant lags
    'pm2_5_lag1', 'no2_lag1', 'co_lag1', 'o3_lag1',
    'aqi_lag1', 'aqi_lag7',
    # Rolling
    'aqi_roll7_mean', 'aqi_roll7_std',
    # Identity
    'city',  # one-hot encoded
]
```

**Target**: `aqi` (t+1) — next-day AQI regression
**Expected Input Dim**: ~14-20 (depending on city encoding)
**Expected Performance**: MAPE < 12%

### Tier 2: Extended (For Maximum Performance)

```python
FEATURES_TIER_2 = FEATURES_TIER_1 + [
    # Additional pollutant lags
    'pm10_lag1', 'so2_lag1', 'no_lag1',
    'benzene_lag1',
    # Additional rolling
    'aqi_roll30_mean', 'aqi_roll30_std',
    'pm25_roll7_mean',
    # Extended temporal
    'aqi_lag365',
    'quarter',
    # Additional context
    'aqi_roll7_max',
]
```

**Expected Input Dim**: ~25-32
**Expected Performance**: MAPE < 10%

## 3. Feature Engineering Pipeline (Proposed)

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['city', 'date'])

    # Temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features (per city)
    for col in ['aqi', 'pm2_5', 'no2', 'co', 'o3', 'pm10', 'so2', 'no']:
        for lag in [1, 7, 30, 365]:
            df[f'{col}_lag{lag}'] = df.groupby('city')[col].shift(lag)

    # Rolling features
    for col in ['aqi', 'pm2_5']:
        for window in [7, 30]:
            df[f'{col}_roll{window}_mean'] = (
                df.groupby('city')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            )
            df[f'{col}_roll{window}_std'] = (
                df.groupby('city')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
            )

    return df
```

## 4. Feature Importance Estimates (Hypothesized)

Based on domain knowledge of air quality dynamics:

| Rank | Feature | Expected Importance | Why |
|------|---------|-------------------|-----|
| 1 | `aqi_lag1` | Very High | Today's AQI is the best predictor of tomorrow's |
| 2 | `pm2_5_lag1` | High | PM2.5 is the dominant AQI component |
| 3 | `aqi_lag7` | High | Weekly pattern persistence |
| 4 | `aqi_roll7_mean` | High | Recent trend direction |
| 5 | `month_sin/cos` | Medium | Seasonal patterns |
| 6 | `city` (encoded) | Medium | Different cities have different baselines |
| 7 | `no2_lag1` | Medium | Traffic-related pollution persistence |
| 8 | `co_lag1` | Medium | Combustion source indicator |
| 9 | `o3_lag1` | Low-Medium | Photochemical — forms differently |
| 10 | `is_weekend` | Low | Small weekend effect |

## 5. Data Requirements by Model Type

| Model Type | Min Rows | Features Needed | Missing Data Handling |
|-----------|---------|----------------|----------------------|
| Prophet (current) | 365 | Date + AQI only | None required (univariate) |
| XGBoost (next-day) | 5,000 | 14-20 features | Tree-based handles NaN natively |
| Random Forest | 3,000 | 14-20 features | Tree-based handles NaN natively |
| Linear Regression | 1,000 | 14-20 features | Requires imputation |
| LSTM (hourly) | 50,000 | 6-12 features | Requires interpolation |
| Isolation Forest | 1,000 | 12 features | Handles NaN with imputation |

## 6. Feature Availability by Dataset Level

| Feature | City-Daily (29K) | City-Hourly (708K) | Station-Daily (108K) |
|---------|-----------------|-------------------|---------------------|
| All pollutants | ✅ Yes (some missing) | ✅ Yes (higher missing) | ✅ Yes |
| Temporal | ✅ Yes | ✅ Yes (hourly) | ✅ Yes |
| Geographic (city) | ✅ Yes | ✅ Yes | ❌ Station-level |
| Lag features | ✅ Yes | ✅ Yes (more lags) | ✅ Yes |
| Multi-city | ✅ Yes (26) | ✅ Yes (26) | ✅ Yes (110 stations) |
| Station metadata | ❌ N/A | ❌ N/A | ✅ Yes (via stations table) |
