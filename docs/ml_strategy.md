# ML Strategy: india-air-quality

## Data Context

- **Real data**: 26 cities, 12+ pollutants, 2015-01-01 to 2020-07-01 (daily + hourly)
- **Synthetic data**: 6 cities, AQI only, 2020-07-01 to 2024-12-31
- **Training horizon**: 5.5 years (marginal for annual seasonality capture)
- **Current baseline**: Prophet with yearly seasonality only (MAPE 13-16%)

## Opportunity Matrix

| ML Task | Features | Algorithms | Data Needs | Difficulty | Portfolio Value |
|---------|----------|-----------|------------|-----------|-----------------|
| Next-day AQI prediction | 12 pollutants + temporal (day-of-week, month) | XGBoost, LightGBM, Random Forest | Existing daily data (29K rows) | Easy | High |
| 7-day forecasting | Lag features (t-1 to t-7), rolling stats | XGBoost, LSTM | Existing daily data | Easy | High |
| 24-hour forecasting (hourly) | Hourly pollutants + time features | XGBoost, LSTM | city_hour (708K rows) | Medium | High |
| 72-hour forecasting | Hourly lags, diurnal patterns | LSTM, GRU | city_hour | Medium | Medium |
| Weekly forecasting | Aggregated weekly means | Prophet, ARIMA, XGBoost | Aggregated daily data | Easy | Medium |
| Anomaly detection | All pollutants as features | Isolation Forest, Autoencoder | Existing daily data | Medium | High |
| AQI bucket classification | 12 pollutants + temporal | XGBoost classifier, Random Forest | Existing daily + labels | Easy | Medium |
| Multi-city ensemble | City as categorical feature | XGBoost, Stacking | Combined 26-city data (29K rows) | Medium | High |
| Pollutant imputation | Cross-pollutant correlations | KNN, MICE, Matrix Factorization | Station data (108K rows) | Medium | Medium |
| Changepoint detection | AQI sequence + covariates | RBF, Bayesian changepoint | Existing daily data | Hard | Medium |

## Recommended ML Pipeline

### Phase 1: Baseline Improvements (Easy, High Value)

#### Next-Day AQI Prediction (XGBoost)

**Features:**
- t-1: PM2.5, PM10, NO2, CO, O3, SO2, NH3 (lag1)
- t-7 rolling mean of AQI
- Day of week (0-6), Month (1-12), Quarter (1-4)
- Day of year (sin/cos encoding)
- Is weekend (binary)
- Is holiday (binary, if available)

**Target:** AQI (regression) or AQI_Bucket (classification)

**Architecture:**
```python
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20
)
```

**Expected performance:** MAPE < 10% (improvement over Prophet's 13-16%)

**Why this matters for portfolio:**
- Demonstrates feature engineering on real multi-pollutant data
- Shows understanding of time-series cross-validation
- Comparisons against Prophet baseline show ML awareness
- Easy to visualize results (predicted vs actual scatter plot)

#### AQI Bucket Classification

**Features:** Same as above + all lagged pollutants

**Target:** AQI_Bucket (6 classes)

**Model:** XGBoostClassifier or RandomForestClassifier

**Metric:** Accuracy, F1 per class, confusion matrix

**Why:** Real-world use case for health alerts. Easy to show in dashboard.

### Phase 2: Deep Learning (Medium Difficulty)

#### Multi-Step Hourly Forecasting (LSTM)

**Features:**
- Sequence of past 72 hours of PM2.5, PM10, NO2, CO, O3
- Diurnal encoding (hour of day sin/cos)

**Architecture:**
```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(72, n_features)),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(24)  # predict next 24 hours
])
```

**Data:** city_hour.csv (708K hourly + station_hour ~4M)

**Challenge:** Hourly data has higher missing rates; requires interpolation.

**Why:** Demonstrates deep learning on time series. High portfolio signal.

#### Multi-City Ensemble

**Architecture:**
```python
# City-specific models
city_models = {city: XGBRegressor() for city in cities}
# OR
# Single model with city embedding
model = XGBRegressor()  # city as categorical feature
```

**Feature:** City ID as categorical (one-hot or label encoded)

**Why:** Shows you can handle structured multi-entity data. Practical scaling strategy.

### Phase 3: Advanced Topics (Hard)

#### Anomaly Detection

**Approach 1: Isolation Forest**
```python
from sklearn.ensemble import IsolationForest

# Fit on all 12 pollutants
iso_forest = IsolationForest(contamination=0.05)
anomaly_score = iso_forest.fit_predict(X)
```

**Approach 2: Autoencoder**
```python
# Reconstruct normal pollution patterns
# High reconstruction error = anomaly
autoencoder.fit(X_normal)
anomalies = mse(X, autoencoder.predict(X)) > threshold
```

**Why:** Portfolio differentiation — most AQI projects don't do anomaly detection.

**Detection targets:**
- Diwali firecracker spikes (Oct/Nov)
- Stubble burning events (Oct/Nov in Delhi)
- COVID lockdown drop (March 2020)
- Dust storms (pre-monsoon)

#### Changepoint Detection

**Method:** `ruptures` library or Bayesian changepoint detection

**Expected changepoints:**
- March 2020 (COVID lockdown)
- November policy interventions (odd-even, stubble burning bans)

**Why:** Shows causal reasoning ability.

## Model Evaluation Framework

### Metrics

| Task | Primary | Secondary |
|------|---------|-----------|
| AQI Regression | MAPE | RMSE, MAE, R² |
| AQI Classification | Accuracy | F1 (per class), Precision, Recall |
| Forecasting | MAPE (on test period) | Coverage (80% CI), MASE |
| Anomaly Detection | F1 | Precision, Recall at various thresholds |

### Validation Strategy

| Method | When | Why |
|--------|------|-----|
| Time-series split | All forecasting tasks | Prevents lookahead bias |
| Expanding window | Prophet/XGBoost comparison | Mimics production retraining |
| GroupKFold (by city) | Multi-city models | Tests generalization to unseen cities |
| Blocked cross-validation | Changepoint detection | Preserves temporal structure |

### Baseline Models

| Task | Baseline | Reason |
|------|----------|--------|
| AQI prediction | Persistence (t-1) | Simplest baseline |
| AQI prediction | Prophet (current) | Project current state |
| AQI classification | Majority class | Trivial baseline |
| Anomaly | Z-score > 3 | Classical approach |

## Expected Difficulty & Timeline

| Phase | Tasks | Timeline | Prerequisites |
|-------|-------|----------|--------------|
| 1 | XGBoost next-day, category classification | 1 week | Data cleaning |
| 2 | LSTM hourly, multi-city ensemble | 2 weeks | Phase 1 + hourly preprocessing |
| 3 | Anomaly detection, changepoints | 2 weeks | Phase 1 + signal knowledge |

## Portfolio Impact Assessment

| ML Task | Portfolio Signal | Why |
|---------|-----------------|-----|
| XGBoost + Prophet comparison | High | Shows model selection awareness |
| LSTM hourly forecasting | High | Deep learning on real sensor data |
| Anomaly detection | High | Differentiator from basic forecasting |
| Multi-city ensemble | Medium | Shows scalability thinking |
| Changepoint analysis | Medium | Shows causal reasoning |
| Pollutant imputation | Low-Niche | Only valuable for certain roles |

## Recommendation

**Build Phase 1 first** (XGBoost next-day + AQI classification). It uses existing data, requires no new data collection, and produces immediately visualizable results that compare favorably against the current Prophet baseline. This single comparison (Prophet MAPE 15% → XGBoost MAPE <10%) tells a compelling portfolio story.

**Add Phase 2** (LSTM hourly) only if you want to demonstrate deep learning. This requires significant preprocessing but shows end-to-end data engineering + DL skills.

**Add Phase 3** (anomaly detection) as a differentiator. Map detected anomalies to real-world events (Diwali, COVID) to demonstrate domain understanding.
