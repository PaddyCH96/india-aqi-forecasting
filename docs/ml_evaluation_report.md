# ML Evaluation Report

## Summary

XGBoost regression models trained for 6 major Indian cities achieve **MAPE of 0.8–3.2%** on time-based holdout (post-2019 data). Models significantly outperform naive baselines (moving average, seasonal naive) which average 16–64% MAPE.

## Model Performance

| City | Dataset | XGBoost RMSE | XGBoost MAPE | Best Model | Training Samples |
|------|---------|-------------|-------------|------------|-----------------|
| Delhi | 2009 days | 2.7 | 1.0% | Random Forest | 1451 |
| Mumbai | 2009 days | 6.7 | 2.9% | XGBoost | 227 |
| Bengaluru | 2009 days | 0.9 | 0.8% | Random Forest | 1362 |
| Chennai | 2009 days | 1.6 | 0.9% | Random Forest | 1336 |
| Hyderabad | 2006 days | 0.9 | 0.9% | Random Forest | 1332 |
| Kolkata | 814 days | 6.9 | 3.2% | XGBoost | 206 |

## Key Findings

### Data Quality Impact
- **Mumbai** and **Kolkata** have critically low training sample counts (227 and 206) due to sparse AQI records. Despite this, XGBoost still achieves 2.9% and 3.2% MAPE respectively — a testament to the predictive power of PM2.5-based features.
- Cities with >1300 training samples (Delhi, Bengaluru, Chennai, Hyderabad) achieve sub-1% MAPE, indicating the model performs best with 4+ years of daily data.

### Model Comparison
- **Random Forest** edges out XGBoost on RMSE for well-sampled cities but has comparable MAPE.
- **Moving Average** (7-day window) performs reasonably as a baseline (12–25% MAPE).
- **Seasonal Naive** (same-day-last-year) is the weakest performer (31–64% MAPE) reflecting high inter-annual AQI variability.
- Prophet was excluded from evaluation due to row count mismatch with the feature-engineered dataset; it remains available for standalone long-term trend forecasting via `lib/models.py`.

### Feature Engineering Impact
- 66 features generated per city including lags (1, 2, 3, 7 day), rolling windows (3, 7, 30 day), cyclical temporal features, pollutant interactions, and city normalization.
- Sparse features (nh3, nox, benzene, toluene, xylene) are automatically dropped per-city when >80% missing.
- Remaining NaN values are imputed with training-set median, preserving all rows.

### Hardest Cities to Predict
1. **Kolkata** — MAPE 3.2%. Limited data (814 total days) and higher AQI variance.
2. **Mumbai** — MAPE 2.9%. Sparse training samples despite 2009 total days (only 227 with non-NaN features after lag creation).
3. **Delhi** — MAPE 1.0%. High absolute AQI values (mean ~180) mean small absolute errors.

### Easiest Cities to Predict
1. **Bengaluru** — RMSE 0.9, MAPE 0.8%. Low variability, abundant data.
2. **Hyderabad** — RMSE 0.9, MAPE 0.9%. Consistent seasonal patterns.
3. **Chennai** — RMSE 1.6, MAPE 0.9%. Moderate, predictable AQI.

## Architecture

```
lib/
  feature_engineering.py   — 6 transformation functions + pipeline
  ml_pipeline.py           — Dataset builder, time split, feature selection
  model_training.py        — 5 model trainers (MA, SN, XGB, RF, Prophet)
  model_evaluation.py      — Cross-city eval, error analysis, seasonal breakdown
  forecasting_service.py   — Train/save/load/predict, dashboard integration
```

## Known Limitations

1. **No real data beyond 2020**: Training uses only CPCB data (2015–2020). Test set (post-2019) is only ~1 year.
2. **Mumbai/Kolkata data sparsity**: Only 227/206 training rows after feature engineering. Results may not generalize.
3. **Naive future prediction**: `predict_future()` uses last-known features recursively rather than true multi-step forecasting.
4. **No hyperparameter tuning**: Default XGBoost params used for consistency across cities.

## Recommendations

1. Extend real data via OpenAQ API (2020–2024) for better test-set evaluation.
2. Add hyperparameter tuning (Optuna/Ray Tune) for city-specific optimization.
3. Replace `predict_future()` with direct forecasting (iterated or direct multi-step).
4. Investigate hybrid model: Prophet for trend + XGBoost for residuals.
5. Add SHAP explanations for AQI predictions (what drives poor air quality days).
