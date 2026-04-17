#!/usr/bin/env python3
"""
Prophet Model Validation with Cross-Validation
Compares: Basic vs Pre-COVID vs Full Dataset models
"""

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np

def load_data(city='Hyderabad'):
    """Load full 2015-2024 data from PostgreSQL"""
    engine = create_engine('postgresql://postgres@localhost:5432/india_air_quality')
    
    df = pd.read_sql(f"""
        SELECT date as ds, aqi as y
        FROM city_day
        WHERE city = '{city}' AND aqi IS NOT NULL
        ORDER BY date
    """, engine)
    
    df['ds'] = pd.to_datetime(df['ds'])
    print(f"Loaded {len(df)} days for {city} ({df['ds'].min().date()} to {df['ds'].max().date()})")
    return df

def calculate_mape(actual, predicted):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def evaluate_model(df, model_name, cutoff_date):
    """
    Train on data before cutoff, validate on after
    Returns metrics dictionary
    """
    # Split
    train = df[df['ds'] < cutoff_date].copy()
    test = df[df['ds'] >= cutoff_date].copy()
    
    if len(test) == 0:
        print(f"  ⚠️  No test data after {cutoff_date}")
        return None
    
    print(f"  Train: {len(train)} days | Test: {len(test)} days")
    
    # Train Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(train)
    
    # Predict on test period
    future = model.make_future_dataframe(periods=len(test), freq='D')
    forecast = model.predict(future)
    
    # Extract test period predictions
    predictions = forecast[forecast['ds'].isin(test['ds'])][['ds', 'yhat']]
    
    # Merge with actual
    results = test.merge(predictions, on='ds', how='left')
    
    # Calculate metrics
    mape = calculate_mape(results['y'], results['yhat'])
    rmse = np.sqrt(np.mean((results['y'] - results['yhat'])**2))
    mae = np.mean(np.abs(results['y'] - results['yhat']))
    
    print(f"  MAPE: {mape:.1f}% | RMSE: {rmse:.1f} | MAE: {mae:.1f}")
    
    return {
        'model_name': model_name,
        'train_days': len(train),
        'test_days': len(test),
        'mape': mape,
        'rmse': rmse,
        'mae': mae,
        'predictions': results
    }

def prophet_cross_validation(df, model_name):
    """
    Prophet's built-in cross-validation
    """
    print(f"\n{'='*60}")
    print(f"Cross-Validation: {model_name}")
    print(f"{'='*60}")
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(df)
    
    # Cross-validation: 365-day forecasts, every 180 days
    df_cv = cross_validation(
        model, 
        initial='1825 days',      # 5 years training
        period='365 days',        # New forecast every year
        horizon='365 days',       # 1-year forecast horizon
        parallel="processes"
    )
    
    # Performance metrics
    df_p = performance_metrics(df_cv)
    
    print(f"\nCross-Validation Results (365-day horizon):")
    print(f"  MAPE: {df_p['mape'].mean():.1f}% (±{df_p['mape'].std():.1f}%)")
    print(f"  RMSE: {df_p['rmse'].mean():.1f} (±{df_p['rmse'].std():.1f})")
    print(f"  Coverage (yhat_lower/upper): {df_p['coverage'].mean():.2f}")
    
    return df_cv, df_p

def main():
    print("="*60)
    print("PROPHET MODEL VALIDATION")
    print("Hyderabad AQI Forecasting")
    print("="*60)
    
    # Load data
    df = load_data('Hyderabad')
    
    # Model 1: Train on all data, validate on 2023-2024
    print(f"\n{'='*60}")
    print("Model 1: Full Dataset (2015-2022 train, 2023-2024 test)")
    print(f"{'='*60}")
    results1 = evaluate_model(df, 'Full Dataset', '2023-01-01')
    
    # Model 2: Pre-COVID only (2015-2020 train, 2020-2024 test - unrealistic but interesting)
    print(f"\n{'='*60}")
    print("Model 2: Pre-COVID Baseline (2015-2020:03 train)")
    print(f"{'='*60}")
    df_precovid = df[df['ds'] < '2020-07-01'].copy()
    results2 = evaluate_model(df_precovid, 'Pre-COVID', '2023-01-01')
    
    # Model 3: Exclude COVID period (2015-2020 + 2021-2022 train, 2023-2024 test)
    print(f"\n{'='*60}")
    print("Model 3: Skip COVID (excludes 2020-04 to 2021-06)")
    print(f"{'='*60}")
    df_nocovid = df[(df['ds'] < '2020-03-01') | (df['ds'] > '2021-06-30')].copy()
    results3 = evaluate_model(df_nocovid, 'Skip COVID', '2023-01-01')
    
    # Cross-validation on best model
    print(f"\n{'='*60}")
    print("DETAILED CROSS-VALIDATION (Full Dataset)")
    print(f"{'='*60}")
    cv_results, cv_metrics = prophet_cross_validation(df, 'Full Dataset')
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'MAPE':<10} {'RMSE':<10} {'MAE':<10}")
    print("-"*60)
    for r in [results1, results2, results3]:
        if r:
            print(f"{r['model_name']:<20} {r['mape']:<10.1f} {r['rmse']:<10.1f} {r['mae']:<10.1f}")
    
    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    best = min([r for r in [results1, results2, results3] if r], key=lambda x: x['mape'])
    print(f"Best model: {best['model_name']} (MAPE: {best['mape']:.1f}%)")
    print(f"\nFor 2030 forecasting:")
    print(f"  - Expected error: ±{best['mape']:.0f}%")
    print(f"  - If 2030 prediction is 100 AQI:")
    print(f"    True value likely between {100*(1-best['mape']/100):.0f} and {100*(1+best['mape']/100):.0f}")
    
    # Save validation results
    cv_results.to_csv('outputs/prophet_cross_validation.csv', index=False)
    print(f"\n✓ Saved: outputs/prophet_cross_validation.csv")

if __name__ == '__main__':
    main()
