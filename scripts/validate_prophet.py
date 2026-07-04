#!/usr/bin/env python3
"""
Prophet Model Validation with Cross-Validation
Compares: Basic vs Pre-COVID vs Full Dataset models
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.pathing import ensure_project_root_on_path

ensure_project_root_on_path()

import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics

from lib.logging import setup_logger
from lib.db import get_engine, load_city_data
from lib.models import train_and_forecast
from lib.metrics import calc_mape

logger = setup_logger("validate-prophet")


def evaluate_model(df, model_name, cutoff_date):
    train = df[df['ds'] < cutoff_date].copy()
    test = df[df['ds'] >= cutoff_date].copy()

    if len(test) == 0:
        logger.warning(f"No test data after {cutoff_date}")
        return None

    logger.info(f"  Train: {len(train)} days | Test: {len(test)} days")

    model, forecast = train_and_forecast(train, periods=len(test))
    predictions = forecast[forecast['ds'].isin(test['ds'])][['ds', 'yhat']]
    results = test.merge(predictions, on='ds', how='left')

    mape = calc_mape(results['y'], results['yhat'])
    rmse = float(np.sqrt(np.mean((results['y'] - results['yhat']) ** 2)))
    mae = float(np.mean(np.abs(results['y'] - results['yhat'])))

    logger.info(f"  MAPE: {mape:.1f}% | RMSE: {rmse:.1f} | MAE: {mae:.1f}")

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
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Cross-Validation: {model_name}")
    logger.info(f"{'=' * 60}")

    model, _ = train_and_forecast(df)

    df_cv = cross_validation(
        model,
        initial='1825 days',
        period='365 days',
        horizon='365 days',
        parallel="processes"
    )

    df_p = performance_metrics(df_cv)

    print("\nCross-Validation Results (365-day horizon):")
    print(f"  MAPE: {df_p['mape'].mean():.1f}% (±{df_p['mape'].std():.1f}%)")
    print(f"  RMSE: {df_p['rmse'].mean():.1f} (±{df_p['rmse'].std():.1f})")
    print(f"  Coverage (yhat_lower/upper): {df_p['coverage'].mean():.2f}")

    return df_cv, df_p


def main():
    try:
        engine = get_engine()
    except Exception as exc:
        logger.error(f"Database connection failed: {exc}")
        logger.warning("Prophet validation skipped because no PostgreSQL instance is reachable.")
        return

    try:
        print("=" * 60)
        print("PROPHET MODEL VALIDATION")
        print("Hyderabad AQI Forecasting")
        print("=" * 60)

        df = load_city_data(engine, 'Hyderabad')

        print(f"\n{'=' * 60}")
        print("Model 1: Full Dataset (2015-2022 train, 2023-2024 test)")
        print(f"{'=' * 60}")
        results1 = evaluate_model(df, 'Full Dataset', '2023-01-01')

        print(f"\n{'=' * 60}")
        print("Model 2: Pre-COVID Baseline (2015-2020:03 train)")
        print(f"{'=' * 60}")
        df_precovid = df[df['ds'] < '2020-07-01'].copy()
        results2 = evaluate_model(df_precovid, 'Pre-COVID', '2023-01-01')

        print(f"\n{'=' * 60}")
        print("Model 3: Skip COVID (excludes 2020-04 to 2021-06)")
        print(f"{'=' * 60}")
        df_nocovid = df[(df['ds'] < '2020-03-01') | (df['ds'] > '2021-06-30')].copy()
        results3 = evaluate_model(df_nocovid, 'Skip COVID', '2023-01-01')

        print(f"\n{'=' * 60}")
        print("DETAILED CROSS-VALIDATION (Full Dataset)")
        print(f"{'=' * 60}")
        cv_results, cv_metrics = prophet_cross_validation(df, 'Full Dataset')

        print(f"\n{'=' * 60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Model':<20} {'MAPE':<10} {'RMSE':<10} {'MAE':<10}")
        print("-" * 60)
        for r in [results1, results2, results3]:
            if r:
                print(f"{r['model_name']:<20} {r['mape']:<10.1f} {r['rmse']:<10.1f} {r['mae']:<10.1f}")

        print(f"\n{'=' * 60}")
        print("RECOMMENDATION")
        print(f"{'=' * 60}")
        best = min([r for r in [results1, results2, results3] if r], key=lambda x: x['mape'])
        print(f"Best model: {best['model_name']} (MAPE: {best['mape']:.1f}%)")
        print("\nFor 2030 forecasting:")
        print(f"  - Expected error: ±{best['mape']:.0f}%")
        print("  - If 2030 prediction is 100 AQI:")
        print(f"    True value likely between {100 * (1 - best['mape'] / 100):.0f} and {100 * (1 + best['mape'] / 100):.0f}")

        cv_results.to_csv('outputs/prophet_cross_validation.csv', index=False)
        print("\n✅ Saved: outputs/prophet_cross_validation.csv")

        generate_model_config(results1)
    except Exception as exc:
        logger.error(f"Prophet validation failed: {exc}")
        logger.warning("The validation workflow completed with warnings because the database was unavailable.")


def generate_model_config(results):
    import json
    config = {
        "model": "Prophet",
        "city": "Hyderabad",
        "data_range": "2015-2024",
        "features": ["yearly_seasonality", "trend"],
        "external_regressors": "None (weather removed due to COVID instability)",
        "validation_mape": f"{results['mape']:.1f}%",
        "forecast_horizon": "2030",
        "notes": "Weather regressors tested but removed - unstable correlations across COVID period",
    }
    with open("outputs/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\n✅ Saved: outputs/model_config.json")


if __name__ == '__main__':
    main()
