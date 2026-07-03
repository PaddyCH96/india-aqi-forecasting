"""Unified model evaluation framework for AQI forecasting.

Supports side-by-side comparison across models, error analysis,
and per-city / per-season breakdown.
"""

import numpy as np
import pandas as pd

from lib.ml_pipeline import time_based_split, prepare_ml_data
from lib.model_training import (
    train_moving_average,
    train_seasonal_naive,
    train_xgboost,
    train_random_forest,
    train_prophet,
)


def evaluate_all_models(
    df: pd.DataFrame,
    target: str = "aqi",
    cutoff_date: str = "2019-01-01",
    include_baselines: bool = True,
    include_ml: bool = True,
    include_prophet: bool = True,
    xgboost_params: dict | None = None,
) -> dict:
    """Run all applicable models and return unified results dict.

    Returns:
        {
            "city": ...,
            "cutoff_date": ...,
            "models": {
                "Moving Average": {metrics, predictions...},
                "Seasonal Naive": {...},
                "XGBoost": {...},
                "Random Forest": {...},
                "Prophet": {...},
            },
            "best_model": "XGBoost",
            "summary": {model: mape, ...}
        }
    """
    city = df["city"].iloc[0] if "city" in df.columns else "unknown"
    X_train, X_test, y_train, y_test = time_based_split(
        df, cutoff_date=cutoff_date, target=target
    )
    X_train_c, X_test_c, y_train_c, y_test_c = prepare_ml_data(
        X_train, y_train, X_test, y_test
    )

    models = {}
    results = {"city": city, "cutoff_date": cutoff_date, "models": {}}

    train_mask = df["date"] < pd.Timestamp(cutoff_date)
    test_mask = df["date"] >= pd.Timestamp(cutoff_date)
    train_full = df[train_mask].copy()
    test_full = df[test_mask].copy()

    if include_baselines:
        models["Moving Average"] = train_moving_average(train_full, test_full, target=target)
        models["Seasonal Naive"] = train_seasonal_naive(train_full, test_full, target=target)

    if include_ml:
        models["XGBoost"] = train_xgboost(
            X_train_c, y_train_c, X_test_c, y_test_c, params=xgboost_params
        )
        if len(X_train_c) > 500:
            models["Random Forest"] = train_random_forest(
                X_train_c, y_train_c, X_test_c, y_test_c
            )

    if include_prophet and len(train_full) > 365:
        prophet_result = train_prophet(train_full, test_full, target=target)
        if "error" not in prophet_result:
            models["Prophet"] = prophet_result

    results["models"] = models

    # Best model by RMSE
    valid = {k: v["metrics"].get("rmse", np.inf)
             for k, v in models.items() if "metrics" in v and v["metrics"]}
    best = min(valid, key=valid.get) if valid else None
    results["best_model"] = best
    results["summary"] = {
        k: v["metrics"] for k, v in models.items() if "metrics" in v
    }
    return results


def model_rankings_by_metric(
    results: dict,
    metric: str = "rmse",
) -> list[tuple[str, float]]:
    """Return sorted list of (model_name, metric_value) ascending."""
    rankings = []
    for name, model in results.get("models", {}).items():
        m = model.get("metrics", {})
        if metric in m:
            rankings.append((name, m[metric]))
    return sorted(rankings, key=lambda x: x[1])


def cross_city_evaluation(
    city_dfs: dict[str, pd.DataFrame],
    target: str = "aqi",
    cutoff_date: str = "2019-01-01",
) -> pd.DataFrame:
    """Run all models on multiple cities and return comparison table.

    Returns DataFrame with columns: city, model, mape, rmse, mae, n_test.
    """
    rows = []
    for city, df in city_dfs.items():
        if df.empty or len(df) < 200:
            continue
        try:
            results = evaluate_all_models(df, target=target, cutoff_date=cutoff_date)
            for model_name, model_data in results["models"].items():
                metrics = model_data.get("metrics", {})
                row = {
                    "city": city,
                    "model": model_name,
                    "mape": metrics.get("mape"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "n_test": len(model_data.get("predictions", [])),
                }
                rows.append(row)
        except Exception:
            pass
    return pd.DataFrame(rows)


def error_analysis(results: dict) -> dict:
    """Analyze prediction errors by season/band for best model.

    Returns dict with error breakdowns.
    """
    best = results.get("best_model")
    if not best:
        return {"error": "No best model found"}
    model_data = results["models"].get(best)
    if not model_data or "predictions" not in model_data:
        return {"error": f"No predictions for {best}"}

    actuals = np.array(model_data["actuals"])
    preds = np.array(model_data["predictions"])
    errors = actuals - preds
    abs_pct_error = np.abs(errors) / (actuals + 1) * 100

    aqi_bands = [
        ("Good (0-50)", actuals <= 50),
        ("Satisfactory (51-100)", (actuals > 50) & (actuals <= 100)),
        ("Moderate (101-200)", (actuals > 100) & (actuals <= 200)),
        ("Poor (201-300)", (actuals > 200) & (actuals <= 300)),
        ("Very Poor (300+)", actuals > 300),
    ]

    band_errors = {}
    for label, mask in aqi_bands:
        if mask.sum() > 0:
            band_errors[label] = {
                "count": int(mask.sum()),
                "mape": float(np.mean(abs_pct_error[mask])),
                "bias": float(np.mean(errors[mask])),
            }

    return {
        "best_model": best,
        "overall_mape": float(np.mean(abs_pct_error)),
        "overall_bias": float(np.mean(errors)),
        "by_aqi_band": band_errors,
        "worst_prediction_idx": int(np.argmax(abs_pct_error)),
        "best_prediction_idx": int(np.argmin(abs_pct_error)),
    }


def seasonal_error_analysis(
    df: pd.DataFrame,
    results: dict,
) -> dict:
    """Break down errors by season."""
    best = results.get("best_model")
    if not best:
        return {}
    model_data = results["models"].get(best)
    if not model_data:
        return {}

    date_col = "datetime" if "datetime" in df.columns else "date"
    test_mask = df["date"] >= pd.Timestamp(results.get("cutoff_date", "2019-01-01"))
    test_dates = df.loc[test_mask.values, date_col].values
    n_dates = min(len(test_dates), len(model_data.get("predictions", [])))
    test_dates = test_dates[:n_dates]
    preds = model_data["predictions"][:n_dates]
    actuals = model_data["actuals"][:n_dates]
    errors = actuals - preds
    abs_pct = np.abs(errors) / (actuals + 1) * 100

    seasons = {
        "Winter (Dec-Feb)": [12, 1, 2],
        "Spring (Mar-May)": [3, 4, 5],
        "Summer/Monsoon (Jun-Sep)": [6, 7, 8, 9],
        "Autumn (Oct-Nov)": [10, 11],
    }

    result = {}
    for season_name, months in seasons.items():
        dates_series = pd.to_datetime(test_dates)
        mask = np.isin(dates_series.month, months)
        if mask.sum() > 0:
            result[season_name] = {
                "count": int(mask.sum()),
                "mape": float(np.mean(abs_pct[mask])),
                "bias": float(np.mean(errors[mask])),
            }

    return result
