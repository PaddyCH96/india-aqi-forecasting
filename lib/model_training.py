"""Model training pipelines for AQI forecasting.

Provides consistent training interfaces for:
1. Moving Average baseline
2. Seasonal Naive baseline
3. XGBoost regression
4. Prophet (uses existing lib/models.py)
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from lib.models import train_and_forecast as prophet_train
from lib.metrics import calc_mape, calc_rmse, calc_mae


def train_moving_average(
    train: pd.DataFrame,
    test: pd.DataFrame,
    window: int = 7,
    target: str = "aqi",
) -> dict:
    """Moving average baseline: predict next value as mean of last N days.

    Returns dict with 'predictions', 'model_type', 'params', 'metrics'.
    """
    date_col = "datetime" if "datetime" in train.columns else "date"
    full = pd.concat([train, test]).sort_values(date_col).reset_index(drop=True)
    values = full[target].values
    predictions = np.full(len(values), np.nan)

    for i in range(len(train), len(values)):
        if i >= window:
            predictions[i] = np.mean(values[i - window:i])

    train_preds = predictions[len(train):]
    train_actuals = values[len(train):]
    valid = ~(np.isnan(train_preds) | np.isnan(train_actuals))

    return {
        "model_type": "Moving Average",
        "params": {"window": window},
        "predictions": train_preds,
        "actuals": train_actuals,
        "metrics": {
            "mape": calc_mape(pd.Series(train_actuals[valid]), pd.Series(train_preds[valid])),
            "rmse": calc_rmse(pd.Series(train_actuals[valid]), pd.Series(train_preds[valid])),
            "mae": calc_mae(pd.Series(train_actuals[valid]), pd.Series(train_preds[valid])),
        },
    }


def train_seasonal_naive(
    train: pd.DataFrame,
    test: pd.DataFrame,
    period: int = 365,
    target: str = "aqi",
) -> dict:
    """Seasonal naive baseline: predict as value from N periods ago.

    For daily data with period=365: today's prediction = same day last year.
    """
    date_col = "datetime" if "datetime" in train.columns else "date"
    full = pd.concat([train, test]).sort_values(date_col).reset_index(drop=True)
    values = full[target].values
    predictions = np.full(len(values), np.nan)

    for i in range(len(train), len(values)):
        if i >= period:
            predictions[i] = values[i - period]

    train_preds = predictions[len(train):]
    train_actuals = values[len(train):]
    valid = ~(np.isnan(train_preds) | np.isnan(train_actuals))

    return {
        "model_type": "Seasonal Naive",
        "params": {"period": period},
        "predictions": train_preds,
        "actuals": train_actuals,
        "metrics": {
            "mape": calc_mape(pd.Series(train_actuals[valid]), pd.Series(train_preds[valid])),
            "rmse": calc_rmse(pd.Series(train_actuals[valid]), pd.Series(train_preds[valid])),
            "mae": calc_mae(pd.Series(train_actuals[valid]), pd.Series(train_preds[valid])),
        },
    }


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> dict:
    """Train XGBoost regressor for AQI prediction.

    Returns dict with model, predictions, metrics, and feature importance.
    """
    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    model = XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    valid = ~(np.isnan(preds) | np.isnan(y_test.values))
    metrics = {}
    if valid.sum() > 0:
        metrics = {
            "mape": calc_mape(
                pd.Series(y_test.values[valid]), pd.Series(preds[valid])
            ),
            "rmse": calc_rmse(
                pd.Series(y_test.values[valid]), pd.Series(preds[valid])
            ),
            "mae": calc_mae(
                pd.Series(y_test.values[valid]), pd.Series(preds[valid])
            ),
        }

    feature_importance = None
    if hasattr(model, "feature_importances_"):
        importance = sorted(
            zip(X_train.columns, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        feature_importance = importance[:20]

    return {
        "model_type": "XGBoost",
        "params": default_params,
        "model": model,
        "predictions": preds,
        "actuals": y_test.values,
        "metrics": metrics,
        "feature_importance": feature_importance,
    }


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
) -> dict:
    """Train Random Forest regressor (baseline comparison for XGBoost)."""
    default_params = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = RandomForestRegressor(**default_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    valid = ~(np.isnan(preds) | np.isnan(y_test.values))
    metrics = {}
    if valid.sum() > 0:
        metrics = {
            "mape": calc_mape(
                pd.Series(y_test.values[valid]), pd.Series(preds[valid])
            ),
            "rmse": calc_rmse(
                pd.Series(y_test.values[valid]), pd.Series(preds[valid])
            ),
            "mae": calc_mae(
                pd.Series(y_test.values[valid]), pd.Series(preds[valid])
            ),
        }

    return {
        "model_type": "Random Forest",
        "params": default_params,
        "model": model,
        "predictions": preds,
        "actuals": y_test.values,
        "metrics": metrics,
    }


def train_prophet(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "aqi",
) -> dict:
    """Train Prophet model. Uses existing lib/models infrastructure.

    Expects train/test with 'date' and target columns.
    Renames internally to Prophet's 'ds'/'y' convention.
    """
    train_df = train.rename(columns={"date": "ds", target: "y"})[["ds", "y"]].dropna()
    test_df = test.rename(columns={"date": "ds", target: "y"})[["ds", "y"]].dropna()

    if len(test_df) < 5:
        return {"model_type": "Prophet", "error": "Insufficient test data"}

    try:
        _, _, results = prophet_train(train_df, test_df)
        preds = results["yhat"].values
        actuals = results["y"].values

        metrics = {
            "mape": calc_mape(pd.Series(actuals), pd.Series(preds)),
            "rmse": calc_rmse(pd.Series(actuals), pd.Series(preds)),
            "mae": calc_mae(pd.Series(actuals), pd.Series(preds)),
        }

        return {
            "model_type": "Prophet",
            "params": {},
            "predictions": preds,
            "actuals": actuals,
            "metrics": metrics,
        }
    except Exception as e:
        return {"model_type": "Prophet", "error": str(e)}
