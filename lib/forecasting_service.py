"""Production forecasting service.

Wraps trained models into a simple prediction interface for the dashboard.
"""

import pandas as pd
import pickle
import os
from datetime import datetime, timedelta

from lib.db import get_engine, load_city_pollutants
from lib.feature_engineering import build_feature_pipeline
from lib.ml_pipeline import prepare_ml_data
from lib.model_training import train_xgboost
from lib.logging import setup_logger

logger = setup_logger(__name__)
from lib.ml_pipeline import time_based_split


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def get_model_path(city: str, model_type: str = "xgboost") -> str:
    return os.path.join(MODEL_DIR, f"{city}_{model_type}.pkl")


def train_and_save_model(
    city: str,
    target: str = "aqi",
    use_synthetic: bool = False,
    engine=None,
) -> dict:
    """Train XGBoost for a city, save to disk, return evaluation results."""
    if engine is None:
        engine = get_engine()
    df = load_city_pollutants(engine, city, use_synthetic=use_synthetic)
    if df.empty or len(df) < 200:
        return {"error": f"Insufficient data for {city}: {len(df)} rows"}
    df["city"] = city
    df = build_feature_pipeline(df, target=target)

    X_train, X_test, y_train, y_test = time_based_split(df, target=target)
    X_train_c, X_test_c, y_train_c, y_test_c = prepare_ml_data(
        X_train, y_train, X_test, y_test
    )

    if len(X_train_c) < 100:
        return {"error": f"Insufficient training data after NaN removal: {len(X_train_c)} rows"}

    result = train_xgboost(X_train_c, y_train_c, X_test_c, y_test_c)

    ensure_model_dir()
    with open(get_model_path(city), "wb") as f:
        pickle.dump({
            "model": result["model"],
            "feature_names": list(X_train_c.columns),
            "target": target,
            "city": city,
            "trained_at": datetime.now().isoformat(),
        }, f)

    result["n_train"] = len(X_train_c)
    result["n_test"] = len(X_test_c)
    return result


def load_model(city: str, model_type: str = "xgboost") -> dict | None:
    """Load a saved model from disk.

    Committed pickles can fail to load under a different xgboost or scikit-learn
    build than the one that wrote them. Treat that as "no model available" so the
    caller offers retraining instead of raising.
    """
    path = get_model_path(city, model_type)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:  # noqa: BLE001 - unpickling can raise almost anything
        logger.warning("Could not load %s model for %s: %s", model_type, city, exc)
        return None


def predict_future(
    city: str,
    days: int = 30,
    target: str = "aqi",
    use_synthetic: bool = False,
    engine=None,
) -> pd.DataFrame | None:
    """Generate future predictions using the latest trained model.

    Uses the last known feature values as a simplified forecasting method.
    For production, this should use iterative multi-step or direct forecasting.

    Returns DataFrame with columns: date, prediction, lower_bound, upper_bound.
    """
    model_data = load_model(city)
    if model_data is None:
        return None

    if engine is None:
        engine = get_engine()
    df = load_city_pollutants(engine, city, use_synthetic=use_synthetic)
    if df.empty:
        return None
    df["city"] = city
    df_feat = build_feature_pipeline(df, target=target)
    df_feat = df_feat.sort_values("date").reset_index(drop=True)

    model = model_data["model"]
    feature_names = model_data["feature_names"]

    futures = []
    last_idx = len(df_feat) - 1
    last_date = df_feat["date"].iloc[last_idx]
    last_features = df_feat.iloc[last_idx]

    for i in range(1, days + 1):
        next_date = last_date + timedelta(days=i)
        row = {"date": next_date, "prediction": None,
               "lower_bound": None, "upper_bound": None}

        feat_vector = {}
        for fn in feature_names:
            if fn in df_feat.columns:
                val = last_features[fn]
                if pd.notna(val):
                    feat_vector[fn] = val
                else:
                    feat_vector[fn] = 0
            else:
                feat_vector[fn] = 0

        inp = pd.DataFrame([feat_vector])
        pred = model.predict(inp)[0]

        row["prediction"] = round(pred, 1)
        row["lower_bound"] = round(max(0, pred * 0.85), 1)
        row["upper_bound"] = round(pred * 1.15, 1)
        futures.append(row)

    return pd.DataFrame(futures)


def get_forecast_for_dashboard(
    city: str,
    horizon_hours: int = 72,
    use_synthetic: bool = False,
    engine=None,
) -> dict:
    """Generate forecast data for dashboard rendering.

    Returns dict with model status and forecast DataFrame or error.
    """
    if engine is None:
        engine = get_engine()
    days = max(1, horizon_hours // 24 + 1)

    if not os.path.exists(get_model_path(city)):
        result = train_and_save_model(city, use_synthetic=use_synthetic, engine=engine)
        if "error" in result:
            return {"status": "error", "message": result["error"]}

    forecast = predict_future(city, days=days, use_synthetic=use_synthetic, engine=engine)
    if forecast is None:
        return {"status": "error", "message": "Could not generate forecast"}

    return {
        "status": "ok",
        "model": "XGBoost",
        "forecast": forecast.to_dict(orient="records"),
        "n_days": days,
        "generated_at": datetime.now().isoformat(),
    }


def list_trained_models() -> list[dict]:
    """List all saved models with metadata."""
    ensure_model_dir()
    models = []
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".pkl"):
            parts = fname.replace(".pkl", "").split("_")
            city = parts[0]
            model_type = parts[1] if len(parts) > 1 else "xgboost"
            path = os.path.join(MODEL_DIR, fname)
            mtime = os.path.getmtime(path)
            models.append({
                "city": city,
                "model_type": model_type,
                "size_kb": round(os.path.getsize(path) / 1024, 1),
                "trained_at": datetime.fromtimestamp(mtime).isoformat(),
            })
    return models


# ---------------------------------------------------------------------------
# Honest forecasting: direct per-horizon models, with the baseline alongside
# ---------------------------------------------------------------------------


def forecast_with_baseline(
    city: str,
    days: int = 7,
    target: str = "aqi",
    use_synthetic: bool = False,
    engine=None,
) -> dict | None:
    """Forecast ``days`` ahead and return the persistence baseline with it.

    Replaces predict_future, which held every feature at its last observed value
    and so returned the same number for every day -- a flat line presented as a
    forecast. Here a separate model is fit per horizon, mapping features known
    today to the AQI that many days out, so the curve actually varies.

    The persistence baseline is returned alongside because, on this data, it
    wins beyond about three days. Showing both is the honest presentation.
    """
    from lib.backtesting import make_direct_dataset, persistence_forecast
    from lib.feature_engineering import build_feature_pipeline
    from lib.ml_pipeline import get_feature_names, prepare_ml_data
    from lib.model_training import train_xgboost

    if engine is None:
        engine = get_engine()
    history = load_city_pollutants(engine, city, use_synthetic=use_synthetic)
    if history.empty or history[target].notna().sum() < 400:
        return None
    history = history.copy()
    history["city"] = city

    feat = build_feature_pipeline(history, target=target)
    names = get_feature_names(feat, target=target)
    last_date = pd.to_datetime(feat["date"].iloc[-1])

    rows = []
    for h in range(1, days + 1):
        try:
            X, y = make_direct_dataset(feat, target, h, names)
            X_train, _, y_train, _ = prepare_ml_data(X, y, X.tail(1), y.tail(1))
            trained = train_xgboost(X_train, y_train, X_train.tail(1), y_train.tail(1))
            model, used = trained["model"], list(X_train.columns)

            latest = feat.iloc[-1]
            target_date = last_date + timedelta(days=h)
            calendar = _calendar_for(target_date)
            vector = {}
            for name in used:
                if name in calendar:
                    vector[name] = calendar[name]
                    continue
                value = latest.get(name, float("nan")) if name in feat.columns else float("nan")
                vector[name] = 0.0 if pd.isna(value) else float(value)
            pred = max(float(model.predict(pd.DataFrame([vector]))[0]), 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s h=%d forecast failed: %s", city, h, exc)
            continue
        rows.append({"date": last_date + timedelta(days=h),
                     "horizon": h, "prediction": round(pred, 1)})

    if not rows:
        return None

    forecast = pd.DataFrame(rows)
    baseline = persistence_forecast(history, days, target)
    forecast = forecast.merge(
        baseline[["horizon", "prediction"]].rename(
            columns={"prediction": "persistence"}),
        on="horizon", how="left",
    )
    return {
        "city": city,
        "forecast": forecast,
        "last_observed": float(history[target].dropna().iloc[-1]),
        "last_date": last_date,
    }


def _calendar_for(when) -> dict:
    """Calendar values for a future date -- known in advance, not predicted."""
    import numpy as np
    when = pd.Timestamp(when)
    return {
        "hour": float(when.hour),
        "day_of_week": float(when.dayofweek),
        "month": float(when.month),
        "quarter": float(when.quarter),
        "is_weekend": float(when.dayofweek >= 5),
        "month_sin": float(np.sin(2 * np.pi * when.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * when.month / 12)),
        "dow_sin": float(np.sin(2 * np.pi * when.dayofweek / 7)),
        "dow_cos": float(np.cos(2 * np.pi * when.dayofweek / 7)),
    }


def load_backtest_results() -> dict | None:
    """Read data/backtest_results.json, the source of the accuracy figures."""
    import json
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "backtest_results.json",
    )
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read backtest results: %s", exc)
        return None


def load_precomputed_forecast(city: str, days: int) -> dict | None:
    """Read a precomputed forecast for the bundled demo database.

    Fitting one model per horizon takes tens of seconds on a small hosted
    instance, long enough that the forecasting section never finishes rendering.
    The demo dataset never changes, so the result is deterministic and is
    generated ahead of time by scripts/precompute_forecasts.py.
    """
    import json
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "forecasts.json",
    )
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            payload = json.load(fh)
        entry = payload.get("cities", {}).get(city)
        if not entry:
            return None
        frame = pd.DataFrame(entry["rows"])
        if frame.empty:
            return None
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[frame["horizon"] <= days].reset_index(drop=True)
        if frame.empty:
            return None
        return {
            "city": city,
            "forecast": frame,
            "last_observed": entry["last_observed"],
            "last_date": pd.to_datetime(entry["last_date"]),
            "precomputed": True,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read precomputed forecast: %s", exc)
        return None
