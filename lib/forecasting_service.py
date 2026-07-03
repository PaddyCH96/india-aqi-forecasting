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
    """Load a saved model from disk."""
    path = get_model_path(city, model_type)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


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
