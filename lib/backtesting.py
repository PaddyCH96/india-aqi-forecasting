"""Honest multi-step forecast evaluation.

The dashboard previously reported one-step-ahead error from a model whose
features include ``aqi_lag1``. Daily AQI is strongly autocorrelated, so that
number mostly measures how close yesterday is to today -- it flatters the model
and says little about forecasting several days ahead.

This module measures what a forecast actually has to do:

* ``recursive_forecast`` predicts one day, feeds that prediction back in as the
  next day's lag, and repeats. Errors compound, which is the point.
* ``persistence_forecast`` is the baseline any autocorrelated series must beat:
  every future day equals the last observed value.
* ``rolling_origin_backtest`` refits at successive cut-offs and reports error
  per horizon, so 1-day and 14-day accuracy are never conflated.
* ``residual_intervals`` derives prediction bands from realised backtest errors
  instead of assuming a fixed percentage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib.feature_engineering import build_feature_pipeline
from lib.logging import setup_logger
from lib.metrics import calc_mae, calc_mape, calc_rmse

logger = setup_logger(__name__)

DEFAULT_HORIZONS = (1, 3, 7, 14)


def persistence_forecast(history: pd.DataFrame, horizon: int,
                         target: str = "aqi") -> pd.DataFrame:
    """Predict every future day as the last observed value.

    The standard baseline for an autocorrelated series. A model that cannot
    beat this has not learned anything beyond "tomorrow resembles today".
    """
    hist = history.dropna(subset=[target]).sort_values("date")
    if hist.empty:
        return pd.DataFrame(columns=["date", "prediction", "horizon"])

    last_value = float(hist[target].iloc[-1])
    last_date = pd.to_datetime(hist["date"].iloc[-1])
    return pd.DataFrame([
        {
            "date": last_date + pd.Timedelta(days=h),
            "prediction": last_value,
            "horizon": h,
        }
        for h in range(1, horizon + 1)
    ])


def seasonal_naive_forecast(history: pd.DataFrame, horizon: int,
                            season_length: int = 365,
                            target: str = "aqi") -> pd.DataFrame:
    """Predict each future day as the value one season earlier."""
    hist = history.dropna(subset=[target]).sort_values("date").reset_index(drop=True)
    if hist.empty:
        return pd.DataFrame(columns=["date", "prediction", "horizon"])

    values = hist[target].to_numpy()
    last_date = pd.to_datetime(hist["date"].iloc[-1])
    rows = []
    for h in range(1, horizon + 1):
        idx = len(values) - season_length + h - 1
        # Fall back to the last observation when history is shorter than a season.
        pred = float(values[idx]) if 0 <= idx < len(values) else float(values[-1])
        rows.append({"date": last_date + pd.Timedelta(days=h),
                     "prediction": pred, "horizon": h})
    return pd.DataFrame(rows)


def recursive_forecast(
    model,
    feature_names: list[str],
    history: pd.DataFrame,
    city: str,
    horizon: int,
    target: str = "aqi",
) -> pd.DataFrame:
    """Forecast ``horizon`` days ahead by feeding predictions back as inputs.

    At each step the predicted target is appended to the history, features are
    rebuilt so lag and rolling terms see that prediction, and the next day is
    predicted from the updated frame.

    Exogenous pollutants (pm2_5, no2, co, ...) are unknown for future dates and
    are carried forward from the last observation. That assumption is why error
    grows with horizon, and it is stated rather than hidden: forecasting each
    pollutant separately would be a larger project than this dashboard needs.
    """
    hist = history.sort_values("date").copy()
    hist["date"] = pd.to_datetime(hist["date"])
    if hist.empty:
        return pd.DataFrame(columns=["date", "prediction", "horizon"])

    exog = [c for c in hist.columns if c not in ("city", "date", target)]
    last_known = hist.iloc[-1]
    rows = []

    for h in range(1, horizon + 1):
        feat = build_feature_pipeline(hist, target=target)
        feat = feat.sort_values("date")
        latest = feat.iloc[-1]

        vector = {}
        for name in feature_names:
            value = latest.get(name, np.nan) if name in feat.columns else np.nan
            vector[name] = 0.0 if pd.isna(value) else float(value)

        pred = float(model.predict(pd.DataFrame([vector]))[0])
        pred = max(pred, 0.0)

        next_date = pd.to_datetime(hist["date"].iloc[-1]) + pd.Timedelta(days=1)
        rows.append({"date": next_date, "prediction": round(pred, 1), "horizon": h})

        new_row = {"city": city, "date": next_date, target: pred}
        for col in exog:
            new_row[col] = last_known.get(col, np.nan)
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(rows)


def _score(actual: pd.Series, predicted: pd.Series) -> dict:
    valid = actual.notna() & predicted.notna()
    if valid.sum() == 0:
        return {"mape": np.nan, "rmse": np.nan, "mae": np.nan, "n": 0}
    a, p = actual[valid], predicted[valid]
    return {
        "mape": calc_mape(a, p),
        "rmse": calc_rmse(a, p),
        "mae": calc_mae(a, p),
        "n": int(valid.sum()),
    }


def rolling_origin_backtest(
    df: pd.DataFrame,
    city: str,
    train_fn,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    n_origins: int = 8,
    min_train_days: int = 365,
    target: str = "aqi",
) -> dict:
    """Refit at successive cut-offs and score each horizon separately.

    At every origin the model sees only data up to that date, forecasts forward
    recursively, and is compared against the actuals that follow -- alongside
    persistence over the same window, so the comparison is like for like.

    Returns a dict with per-horizon metrics for the model and the baseline, and
    the raw per-origin errors used to build prediction intervals.
    """
    data = df[df["city"] == city].dropna(subset=[target]).sort_values("date")
    data = data.reset_index(drop=True)
    max_h = max(horizons)

    if len(data) < min_train_days + max_h + n_origins:
        logger.warning("%s: %d usable days, too few to backtest", city, len(data))
        return {"city": city, "n_origins": 0, "horizons": {}, "errors": []}

    last_origin = len(data) - max_h
    first_origin = max(min_train_days, last_origin - (n_origins - 1) * max_h)
    origins = np.linspace(first_origin, last_origin, num=n_origins, dtype=int)
    origins = sorted(set(int(o) for o in origins))

    records = []
    for origin in origins:
        train = data.iloc[:origin]
        future = data.iloc[origin:origin + max_h]
        if future.empty:
            continue

        try:
            model, feature_names = train_fn(train, target)
        except Exception as exc:  # noqa: BLE001 - a bad origin must not kill the run
            logger.warning("%s: training failed at origin %d: %s", city, origin, exc)
            continue

        model_fc = recursive_forecast(model, feature_names, train, city, max_h, target)
        base_fc = persistence_forecast(train, max_h, target)

        actual = future[["date", target]].rename(columns={target: "actual"})
        actual["date"] = pd.to_datetime(actual["date"])
        actual["horizon"] = range(1, len(actual) + 1)

        merged = actual.merge(
            model_fc[["horizon", "prediction"]], on="horizon", how="left"
        ).merge(
            base_fc[["horizon", "prediction"]], on="horizon", how="left",
            suffixes=("_model", "_persistence"),
        )
        merged["origin"] = str(pd.to_datetime(train["date"].iloc[-1]).date())
        records.append(merged)

    if not records:
        return {"city": city, "n_origins": 0, "horizons": {}, "errors": []}

    allr = pd.concat(records, ignore_index=True)
    per_horizon = {}
    for h in horizons:
        sub = allr[allr["horizon"] <= h]
        per_horizon[h] = {
            "model": _score(sub["actual"], sub["prediction_model"]),
            "persistence": _score(sub["actual"], sub["prediction_persistence"]),
        }

    return {
        "city": city,
        "n_origins": len(records),
        "horizons": per_horizon,
        "errors": allr,
    }


def residual_intervals(errors: pd.DataFrame, level: float = 0.80) -> dict:
    """Empirical prediction bands, as residual quantiles per horizon.

    Replaces a hardcoded +/-15%: the band at each horizon is as wide as that
    horizon's realised backtest errors say it should be.
    """
    if errors is None or len(errors) == 0:
        return {}
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    resid = errors.dropna(subset=["actual", "prediction_model"]).copy()
    resid["residual"] = resid["prediction_model"] - resid["actual"]
    out = {}
    for h, grp in resid.groupby("horizon"):
        if len(grp) >= 3:
            out[int(h)] = {
                "lower": float(grp["residual"].quantile(lo_q)),
                "upper": float(grp["residual"].quantile(hi_q)),
                "n": int(len(grp)),
            }
    return out


def skill_score(model_mape: float, baseline_mape: float) -> float | None:
    """Percent improvement over the baseline. Negative means worse than it."""
    if not baseline_mape or np.isnan(model_mape) or np.isnan(baseline_mape):
        return None
    return round((baseline_mape - model_mape) / baseline_mape * 100, 1)


# ---------------------------------------------------------------------------
# Direct multi-horizon forecasting
# ---------------------------------------------------------------------------
#
# Recursive forecasting has two weaknesses here: errors compound, and exogenous
# pollutants have to be frozen at their last observed value because nobody knows
# next week's PM2.5. Direct forecasting sidesteps both. One model per horizon is
# trained to map "features known at time t" straight to "AQI at t + h", so
# nothing is fed back and no unknown future input is required.


# Calendar features are deterministic: the month and weekday of a date two weeks
# out are known today. For direct forecasting they should describe the date being
# predicted, not the date the forecast is made from. Shifting them is not leakage.
KNOWN_AHEAD = (
    "hour", "day_of_week", "month", "quarter", "is_weekend",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
)


def _calendar_features(when: pd.Timestamp) -> dict:
    """Deterministic calendar values for a future date, matching KNOWN_AHEAD."""
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


def make_direct_dataset(features: pd.DataFrame, target: str, horizon: int,
                        feature_names: list[str], delta: bool = False):
    """Align features at t with the target at t + horizon.

    Observed predictors stay at t, because that is all a forecaster knows.
    Calendar features are advanced to t + horizon, because the calendar is not
    something you have to predict.
    """
    frame = features.sort_values("date").reset_index(drop=True).copy()
    for col in KNOWN_AHEAD:
        if col in frame.columns:
            frame[col] = frame[col].shift(-horizon)
    future = frame[target].shift(-horizon)
    # Delta targeting: learn how far the AQI moves from today rather than its
    # absolute level. The forecast is then today's value plus the predicted
    # change, so the model starts from persistence and only has to improve on
    # it -- rather than re-deriving the level and adding variance doing so.
    y = (future - frame[target]) if delta else future
    mask = y.notna() & frame[target].notna() & frame[list(
        c for c in KNOWN_AHEAD if c in frame.columns
    )].notna().all(axis=1)
    return frame.loc[mask, feature_names], y[mask]


def train_direct_models(history: pd.DataFrame, train_one, target: str,
                        horizons, feature_builder) -> dict:
    """Fit one model per horizon. Returns {horizon: (model, feature_names)}."""
    feat = feature_builder(history, target)
    models = {}
    for h in horizons:
        try:
            models[h] = train_one(feat, target, h)
        except Exception as exc:  # noqa: BLE001
            logger.warning("direct model failed at h=%d: %s", h, exc)
    return models


def direct_forecast(models: dict, history: pd.DataFrame, target: str,
                    feature_builder) -> pd.DataFrame:
    """Predict each horizon from the most recent observed feature row."""
    feat = feature_builder(history, target).sort_values("date")
    if feat.empty:
        return pd.DataFrame(columns=["date", "prediction", "horizon"])

    latest = feat.iloc[-1]
    last_date = pd.to_datetime(feat["date"].iloc[-1])
    rows = []
    for h, (model, names) in sorted(models.items()):
        target_date = last_date + pd.Timedelta(days=h)
        calendar = _calendar_features(target_date)
        vector = {}
        for name in names:
            if name in calendar:            # describe the date being predicted
                vector[name] = calendar[name]
                continue
            value = latest.get(name, np.nan) if name in feat.columns else np.nan
            vector[name] = 0.0 if pd.isna(value) else float(value)
        pred = max(float(model.predict(pd.DataFrame([vector]))[0]), 0.0)
        rows.append({"date": last_date + pd.Timedelta(days=h),
                     "prediction": round(pred, 1), "horizon": h})
    return pd.DataFrame(rows)


def rolling_origin_backtest_direct(
    df: pd.DataFrame,
    city: str,
    train_one,
    feature_builder,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    n_origins: int = 8,
    min_train_days: int = 365,
    target: str = "aqi",
) -> dict:
    """Rolling-origin backtest of the direct-per-horizon approach."""
    data = df[df["city"] == city].dropna(subset=[target]).sort_values("date")
    data = data.reset_index(drop=True)
    max_h = max(horizons)

    if len(data) < min_train_days + max_h + n_origins:
        logger.warning("%s: %d usable days, too few to backtest", city, len(data))
        return {"city": city, "n_origins": 0, "horizons": {}, "errors": []}

    last_origin = len(data) - max_h
    first_origin = max(min_train_days, last_origin - (n_origins - 1) * max_h)
    origins = sorted({int(o) for o in
                      np.linspace(first_origin, last_origin, num=n_origins, dtype=int)})

    records = []
    for origin in origins:
        train = data.iloc[:origin]
        future = data.iloc[origin:origin + max_h]
        if future.empty:
            continue

        models = train_direct_models(train, train_one, target, horizons, feature_builder)
        if not models:
            continue

        model_fc = direct_forecast(models, train, target, feature_builder)
        base_fc = persistence_forecast(train, max_h, target)

        actual = future[["date", target]].rename(columns={target: "actual"})
        actual["date"] = pd.to_datetime(actual["date"])
        actual["horizon"] = range(1, len(actual) + 1)
        actual = actual[actual["horizon"].isin(horizons)]

        merged = actual.merge(
            model_fc[["horizon", "prediction"]], on="horizon", how="left"
        ).merge(
            base_fc[["horizon", "prediction"]], on="horizon", how="left",
            suffixes=("_model", "_persistence"),
        )
        merged["origin"] = str(pd.to_datetime(train["date"].iloc[-1]).date())
        records.append(merged)

    if not records:
        return {"city": city, "n_origins": 0, "horizons": {}, "errors": []}

    allr = pd.concat(records, ignore_index=True)
    per_horizon = {}
    for h in horizons:
        sub = allr[allr["horizon"] == h]
        per_horizon[h] = {
            "model": _score(sub["actual"], sub["prediction_model"]),
            "persistence": _score(sub["actual"], sub["prediction_persistence"]),
        }

    return {"city": city, "n_origins": len(records),
            "horizons": per_horizon, "errors": allr}


# ---------------------------------------------------------------------------
# Block backtest -- the evaluator the reported numbers come from
# ---------------------------------------------------------------------------
#
# Scoring a handful of origins gives one point per horizon per origin, which is
# far too few to separate signal from noise: repeated runs swung by tens of
# percent. This evaluator refits a few times and scores a *block* of consecutive
# test days at each refit, so every horizon is measured over hundreds of points.


def _day_of_year_climatology(history: pd.DataFrame, target: str,
                             window: int = 7) -> dict:
    """Mean target per day-of-year, smoothed over a +/- ``window`` day band."""
    hist = history.dropna(subset=[target]).copy()
    if hist.empty:
        return {}
    hist["doy"] = pd.to_datetime(hist["date"]).dt.dayofyear
    daily = hist.groupby("doy")[target].mean()
    full = daily.reindex(range(1, 367))
    # Wrap the year so late December is smoothed against early January.
    tripled = pd.concat([full, full, full], ignore_index=True)
    smoothed = tripled.rolling(2 * window + 1, min_periods=1, center=True).mean()
    middle = smoothed.iloc[366:732].reset_index(drop=True)
    overall = float(hist[target].mean())
    return {doy: (float(v) if pd.notna(v) else overall)
            for doy, v in zip(range(1, 367), middle)}


def block_backtest(
    df: pd.DataFrame,
    city: str,
    train_one,
    feature_builder,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    n_folds: int = 4,
    test_block: int = 90,
    min_train_days: int = 540,
    target: str = "aqi",
    delta: bool = False,
) -> dict:
    """Expanding-window backtest scored over blocks of consecutive days.

    For each fold the model trains on everything before the block, then for every
    day t inside the block predicts the AQI at t + h from features known at t.
    Persistence predicts the value observed at t. Both are scored on identical
    (t, h) pairs, so the comparison is like for like.
    """
    data = df[df["city"] == city].dropna(subset=[target]).sort_values("date")
    data = data.reset_index(drop=True)
    max_h = max(horizons)
    need = min_train_days + n_folds * test_block + max_h

    if len(data) < need:
        n_folds = max(1, (len(data) - min_train_days - max_h) // test_block)
        if n_folds < 1:
            logger.warning("%s: %d days, too few to backtest", city, len(data))
            return {"city": city, "n_folds": 0, "horizons": {}, "errors": None}

    feat_all = feature_builder(data, target)
    feat_all = feat_all.sort_values("date").reset_index(drop=True)
    names = [c for c in feat_all.columns
             if c not in (target, "date", "city", "aqi_bucket",
                          "is_synthetic", "data_source")
             and feat_all[c].dtype.kind in "fi"]

    fold_end = len(data) - max_h
    starts = [fold_end - (i + 1) * test_block for i in range(n_folds)][::-1]
    starts = [s for s in starts if s >= min_train_days]
    if not starts:
        return {"city": city, "n_folds": 0, "horizons": {}, "errors": None}

    rows = []
    for start in starts:
        train_slice = data.iloc[:start]
        # Climatology: the average AQI for that day of the year in the training
        # window, smoothed across a +/-7 day window so single years do not
        # dominate. Known arbitrarily far ahead, so it is the natural baseline
        # once persistence goes stale.
        climatology = _day_of_year_climatology(train_slice, target)
        for h in horizons:
            try:
                model, used = train_one(feature_builder(train_slice, target), target, h)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s h=%d: train failed: %s", city, h, exc)
                continue

            block = range(start, min(start + test_block, fold_end))
            X = feat_all.loc[list(block), [c for c in used if c in feat_all.columns]]
            X = X.reindex(columns=used).fillna(0.0)
            if X.empty:
                continue

            preds = model.predict(X)
            for offset, idx in enumerate(block):
                if idx + h >= len(data):
                    continue
                base = float(data[target].iloc[idx])
                point = (base + preds[offset]) if delta else preds[offset]
                target_doy = pd.to_datetime(data["date"].iloc[idx + h]).dayofyear
                rows.append({
                    "horizon": h,
                    "date": data["date"].iloc[idx + h],
                    "actual": float(data[target].iloc[idx + h]),
                    "prediction_model": float(max(point, 0.0)),
                    "prediction_persistence": float(data[target].iloc[idx]),
                    "prediction_climatology": float(
                        climatology.get(target_doy, train_slice[target].mean())
                    ),
                    "fold": str(data["date"].iloc[start].date()),
                })

    if not rows:
        return {"city": city, "n_folds": 0, "horizons": {}, "errors": None}

    allr = pd.DataFrame(rows)
    per_horizon = {}
    for h in horizons:
        sub = allr[allr["horizon"] == h]
        per_horizon[h] = {
            "model": _score(sub["actual"], sub["prediction_model"]),
            "persistence": _score(sub["actual"], sub["prediction_persistence"]),
            "climatology": _score(sub["actual"], sub["prediction_climatology"]),
        }
    return {"city": city, "n_folds": len(starts),
            "horizons": per_horizon, "errors": allr}
