#!/usr/bin/env python3
"""Rolling-origin backtest for every dashboard city.

Produces the numbers the dashboard and case study quote. Run after changing
features or model settings:

    python scripts/backtest.py                 # all cities, default horizons
    python scripts/backtest.py --cities Delhi  # one city
    python scripts/backtest.py --origins 10    # more cut-offs, slower

Writes data/backtest_results.json.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.pathing import ensure_project_root_on_path

ensure_project_root_on_path()

warnings.filterwarnings("ignore")

from lib.backtesting import (  # noqa: E402
    DEFAULT_HORIZONS,
    block_backtest,
    make_direct_dataset,
    residual_intervals,
    skill_score,
)
from lib.db import get_engine, load_city_pollutants  # noqa: E402
from lib.feature_engineering import build_feature_pipeline  # noqa: E402
from lib.logging import setup_logger  # noqa: E402
from lib.ml_pipeline import get_feature_names, prepare_ml_data  # noqa: E402
from lib.model_training import train_xgboost  # noqa: E402

logger = setup_logger("backtest")

OUTPUT = os.path.join(REPO_ROOT, "data", "backtest_results.json")
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad", "Kolkata"]


def feature_builder(history, target):
    return build_feature_pipeline(history, target=target)


def train_one(features, target, horizon):
    """Fit one direct model mapping features at t to the target at t + horizon."""
    names = get_feature_names(features, target=target)
    X, y = make_direct_dataset(features, target, horizon, names)
    X_train, _, y_train, _ = prepare_ml_data(X, y, X.tail(1), y.tail(1))
    result = train_xgboost(X_train, y_train, X_train.tail(1), y_train.tail(1))
    return result["model"], list(X_train.columns)


def run(cities, horizons, n_origins):
    engine = get_engine()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": list(horizons),
        "n_origins_requested": n_origins,
        "method": (
            "Expanding-window block backtest. At each fold a separate XGBoost "
            "model per horizon is fit on data before the block, then for every "
            "day t in a 90-day held-out block it predicts AQI at t+h from "
            "features known at t. Scored against persistence (value at t) and "
            "climatology (smoothed day-of-year mean) on identical (t, h) pairs."
        ),
        "cities": {},
    }

    for city in cities:
        df = load_city_pollutants(engine, city)
        if df.empty:
            logger.warning("%s: no data, skipped", city)
            continue
        df["city"] = city
        logger.info("%s: backtesting over %d days...", city, len(df))

        result = block_backtest(
            df, city, train_one, feature_builder,
            horizons=tuple(horizons), n_folds=n_origins,
        )
        if result["n_folds"] == 0:
            logger.warning("%s: not enough history to backtest honestly", city)
            payload["cities"][city] = {"n_folds": 0, "horizons": {},
                                       "note": "insufficient history"}
            continue

        entry = {"n_folds": result["n_folds"], "horizons": {}}
        for h, scores in result["horizons"].items():
            m, p, c = scores["model"], scores["persistence"], scores["climatology"]
            skill = skill_score(m["mape"], p["mape"])
            entry["horizons"][str(h)] = {
                "model_mape": round(m["mape"], 2) if m["n"] else None,
                "model_rmse": round(m["rmse"], 2) if m["n"] else None,
                "persistence_mape": round(p["mape"], 2) if p["n"] else None,
                "climatology_mape": round(c["mape"], 2) if c["n"] else None,
                "skill_vs_persistence_pct": skill,
                "beats_persistence": bool(skill is not None and skill > 0),
                "n_points": m["n"],
            }
            logger.info(
                "  h=%2dd  n=%4d  model %5.1f%%  persist %5.1f%%  clim %5.1f%%  skill %+.1f%%",
                h, m["n"], m["mape"], p["mape"], c["mape"], skill or 0.0,
            )

        entry["intervals"] = residual_intervals(result["errors"])
        payload["cities"][city] = entry

    engine.dispose()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", OUTPUT)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cities", nargs="+", default=CITIES)
    ap.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    ap.add_argument("--origins", type=int, default=8)
    args = ap.parse_args()
    run(args.cities, args.horizons, args.origins)


if __name__ == "__main__":
    main()
