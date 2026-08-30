#!/usr/bin/env python3
"""Precompute forecasts for the bundled demo database.

The hosted dashboard cannot fit a model per horizon on request: on Streamlit
Community Cloud that takes long enough that the forecasting section never
renders. The demo dataset is fixed, so the forecasts are deterministic and can
be computed once here and read instantly at runtime.

    python scripts/precompute_forecasts.py

Writes data/forecasts.json. Re-run whenever the demo data or the model changes.
"""

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

from lib.db import get_engine  # noqa: E402
from lib.forecasting_service import forecast_with_baseline  # noqa: E402
from lib.logging import setup_logger  # noqa: E402

logger = setup_logger("precompute-forecasts")
OUTPUT = os.path.join(REPO_ROOT, "data", "forecasts.json")
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad", "Kolkata"]
MAX_DAYS = 14


def main():
    engine = get_engine()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_days": MAX_DAYS,
        "note": (
            "Precomputed against the bundled demo database so the hosted app "
            "does not fit models on request. Regenerate with "
            "scripts/precompute_forecasts.py after changing the data or model."
        ),
        "cities": {},
    }

    for city in CITIES:
        logger.info("Forecasting %s...", city)
        result = forecast_with_baseline(city, days=MAX_DAYS, engine=engine)
        if result is None:
            logger.warning("  %s: insufficient history, skipped", city)
            continue
        fdf = result["forecast"].copy()
        fdf["date"] = fdf["date"].astype(str)
        payload["cities"][city] = {
            "last_observed": result["last_observed"],
            "last_date": str(result["last_date"].date()),
            "rows": fdf.to_dict(orient="records"),
        }
        logger.info("  %s: %d days", city, len(fdf))

    engine.dispose()
    with open(OUTPUT, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
