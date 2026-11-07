# ML Evaluation Report

**Source of truth:** [`data/backtest_results.json`](../data/backtest_results.json).
Every figure below is read from that file. No other accuracy number in this
repository supersedes it.

## Summary

Forecast error (MAPE, lower is better) comes from an expanding-window block
backtest. At each fold a separate XGBoost model per horizon is fit on data
strictly before a held-out 90-day block, then for every day *t* in that block it
predicts AQI at *t+h* using only features known at *t*. **Persistence** — simply
assuming the AQI at *t+h* equals the AQI at *t* — is scored on identical
`(t, h)` pairs, as is a day-of-year climatology baseline.

## Model Performance vs Persistence

Model MAPE % / persistence MAPE %. Bold marks the cells where the model beats
persistence.

| City | 1 day | 3 days | 7 days | 14 days |
|---|:--:|:--:|:--:|:--:|
| Bengaluru | 11 / 10 | 17 / 15 | 19 / 18 | 21 / 19 |
| Mumbai | 13 / 13 | 36 / 25 | 33 / 27 | 46 / 31 |
| Hyderabad | 14 / 13 | 32 / 24 | 35 / 30 | 41 / 32 |
| Delhi | **15 / 16** | **31 / 31** | 37 / 36 | 47 / 40 |
| Kolkata | 17 / 17 | 39 / 27 | 36 / 31 | **37 / 38** |
| Chennai | 23 / 18 | 41 / 27 | 48 / 31 | 49 / 33 |

Fold counts: Delhi, Bengaluru, Hyderabad and Chennai have 4 folds (360 scored
points per horizon); Mumbai and Kolkata have 2 folds (180 points per horizon).

## Key Findings

### The headline finding is a negative one, and it is the point of the project

Persistence is a very strong baseline for daily city-level AQI. Across six
cities and four horizons — 24 city/horizon combinations — gradient boosting
beats persistence in four of them, and only for Delhi at one and three days is
the win both positive and part of a consistent pattern. Kolkata's two wins (1
day and 14 days) are fractional, sit either side of two large losses at 3 and 7
days, and rest on the smallest fold count in the study; they should be read as
noise, not skill.

Everywhere else the model is worse than doing nothing, and often much worse:
Chennai loses to persistence by 29% to 56% of relative skill at every horizon,
Mumbai by up to 49%, Hyderabad by up to 31%.

Three attempts to improve on this — removing target leakage, training one model
per horizon rather than recursing a single model, and predicting the change
rather than the level — did not overturn it.

### The old city rankings were an artefact and are withdrawn

The previous version of this report ranked Bengaluru and Hyderabad as the
"easiest" cities and Kolkata as the "hardest", and concluded that data quality
rather than model choice determined accuracy. Under the corrected backtest that
conclusion does not survive:

- **Chennai** is the worst performer, not Kolkata, and it is worst by the margin
  that matters — its gap to persistence, not its raw MAPE.
- **Delhi** is the only city where the model reliably earns its keep, despite
  having the highest absolute AQI and the largest variance.
- Bengaluru's low raw MAPE reflects a city with low AQI variability, where
  persistence is correspondingly easy to match and hard to beat. A low MAPE
  ranked against nothing was never evidence of skill.

The right way to rank a forecaster is by skill against a baseline, and by that
measure there is very little to rank.

### Where the model is at its worst

The relative-skill losses grow with horizon in every city except Delhi at 1 day.
Beyond about 3 days the learned model reverts toward the wrong level while
persistence degrades gracefully, so the gap widens. Climatology is worse than
both everywhere (33–65% MAPE) and is included only as a floor.

## Why the original report was wrong

The original report claimed single-digit percentage MAPE — roughly one to three
percent, depending on city — on a time-based holdout. Those figures were
produced by target leakage, not by forecasting skill. Two engineered features
carried the answer into the inputs:

- **`aqi_city_zscore`** was *(today's AQI − city mean) ÷ city standard
  deviation* — an invertible function of the value being predicted, correlating
  **1.000** with it. Any model given this feature can recover the target
  exactly.
- **Rolling means such as `aqi_roll3_mean`** used windows that **included the
  current day**, so each predictor contained a fraction of its own target.

A third defect compounded the first two: the forecast held every feature at its
last observed value and recursed, which returned the same number for every
future day and was never scored against a baseline on matched `(t, h)` pairs.

Both leaks are fixed. The regression guard is
[`tests/test_backtesting.py::TestNoTargetLeakage`](../tests/test_backtesting.py),
which fails if a rolling window includes the current day or if a
target-derived normalisation feature reappears in the feature set.

## Architecture

```
lib/
  feature_engineering.py   — transformation functions + pipeline
  ml_pipeline.py           — Dataset builder, time split, feature selection
  model_training.py        — model trainers (MA, SN, XGB, RF, Prophet)
  model_evaluation.py      — Cross-city eval, error analysis, seasonal breakdown
  forecasting_service.py   — Train/save/load/predict, dashboard integration
```

## Known Limitations

1. **Target leakage previously invalidated all reported accuracy.** Two features
   leaked the target: `aqi_city_zscore`, an invertible transform of the same-day
   AQI correlating 1.000 with it, and rolling aggregates such as
   `aqi_roll3_mean` whose windows included the current day. Both are removed and
   guarded by `tests/test_backtesting.py::TestNoTargetLeakage`. Any figure in
   this repository predating that guard should be treated as unverified.
2. **The model does not beat persistence.** On 20 of 24 city/horizon
   combinations persistence is at least as good, and the four exceptions are
   small. This is the measured result, not a caveat on an otherwise positive
   one.
3. **Thin evidence for Mumbai and Kolkata.** Two folds each, 180 scored points
   per horizon, versus four folds and 360 points for the other cities. Their
   numbers — including Kolkata's two nominal wins — carry correspondingly wide
   uncertainty.
4. **Data coverage.** Real CPCB data ends mid-2020; later coverage is
   synthetic, so the backtest exercises a limited range of real regimes.
5. **No hyperparameter tuning.** Default XGBoost parameters are used for
   consistency across cities. Tuning is unlikely to close a gap this size, but
   it has not been attempted.

## Recommendations

1. Report skill against persistence, never raw MAPE alone, in every future
   evaluation.
2. Extend real observations past 2020 so the backtest covers more real regimes
   and more folds for Mumbai and Kolkata.
3. Investigate whether meteorological covariates (wind, boundary-layer height,
   precipitation) supply the information the pollutant history does not — that
   is the most plausible route to genuine skill beyond 1 day.
4. Keep the negative result published. A forecaster that loses to persistence
   and says so is more useful than one that does not measure.
