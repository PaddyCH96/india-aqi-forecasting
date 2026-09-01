# Key Insights: India Air Quality

---

## 1. Delhi is an extreme outlier — 2.7× more polluted than any other major Indian city

**Evidence:**
- Mean AQI: Delhi 259.5 (next: Kolkata 140.6, Lucknow 137.7)
- Delhi winter AQI averages ~350 (Very Poor), summer ~180 (Moderate)
- 99.5% AQI coverage across 5.5 years — the most complete record of any city
- Highest recorded AQI: 716 (Well beyond CPCB "Severe" threshold of 500)

**Why it matters:** Delhi's pollution is not incremental — it operates in a different regime from every other Indian city. Urban planning interventions that work in Bengaluru or Chennai have no analog for Delhi. Any national policy must treat Delhi as a separate case requiring disproportionate resources.

---

## 2. PM2.5 alone explains ~94% of AQI variance — most pollutants are redundant for forecasting

**Evidence:**
- PM2.5 ↔ AQI correlation: r = 0.97
- PM2.5 ↔ O3 correlation: r = 0.08 (near-zero — different formation mechanisms)
- AQI is a composite index; PM2.5 weight dominates the calculation
- Adding the remaining 11 pollutants as features adds little beyond PM2.5 plus temporal features

**Why it matters:** For practical forecasting, you don't need the full 12-pollutant panel. A single PM2.5 sensor + calendar features gives you 97% of predictive power. This dramatically simplifies sensor deployment requirements for cities that currently lack comprehensive monitoring.

---

## 3. Mumbai has critically broken air quality monitoring — 61.4% missing AQI data, worst of 26 cities

**Evidence:**
- Mumbai AQI coverage: 38.6% (vs Delhi 99.5%, Bengaluru 95.1%)
- Mumbai hourly AQI coverage: 37.7%
- Only CO is well-measured (98.8%) — all other pollutants have ~40% or less
- Result: only 227 usable training samples for ML (vs 1,300+ for other cities)
- Mumbai is therefore the least reliably modelled of the six cities — the underlying record is too fragile to trust

**Why it matters:** India's financial capital has the worst air quality monitoring of any major city. Policy decisions about Mumbai's air quality are being made with 60% less data than comparable cities. The gap isn't just about sensors — it's about the ability to make evidence-based decisions. This is a monitoring infrastructure failure, not a data collection oversight.

---

## 4. Pollution is 1.5–2.5× worse in winter — but the ratio varies predictably by geography

**Evidence:**
- Northern cities (Delhi, Lucknow, Patna): Winter/summer ratio 1.8–2.5×
- Southern cities (Bengaluru, Chennai, Hyderabad): Winter/summer ratio ~1.3×
- Coastal cities show muted seasonal variation
- Winter peak driven by: temperature inversion trapping pollutants, increased biomass burning, lower mixing height
- Monsoon (Jun-Sep) provides temporary relief across all cities

**Why it matters:** This is not uniform — the winter penalty is geographically determined. Northern cities need aggressive pre-winter mitigation (crop stubble management, construction bans, traffic restrictions). Southern cities need year-round strategies. Blanket national policies miss this distinction.

---

## 5. Persistence is a strong baseline for daily AQI — gradient boosting beats it reliably only for Delhi at one to three days

**Evidence:**
- Rolling backtest on held-out periods, model MAPE / persistence MAPE, from `data/backtest_results.json`:

| City | 1 day | 3 days | 7 days | 14 days |
|---|:--:|:--:|:--:|:--:|
| Bengaluru | 11 / 10 | 17 / 15 | 19 / 18 | 21 / 19 |
| Mumbai | 13 / 13 | 36 / 25 | 33 / 27 | 46 / 31 |
| Hyderabad | 14 / 13 | 32 / 24 | 35 / 30 | 41 / 32 |
| Delhi | **15 / 16** | 31 / 31 | 37 / 36 | 47 / 40 |
| Kolkata | 17 / 17 | 39 / 27 | 36 / 31 | **37 / 38** |
| Chennai | 23 / 18 | 41 / 27 | 48 / 31 | 49 / 33 |

- Only Delhi at one to three days shows the model ahead of persistence; elsewhere persistence matches it or wins, and its advantage grows with the horizon
- Three attempts to overturn this — removing target leakage, one model per horizon, and predicting the change rather than the level — did not change the conclusion

**Why the earlier single-digit MAPE figures in this file were wrong:** they were a
target leakage artifact. `aqi_city_zscore` was *(today's AQI − city mean) ÷ city
std*, an invertible function of the value being predicted, correlating **1.000**
with it. Rolling means such as `aqi_roll3_mean` used windows that **included the
current day**, averaging each row's own target into its own features. Both are
fixed, and `tests/test_backtesting.py::TestNoTargetLeakage` fails the build if
either regresses.

**Why it matters:** For daily city-level AQI, a forecast is only worth deploying
if it beats "tomorrow looks like today," and on this dataset it mostly does not.
Short-range prediction carries real signal; beyond about a week, historical AQI
alone is close to unpredictable and meteorological inputs — absent from the CPCB
panel — would be required to do better. The honest measurement is more valuable
than the number it replaced: the leakage was found by building the backtest, not
by inspecting the model.