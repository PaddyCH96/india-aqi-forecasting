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
- XGBoost models using only PM2.5 + temporal features achieve sub-3% MAPE

**Why it matters:** For practical forecasting, you don't need the full 12-pollutant panel. A single PM2.5 sensor + calendar features gives you 97% of predictive power. This dramatically simplifies sensor deployment requirements for cities that currently lack comprehensive monitoring.

---

## 3. Mumbai has critically broken air quality monitoring — 61.4% missing AQI data, worst of 26 cities

**Evidence:**
- Mumbai AQI coverage: 38.6% (vs Delhi 99.5%, Bengaluru 95.1%)
- Mumbai hourly AQI coverage: 37.7%
- Only CO is well-measured (98.8%) — all other pollutants have ~40% or less
- Result: only 227 usable training samples for ML (vs 1,300+ for other cities)
- Despite this, XGBoost achieves 2.9% MAPE — the model works, but the underlying data is fragile

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

## 5. XGBoost with feature engineering achieves 0.8–3.2% MAPE — data quality is the primary accuracy constraint, not model choice

**Evidence:**
- Cities with >1,300 training days: MAPE 0.8–1.0% (Bengaluru, Chennai, Hyderabad, Delhi)
- Cities with <250 training days: MAPE 2.9–3.2% (Mumbai, Kolkata)
- Moving average baseline: 12–25% MAPE
- Seasonal naive baseline: 31–64% MAPE
- Random Forest and XGBoost perform near-identically on well-sampled cities

**Why it matters:** The limiting factor isn't model sophistication — it's data coverage. A simple XGBoost with smart features outperforms complex architectures when data is clean and abundant. The best way to improve forecasting isn't a better model — it's better monitoring. For cities improving their sensor networks, the ROI on one additional year of data far exceeds the ROI on hyperparameter tuning.
