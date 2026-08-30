# Roadmap: v2 — Weather-Aware Forecasting

## Milestone: v2.0 — Give the model the variable it was missing

**Status:** Planned
**Branch:** `v2-benchmark-forecasting`
**Depends on:** v1.0 (complete, on `main`)
**External dependencies:** none — Open-Meteo needs no API key, no registration,
and no payment method

## Why this milestone exists

v1 ended on an honest negative result. After removing target leakage, the
XGBoost models beat a persistence baseline reliably only for Delhi at 1–3 days,
and degraded badly past a week. The diagnosis at the time was that the models
were not short of algorithms — they were short of an input. Air quality is
driven by dispersion, and dispersion is weather.

That diagnosis is now supported by evidence rather than intuition. Joining
Open-Meteo's ERA5 archive to Delhi's existing CPCB record, 912 days across
2018–2020:

| Weather variable | Correlation with daily AQI |
|---|---|
| Mean temperature | **−0.499** |
| Precipitation | −0.314 |
| Max wind speed | −0.199 |
| Relative humidity | −0.085 |

Colder, stiller, drier days are dirtier days — the winter inversion showing up
in the project's own data. None of these variables are in the feature set.

There is a second reason this matters more than the raw correlations suggest.
v1's models degrade with horizon partly because unknown future pollutants are
frozen at their last observed value. Weather is different: **forecasts of it
exist**, so weather features are genuinely available at prediction time. This is
the one input that can improve long-horizon accuracy rather than just fitting
the past better.

## Why Open-Meteo

| | Open-Meteo | OpenWeather | Google AQ |
|---|---|---|---|
| Payment method required | **No** | No | Yes — rejected |
| API key required | **No** | Yes | Yes |
| Weather archive | **1940 →** | paid tiers | — |
| Air quality archive | 2013 → | Nov 2020 → | 30 days |
| Forecast horizon | 7 days | 4 days | 4 days |

Decisive points: the weather archive **covers the existing 2015–2020 training
data**, so meteorology can be added retroactively rather than only going
forward; and it exposes **boundary layer height**, the mixing-height variable
that governs pollution dispersion and which few free sources publish.

Open-Meteo also publishes a CAMS-based air quality **forecast**, which restores
the external benchmark that dropping Google removed — at no cost and with no
card.

## What this milestone is not

- **Not replacing CPCB measurements with model output.** Open-Meteo air quality
  is CAMS model data and its weather archive is ERA5 reanalysis. Neither is a
  station measurement. They are labelled as such or they are not ingested.
- **Not assuming weather helps.** Phase 2 exists to measure that, and a negative
  result is reported the way v1's was.
- **Not chasing a better headline number.** If weather does not move the skill
  score, that is the finding.

## Constraints

1. **Model data, not measurements.** CAMS and ERA5 are modelled. Every ingested
   row must carry a `data_source` that distinguishes it from CPCB station data.
2. **No NAQI.** Open-Meteo returns European and US AQI, not India's 0–500 NAQI.
   Pollutant concentrations are returned, and `lib/aqi.py` already computes CPCB
   AQI from pollutants, so the index is derived rather than taken.
3. **Attribution required:** CAMS ENSEMBLE as data provider, and Open-Meteo as
   the source, wherever the data is shown.
4. **Non-commercial use only** under the keyless tier. This project qualifies;
   the constraint is recorded so it is not forgotten later.
5. **A five-month gap** sits between the CPCB record ending 2020-07-01 and any
   contiguous replacement. Phase 3 must decide how to handle it rather than
   silently interpolating across it.

## Phases

- [ ] **Phase 1: Weather on the existing record** — join ERA5 to 2015–2020 CPCB
      data, with provenance
- [ ] **Phase 2: Does weather actually help?** — re-run the v1 benchmark with
      weather features and measure the change
- [ ] **Phase 3: Extend the record to the present** — close the gap since 2020
      and label what is modelled
- [ ] **Phase 4: Forecast with forecast weather** — predict future dates using
      predicted weather, and benchmark against CAMS
- [ ] **Phase 5: Surface and narrative** — put the result in the dashboard and
      rewrite the case study around it

---

### Phase 1: Weather on the existing record

**Goal:** Every CPCB day in the database carries the weather that produced it.

**Depends on:** nothing. Unblocked.

**Success criteria:**
1. `lib/providers/open_meteo.py` fetches the ERA5 archive for a lat/lon and date
   range, returning temperature, wind speed and direction, precipitation,
   relative humidity, surface pressure and **boundary layer height**.
2. Weather is stored in its own table keyed on `(city, date)`, never merged into
   `city_measurements`, so measured air quality and modelled weather can never
   be confused for one another.
3. Rows carry `data_source='open_meteo_era5'`.
4. The six dashboard cities have weather for their full CPCB range, and the
   city-to-coordinate mapping is written down as an explicit choice — CPCB
   aggregates many stations, ERA5 is a grid point, and the two are not the same
   place.
5. The bundled demo database still works with no network access.
6. `scripts/fetch_openaq.py` is deleted: it targets an endpoint that returns
   HTTP 410 Gone and has been dead code for the life of the repo.

---

### Phase 2: Does weather actually help?

**Goal:** Know whether meteorology improves forecast skill, measured the same
way v1 measured everything else.

**Depends on:** Phase 1.

**Success criteria:**
1. Weather features are added to `build_feature_pipeline` — including lagged
   weather, and weather for the **target** date, which is legitimate because a
   forecaster would have a weather forecast for that date.
2. `tests/test_backtesting.py` gains leakage checks for the new features: no
   weather feature may correlate with the target above the threshold that
   already guards the existing ones.
3. `scripts/backtest.py` runs unchanged and reports MAPE per horizon for six
   cities with and without weather, against persistence.
4. The comparison is stated plainly in `data/backtest_results.json`: skill with
   weather, skill without, and the difference per horizon.
5. **If weather does not improve skill, that is written down as the result** and
   the remaining phases are re-scoped rather than continued on momentum.

**Why this is a gate:** every later phase assumes weather is worth carrying. If
it is not, Phases 3–5 change shape entirely, and finding that out here costs one
phase instead of four.

---

### Phase 3: Extend the record to the present

**Goal:** The project stops being frozen in July 2020.

**Depends on:** Phase 2 clearing its gate.

**Success criteria:**
1. Open-Meteo CAMS air quality is ingested from 2013 to the present for the six
   cities, with pollutant concentrations mapped onto existing column names.
2. NAQI is **computed** from those concentrations via `lib/aqi.py`, never taken
   from a European or US index, and the derivation is documented.
3. Every row carries `data_source='open_meteo_cams'`, and the dashboard's
   existing provenance filter treats it as a third category alongside real CPCB
   and synthetic.
4. The 2015–2020 overlap between CAMS and CPCB is **quantified** — how closely
   does the model agree with the stations? That comparison is itself a finding
   worth reporting, and it tells a reader how much to trust the extended record.
5. The five-month gap after 2020-07-01 is handled explicitly and visibly, not
   interpolated over.

---

### Phase 4: Forecast with forecast weather

**Goal:** Forecasts concern actual future dates, using weather that is itself
forecast — and are scored against a production system.

**Depends on:** Phase 3.

**Success criteria:**
1. The 7-day Open-Meteo weather forecast feeds the model's weather features, so
   predictions use forecast weather rather than frozen last-known values.
2. Open-Meteo's CAMS air quality forecast is scored alongside, restoring the
   three-way comparison: persistence as the floor, this project in the middle, a
   production forecast as the ceiling.
3. Scoring uses only forecasts recorded **before** the outcome was known.
4. Degradation from perfect to forecast weather is measured and reported: how
   much accuracy is lost because tomorrow's weather is itself uncertain. Skipping
   this would overstate the model.
5. The demo still runs offline against the bundled database.

---

### Phase 5: Surface and narrative

**Goal:** A visitor sees what changed and why it matters.

**Depends on:** Phase 4.

**Success criteria:**
1. The forecasting section plots persistence, this model, and CAMS together,
   with the plain-English framing established in v1.
2. The accuracy table reports skill with and without weather, per horizon.
3. The in-app case study is rewritten: v1 found the model barely beat
   persistence and diagnosed a missing input; v2 supplied it and measured the
   result — whatever that result was.
4. Attribution for CAMS and Open-Meteo appears wherever their data is shown.
5. No figure appears anywhere that the backtest output does not support.

---

## Progress

| Phase | Status | Blocked on |
|---|---|---|
| 1. Weather on the existing record | Not started | — |
| 2. Does weather actually help? | Not started | Phase 1 |
| 3. Extend the record to the present | Not started | Phase 2 gate |
| 4. Forecast with forecast weather | Not started | Phase 3 |
| 5. Surface and narrative | Not started | Phase 4 |

## Risks

| Risk | Handling |
|---|---|
| Weather does not improve skill | Phase 2 is a gate; a negative result is reported and the milestone re-scoped |
| Modelled data gets mixed with measured data | Separate table in Phase 1, distinct `data_source` throughout, provenance surfaced in the dashboard |
| Target-date weather is mistaken for leakage | It is not — a forecaster genuinely has a weather forecast. Documented, and the existing leakage tests are extended to cover the new features |
| ERA5 grid point does not represent the city CPCB measures | Mapping recorded as an explicit, reviewable choice in Phase 1 |
| CAMS disagrees badly with CPCB over the overlap | That is a finding, and Phase 3 reports it rather than hiding it |
| Forecast weather is much worse than reanalysis | Phase 4 measures the gap explicitly instead of quoting perfect-weather accuracy |
| Non-commercial licence terms change | Recorded in constraints; the bundled demo works offline regardless |
