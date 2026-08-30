# Roadmap: v2 — Benchmarked Forecasting

## Milestone: v2.0 — Is this model any good, measured against a real one?

**Status:** Planned
**Branch:** `v2-benchmark-forecasting`
**Depends on:** v1.0 (complete, on `main`)

## Why this milestone exists

v1 ended on an honest negative result: after removing target leakage, the
XGBoost models beat a persistence baseline reliably only for Delhi at 1–3 days.
That is a defensible finding, but it leaves one question unanswered — **is that
because the model is weak, or because the problem is hard?**

Persistence is a floor. It says nothing about the ceiling. The Google Maps Air
Quality API publishes an operational 96-hour forecast for the same cities, on
the same `ind_cpcb` NAQI index, over the same seven pollutants. Scoring against
it converts an open question into a measurement:

| Reference | What it tells us |
|---|---|
| Persistence | the floor any forecast must clear |
| This project's XGBoost | where the work currently stands |
| **Google Air Quality** | what a production system achieves |

If Google also degrades sharply past three days, v1's negative result is
evidence about the problem rather than about the model. If Google holds up, the
gap is the roadmap.

## What this milestone is not

- **Not replacing the forecast with Google's.** Displaying a vendor's prediction
  as the product would delete the substance of the project. Google is the
  yardstick, never the output.
- **Not rebuilding the training set from Google.** History is capped at 30 days;
  the models train on ~2,009 days per city. It cannot backfill.
- **Not a fix for long-horizon accuracy.** The missing input is meteorology,
  not another data source for the same variable.

## Constraints discovered before planning

1. **History is 30 days maximum.** Live feed, not an archive.
2. **Forecast horizon is 96 hours.** Comparison is capped at 4 days; the 7- and
   14-day horizons in v1 have no Google counterpart.
3. **Caching is restricted.** Only place IDs may be stored indefinitely; all
   other content falls under the Google agreement's caching terms. This governs
   whether fetched forecasts may be retained for later scoring — **must be
   confirmed against the account's agreement before any storage is designed.**
4. **Billing account required.** 10,000 requests/month free (Essentials tier),
   then $5/1,000. Projected use is ~4,300/month, but billing must be enabled, so
   spend control is a requirement rather than a nicety.
5. **Attribution required:** "Source: Includes air quality data from Google"
   wherever the data is shown.

## Phases

- [ ] **Phase 1: Access and provenance** — a credentialled, rate-limited client
      that fetches `ind_cpcb` data and records where every row came from
- [ ] **Phase 2: Storage decision** — settle what may be retained, and design
      around the answer
- [ ] **Phase 3: Benchmark harness** — score three forecasters on identical
      (origin, horizon) pairs
- [ ] **Phase 4: Live scoreboard** — accumulate real forward-looking results
- [ ] **Phase 5: Surface and narrative** — put the comparison in the dashboard
      and rewrite the case study around it

---

### Phase 1: Access and provenance

**Goal:** The project can fetch Google air quality data for its six cities,
safely, with every row traceable to its source.

**Depends on:** A Google Maps Platform API key with the Air Quality API enabled.
**Blocked until the key exists — this is the only external dependency.**

**Success criteria:**
1. `lib/providers/google_aq.py` fetches current conditions and 96h forecast for
   a lat/lon, requesting the `ind_cpcb` index, and returns a frame matching the
   existing `city_measurements` column names.
2. The API key is read from the environment (`GOOGLE_AQ_API_KEY`) and never
   committed; the bundled demo continues to work with no key present.
3. A hard per-run request cap makes runaway spend structurally impossible, and
   the client refuses to run without the cap set.
4. Every fetched row carries `data_source='google_aq'` and `is_synthetic=false`,
   so Google-derived rows can never be silently mixed with CPCB rows.
5. `scripts/fetch_openaq.py` is deleted — it targets a v2 endpoint that returns
   HTTP 410 Gone and has been dead code for the life of the repo.

**Open questions:**
- Which coordinates represent a "city"? CPCB aggregates many stations; Google is
  point-based at 500m. The choice changes what is being compared and must be
  documented, not assumed.

---

### Phase 2: Storage decision

**Goal:** Know what may legally be retained, and design storage around that
answer rather than discovering it later.

**Depends on:** Phase 1.

**Success criteria:**
1. The caching and retention terms in the account's Google agreement are read
   and summarised in `docs/data_sources.md`, with the permitted retention stated
   explicitly.
2. The design records **whichever of these the terms allow**:
   - if retention is permitted: fetched forecasts stored for later scoring;
   - if it is not: only *derived* scores (error metrics per origin and horizon)
     are stored, and raw responses are discarded after scoring.
3. Whichever path is taken, the reason is written down, because a reviewer
   should see a licensing constraint handled deliberately.

**Note:** This phase exists because getting it wrong is expensive to undo — the
entire scoreboard design depends on the answer.

---

### Phase 3: Benchmark harness

**Goal:** Three forecasters scored on identical (origin, horizon) pairs, so the
comparison is like for like.

**Depends on:** Phase 2.

**Success criteria:**
1. `lib/benchmarking.py` scores persistence, this project's direct per-horizon
   XGBoost, and Google's forecast over the same origins and the same horizons
   (1h–96h, capped by Google's limit).
2. Horizons beyond 96 hours report the two internal forecasters only, and say
   plainly that no external comparison exists at that range — rather than
   leaving a blank a reader might misread as a tie.
3. Results are written to `data/benchmark_results.json` in the same shape as the
   existing `backtest_results.json`, so the dashboard reads both the same way.
4. The comparison is scored on **future** data only: Google's forecast is
   collected before the outcome is known, never fitted after the fact.

---

### Phase 4: Live scoreboard

**Goal:** Forecasts are logged before the outcome is known, then scored once it
arrives — the thing a real forecasting team runs.

**Depends on:** Phase 3.

**Success criteria:**
1. A scheduled GitHub Actions job records, each day: this project's forecast,
   Google's forecast, and the observed value for the previous cycle.
2. Scores accumulate over time and survive restarts, so the record grows rather
   than resetting.
3. The job fails loudly and visibly when the API key is missing, the quota is
   exhausted, or the request cap is hit — silence would let a broken scoreboard
   look like a working one.
4. The demo still runs with no key: the scoreboard degrades to whatever has been
   collected so far, and says so.

---

### Phase 5: Surface and narrative

**Goal:** A visitor sees where this model stands against a production forecast,
and understands what that means.

**Depends on:** Phase 4.

**Success criteria:**
1. The forecasting section plots all three forecasters together, with the same
   plain-English framing as the v1 captions.
2. The accuracy table gains a Google column and names the winner per horizon.
3. The in-app case study is rewritten around the benchmark: what the negative
   result in v1 meant, and what comparing against a production system revealed.
4. Google attribution appears wherever its data is shown.
5. README headline claims are rewritten from measured results — no figure
   appears that the benchmark output does not support.

---

## Progress

| Phase | Status | Blocked on |
|---|---|---|
| 1. Access and provenance | Not started | **Google API key** |
| 2. Storage decision | Not started | Phase 1 + reading the agreement |
| 3. Benchmark harness | Not started | Phase 2 |
| 4. Live scoreboard | Not started | Phase 3 |
| 5. Surface and narrative | Not started | Phase 4 |

## Risks

| Risk | Handling |
|---|---|
| Caching terms forbid retaining forecasts | Phase 2 decides before anything is built on the assumption |
| Google's forecast is much better, at every horizon | That is a finding, not a failure — it quantifies the gap and gives the roadmap |
| Google is *also* poor past 3 days | Vindicates v1's negative result; the strongest outcome for the narrative |
| Runaway API spend | Hard request cap in the client; the client refuses to run without one |
| The live scoreboard has too little data to mean anything early | State the sample size next to every score, as v1 does with `n_points` |
| City-to-coordinate mapping makes the comparison unfair | Documented in Phase 1 as an explicit, reviewable choice |
