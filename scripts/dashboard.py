#!/usr/bin/env python3
"""Unified India AQI Analytics Dashboard.

Pages:
1. Executive Summary — National snapshot, KPI cards, city ranking
2. Historical Trends — Multi-city trends, seasonal analysis
3. Pollutant Drill-Down — Per-pollutant analysis, correlations
4. City Deep-Dive — Single city analysis, diurnal patterns
5. Data Quality — Missing data, coverage warnings
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.pathing import ensure_project_root_on_path

ensure_project_root_on_path()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from lib.db import get_engine, load_city_pollutants, get_data_freshness
from lib.config import CITIES
from lib.charts import (
    plot_multi_city_trends,
    plot_city_ranking,
    plot_missing_heatmap,
    plot_correlation_heatmap,
    plot_pollutant_distribution,
    plot_aqi_category_bars,
    plot_diurnal_pattern,
    plot_seasonal_box,
    plot_history,
)
from lib.analysis import (
    city_ranking,
    aqi_distribution,
    missing_heatmap,
    correlation_matrix,
    summer_winter_comparison,
    year_over_year,
    worst_best_cities,
    monthly_trends,
    pollutant_summary,
)
from lib.forecasting_service import (
    get_forecast_for_dashboard,
    forecast_with_baseline,
    load_backtest_results,
)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="India AQI Analytics", page_icon="🌍", layout="wide")
engine = get_engine()

POLLUTANTS = ["aqi", "pm2_5", "pm10", "no2", "co", "o3", "so2", "no", "nh3"]
POLLUTANT_LABELS = {
    "aqi": "AQI", "pm2_5": "PM2.5 (µg/m³)", "pm10": "PM10 (µg/m³)",
    "no2": "NO₂ (µg/m³)", "co": "CO (mg/m³)", "o3": "O₃ (µg/m³)",
    "so2": "SO₂ (µg/m³)", "no": "NO (µg/m³)", "nh3": "NH₃ (µg/m³)",
}
AQI_BUCKET_COLORS = {
    "Good": "green", "Satisfactory": "lightgreen",
    "Moderate": "orange", "Poor": "red",
    "Very Poor": "darkred", "Severe": "maroon",
}

st.title("🌍 India Air Quality Analytics")
st.markdown("**Comprehensive analytics across 26 Indian cities | CPCB Data 2015-2020**")

freshness = get_data_freshness(engine)

# ─── Sidebar ─────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
use_synthetic = st.sidebar.checkbox(
    "Include synthetic data (2020-2024)", value=False,
    help="Real CPCB data: 2015–2020. Synthetic extends to 2024 for 6 cities."
)

# 26-city selector — default to top 6 polluted
all_cities_list = sorted(CITIES)
default_cities = ["Delhi", "Kolkata", "Mumbai", "Chennai", "Bengaluru", "Hyderabad"]
selected_cities = st.sidebar.multiselect(
    "🌆 Cities", all_cities_list, default=default_cities
)

pollutant = st.sidebar.selectbox(
    "🧪 Pollutant", POLLUTANTS,
    format_func=lambda x: POLLUTANT_LABELS.get(x, x.upper())
)

if not selected_cities:
    st.warning("Select at least one city to begin.")
    st.stop()

# ─── Data Loading ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_pollutant_data(cities, use_synthetic):
    """use_synthetic is an explicit argument so it forms part of the cache key.

    Read from the enclosing scope it would not, and toggling the sidebar
    checkbox would keep serving the previously cached frame.
    """
    all_data = []
    for city in cities:
        df = load_city_pollutants(engine, city, use_synthetic=use_synthetic)
        if not df.empty:
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


@st.cache_data(ttl=300)
def load_hourly_data(cities):
    """Hourly data is optional.

    city_hourly_measurements is built by scripts/ingest_hourly.py from
    data/raw/city_hour.csv, which is not committed. Deployments without it
    (the bundled SQLite demo) return an empty frame; the Hourly Patterns
    section is skipped rather than raising.
    """
    from sqlalchemy import text as sa_text, inspect as sa_inspect

    if not sa_inspect(engine).has_table("city_hourly_measurements"):
        return pd.DataFrame()

    dfs = []
    for city in cities:
        df = pd.read_sql(
            sa_text("""
                SELECT city, datetime, pm2_5, pm10, no2, co, o3, aqi
                FROM city_hourly_measurements
                WHERE city = :city AND NOT is_synthetic
                ORDER BY datetime
            """), engine, params={"city": city}
        )
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    # SQLite returns TIMESTAMP columns as text; the daily loader does the same.
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out


with st.spinner("Loading data..."):
    df = load_pollutant_data(selected_cities, use_synthetic)

if df.empty:
    st.warning("No data loaded. Try selecting different cities or enabling synthetic data.")
    st.stop()

df_hourly = load_hourly_data(selected_cities)

# ─── Case Study ──────────────────────────────────────────────────
st.markdown("---")
with st.expander("📄 **Case study — what this project set out to do, and what it found**", expanded=False):
    st.markdown("""
### The question

Can day-to-day air quality in Indian cities be forecast well enough to be
useful — and how would we know if it could?

India runs one of the world's largest air quality monitoring networks through
the Central Pollution Control Board. This project takes 29,531 daily records
across 26 cities from 2015 to 2020 and asks whether machine learning can
predict what tomorrow, next week, or the week after will look like.

### What was built

A pipeline that ingests the CPCB records, tracks the provenance of every row,
engineers 66 predictive features per city, and trains gradient-boosted models
to forecast AQI at 1, 3, 7 and 14 days ahead — with a dashboard and REST API
on top.

### The finding that mattered

**The first version of this project reported 0.8–3.2% forecast error. That
number was wrong, and finding out why is the most useful thing here.**

Two features leaked the answer into the inputs:

- `aqi_city_zscore` was computed as *(today's AQI − city mean) ÷ city standard
  deviation*. It is an invertible function of the very value being predicted —
  correlation with the target was exactly **1.000**. The model could read the
  answer off it.
- Rolling averages such as `aqi_roll3_mean` were computed over a window that
  **included the current day**, putting a third of the answer inside its own
  predictor.

Both are now computed from prior days only. And the "forecast" itself was
holding every feature frozen at its last observed value, so it returned the
same number for every future day — a flat line presented as a prediction.

### What honest measurement showed

Rebuilt properly — one model per horizon, calendar features advanced to the
date being predicted, rolling-origin backtests scored over held-out blocks of
90 days — the picture changes completely:

| Days ahead | Model error | Persistence baseline | Verdict |
|---|---|---|---|
| 1 day | ~15% | ~16% | Model marginally ahead |
| 3 days | ~31% | ~31% | Level |
| 7 days | ~37% | ~36% | Baseline ahead |
| 14 days | ~47% | ~40% | Baseline clearly ahead |

*(Delhi; full per-city figures in the Forecasting section below.)*

**Persistence** — simply assuming tomorrow looks like today — is a remarkably
strong baseline for daily AQI, and across six cities the machine learning model
beat it reliably only for Delhi at one to three days. Three separate attempts
to improve on it (removing the leakage, direct per-horizon models, and
predicting the change rather than the level) did not overturn that.

### Why report a negative result

Because it is the true one, and because a model that cannot beat "tomorrow is
like today" is not adding value no matter how sophisticated it looks. The
honest conclusion for this data is:

1. **Short-range forecasting works.** One to three days out, there is real
   signal beyond persistence.
2. **Beyond a week, daily city-level AQI is close to unpredictable** from
   historical readings alone. Meteorological inputs — wind, temperature
   inversion, rainfall — are what the model is missing, not more trees.
3. **Data quality determines accuracy far more than model choice.** Mumbai has
   61% of daily AQI records missing; no algorithm recovers from that.

### What I would do next

Add meteorological covariates, which is the single change most likely to move
long-horizon skill. Extend to a live data feed so the forecasts concern
tomorrow rather than a historical hold-out. And forecast the pollutants
themselves rather than freezing them, which is the main structural limit on
the current recursive approach.
""")
    st.caption(
        "Full write-up, including the five headline findings and their evidence, "
        "is in CASE_STUDY.md and INSIGHTS.md in the repository."
    )

# ─── Overview Metrics ────────────────────────────────────────────
st.markdown("---")
kpi_cols = st.columns(5)
worst = worst_best_cities(df, year=df["date"].dt.year.max())

latest = df.dropna(subset=["aqi"]).groupby("city").last().reset_index()
worst_now = latest.loc[latest["aqi"].idxmax()] if not latest.empty else None
best_now = latest.loc[latest["aqi"].idxmin()] if not latest.empty else None

kpi_cols[0].metric("Cities", len(selected_cities))
kpi_cols[1].metric("Date Range",
    f"{df['date'].min().date()} – {df['date'].max().date()}")
kpi_cols[2].metric("Total Rows", f"{len(df):,}")
kpi_cols[3].metric(
    "Highest Avg",
    f"{worst['worst']}" if worst["worst"] else "—"
)
kpi_cols[4].metric(
    "Lowest Avg",
    f"{worst['best']}" if worst["best"] else "—"
)

if use_synthetic:
    st.info(
        "⚠️ **Synthetic data enabled**: Values after July 2020 are simulated. "
        "Not suitable for scientific analysis."
    )
else:
    st.info(
        f"📊 Real CPCB data only: {freshness['real_rows']:,} rows across "
        f"{freshness['cities']} cities. Data ends {freshness['latest_real_date']}."
    )

# ═══════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════
st.header("1️⃣ Executive Summary")
col1, col2 = st.columns([3, 2])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ranking = city_ranking(df, metric=pollutant)
    plot_city_ranking(ax, ranking, metric="Mean", title=f"City Ranking by Mean {pollutant.upper()}")
    st.pyplot(fig)
    st.caption(
        "Each bar is one city's average air quality over the period you selected. Longer bars mean dirtier air. The ordering is what matters more than the exact numbers — it shows which cities carry the heaviest pollution burden."
    )

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    dist = aqi_distribution(df)
    plot_aqi_category_bars(ax, dist, top_n=min(26, len(selected_cities)))
    st.pyplot(fig)
    st.caption(
        "The same cities broken into official AQI categories, from Good to Severe. A bar that is mostly green means most days are breathable; mostly red or purple means residents spend much of the year in unhealthy air."
    )

# Trend overview
st.subheader("National Trend Overview")
fig, ax = plt.subplots(figsize=(14, 4))
yoy = year_over_year(df, pollutant=pollutant)
for city in selected_cities:
    c = yoy[yoy["city"] == city]
    ax.plot(c["year"], c["mean"], marker="o", label=city, alpha=0.7)
ax.set_title(f"Year-over-Year {pollutant.upper()} Trend")
ax.set_xlabel("Year")
ax.set_ylabel(pollutant.upper())
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
st.pyplot(fig)
st.caption(
    "The national picture over time. A falling line means air is getting cleaner across the country; a rising one means the opposite. The sharp improvement in early 2020 is the COVID lockdown, when traffic and industry largely stopped."
)

# ═══════════════════════════════════════════════════════════════════
# PAGE 2: HISTORICAL TRENDS
# ═══════════════════════════════════════════════════════════════════
st.header("2️⃣ Historical Trends & Seasonality")
tab_a, tab_b, tab_c = st.tabs(["Multi-City Trends", "Seasonal Patterns", "Monthly Averages"])

with tab_a:
    freq = st.selectbox("Resample frequency", ["ME", "W", "QE"], 
                        format_func=lambda x: {"ME": "Monthly", "W": "Weekly", "QE": "Quarterly"}[x],
                        key="freq_tab_a")
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_multi_city_trends(ax, df, selected_cities, pollutant=pollutant, freq=freq)
    st.pyplot(fig)
    st.caption(
        "One line per city, so you can compare how their pollution moves over time. Lines that rise and fall together suggest a shared cause — weather or season — rather than anything specific to one city."
    )

with tab_b:
    col_a, col_b = st.columns(2)
    for i, city in enumerate(selected_cities[:6]):
        with (col_a if i % 2 == 0 else col_b):
            fig, ax = plt.subplots(figsize=(7, 3.5))
            plot_seasonal_box(ax, df, city, pollutant=pollutant)
            st.pyplot(fig)

    sw = summer_winter_comparison(df)
    if not sw.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Summer vs Winter AQI")
            st.dataframe(sw, use_container_width=True)
        with col_b:
            fig, ax = plt.subplots(figsize=(7, 4))
            plot_df = sw.dropna().sort_values("winter_summer_ratio")
            colors = ["red" if r > 1 else "green" for r in plot_df["winter_summer_ratio"]]
            ax.barh(plot_df["city"], plot_df["winter_summer_ratio"], color=colors)
            ax.axvline(x=1, color="black", linestyle="--", alpha=0.5)
            ax.set_title("Winter/Summer AQI Ratio (>1 = worse in winter)")
            ax.set_xlabel("Ratio")
            st.pyplot(fig)

with tab_c:
    monthly = monthly_trends(df, pollutant=pollutant)
    fig, ax = plt.subplots(figsize=(14, 5))
    for city in selected_cities:
        c = monthly[monthly["city"] == city]
        ax.plot(c["ym"], c[pollutant], alpha=0.7, label=city)
    ax.set_title(f"Monthly {pollutant.upper()} Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel(pollutant.upper())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.caption(
        "The same data averaged by month, which strips out day-to-day noise and makes the yearly rhythm easier to see. Most Indian cities peak in winter and clear in the monsoon."
    )

# ═══════════════════════════════════════════════════════════════════
# PAGE 3: POLLUTANT DRILL-DOWN
# ═══════════════════════════════════════════════════════════════════
st.header("3️⃣ Pollutant Drill-Down")
tab_d, tab_e, tab_f = st.tabs(["Distribution", "Correlations", "Hourly Patterns"])

with tab_d:
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_pollutant_distribution(ax, df, pollutant=pollutant)
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(7, 4))
        if selected_cities:
            plot_pollutant_distribution(ax, df, pollutant=pollutant, city=selected_cities[0])
            st.pyplot(fig)

    st.subheader("Distribution Across All Cities")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, p in zip(axes.flatten(), POLLUTANTS):
        data = df[p].dropna()
        if len(data) > 0:
            ax.hist(data, bins=40, alpha=0.6, color="steelblue", edgecolor="black")
        ax.set_title(POLLUTANT_LABELS.get(p, p.upper()), fontsize=10)
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "How often each pollutant takes each value. A tall spike on the left means readings are usually low, with occasional bad days forming the long tail to the right — and it is those tail days that cause most harm."
    )

with tab_e:
    corr_df = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_correlation_heatmap(ax, corr_df)
    st.pyplot(fig)
    st.caption(
        "How closely pollutants move together. Dark red squares mean two pollutants rise and fall almost in lockstep, usually because they come from the same source, such as vehicle exhaust or crop burning."
    )

    st.subheader("PM2.5 vs PM10 Relationship")
    fig, ax = plt.subplots(figsize=(8, 6))
    for city in selected_cities[:6]:
        c = df[df["city"] == city].dropna(subset=["pm2_5", "pm10"])
        ax.scatter(c["pm2_5"], c["pm10"], alpha=0.3, s=5, label=city)
    ax.set_xlabel("PM2.5 (µg/m³)")
    ax.set_ylabel("PM10 (µg/m³)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("PM2.5 vs PM10 by City")
    st.pyplot(fig)
    st.caption(
        "Each dot is one day in one city, comparing fine particles (PM2.5) with coarser dust (PM10). Dots forming a tight upward line mean the two travel together, so a single source is likely driving both."
    )

with tab_f:
    if not df_hourly.empty:
        st.subheader("Diurnal Patterns")
        cols = st.columns(3)
        for i, city in enumerate(selected_cities[:6]):
            with cols[i % 3]:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                plot_diurnal_pattern(ax, df_hourly, city, pollutant=pollutant)
                st.pyplot(fig)
    else:
        st.info(
            "No hourly data for the selected cities. The bundled demo database "
            "carries hourly readings from 2019-01 to 2020-07 for Delhi, Mumbai, "
            "Bengaluru, Chennai, Hyderabad and Kolkata. For other cities or the "
            "full 2015-2020 range, run `scripts/ingest_hourly.py` against the "
            "complete CPCB extract."
        )

# ═══════════════════════════════════════════════════════════════════
# PAGE 4: CITY DEEP-DIVE
# ═══════════════════════════════════════════════════════════════════
st.header("4️⃣ City Deep-Dive")
focus_city = st.selectbox("Focus City", selected_cities, key="focus_city")

col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_history(ax, df[df["city"] == focus_city].rename(
        columns={"date": "ds", "aqi": "y"}), focus_city, show_covid=True)
    st.pyplot(fig)
    st.caption(
        "The full history for this city, day by day. The repeating peaks are winters; the troughs are monsoon months when rain washes particles out of the air."
    )

with col_b:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_seasonal_box(ax, df, focus_city, pollutant=pollutant)
    st.pyplot(fig)

# Year-over-year for focus city
st.subheader(f"{focus_city}: Year-over-Year")
yoy_city = year_over_year(df[df["city"] == focus_city], pollutant=pollutant)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(yoy_city["year"].astype(str), yoy_city["mean"], color="steelblue", alpha=0.7)
ax.errorbar(yoy_city["year"].astype(str), yoy_city["mean"],
            yerr=yoy_city["std"], fmt="none", color="black", capsize=3)
ax.set_title(f"{focus_city}: Annual {pollutant.upper()} (Mean ± Std)")
ax.set_xlabel("Year")
ax.set_ylabel(pollutant.upper())
ax.grid(True, alpha=0.3, axis="y")
st.pyplot(fig)
st.caption(
    "One bar per year, with the whisker showing how much the readings varied within that year. A shrinking bar means cleaner air; a shorter whisker means more consistent air rather than wild swings."
)

# Summary stats for focus city
st.subheader(f"{focus_city}: Pollutant Summary")
city_stats = pollutant_summary(df[df["city"] == focus_city])
st.dataframe(city_stats, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 5: DATA QUALITY
# ═══════════════════════════════════════════════════════════════════
st.header("5️⃣ Data Quality & Coverage")
tab_g, tab_h = st.tabs(["Missing Data", "Data Freshness"])

with tab_g:
    missing = missing_heatmap(df)
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_missing_heatmap(ax, missing)
    st.pyplot(fig)
    st.caption(
        "Where the monitoring data is missing. Dark cells are gaps in the record. This matters because a city cannot be managed if it is not measured — and it is why some cities in this dashboard carry less confident numbers."
    )

    st.subheader("Data Completeness Warnings")
    for _, row in missing.iterrows():
        low_cols = [c for c in missing.columns[1:] if row[c] > 50]
        if low_cols:
            st.warning(f"**{row['city']}**: {', '.join(low_cols)} missing > 50%")
    good_cities = [row["city"] for _, row in missing.iterrows()
                   if all(row[c] < 20 for c in missing.columns[1:])]
    if good_cities:
        st.success(f"**Best coverage**: {', '.join(good_cities)} (all pollutants < 20% missing)")

with tab_h:
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Rows (Daily)", f"{freshness['total_rows']:,}")
        st.metric("Real Data Rows", f"{freshness['real_rows']:,}")
        st.metric("Synthetic Data Rows", f"{freshness['synthetic_rows']:,}")
    with col_b:
        st.metric("Data Sources", ", ".join(freshness["data_sources"].keys()))
        st.metric("Latest Real Date", freshness["latest_real_date"] or "—")
        st.metric("Last Ingested", freshness["last_ingested_at"] or "—")

    if use_synthetic:
        st.warning(f"Synthetic data ({freshness['synthetic_rows']:,} rows) is included. "
                   "These are simulated values, not real measurements.")

# ═══════════════════════════════════════════════════════════════════
# PAGE 6: FORECASTING
# ═══════════════════════════════════════════════════════════════════
st.header("6️⃣ AQI Forecasting")

forecast_city = st.selectbox("Forecast City", selected_cities, key="forecast_city")
horizon_days = st.select_slider(
    "Forecast horizon (days)", options=[1, 3, 7, 14], value=7,
    format_func=lambda d: f"{d} day" + ("" if d == 1 else "s"),
)

st.caption(
    "**How to read this.** Each day ahead is predicted by its own model, using "
    "only information available on the forecast date. The grey line is the "
    "*persistence* baseline — simply assuming tomorrow looks like today. A "
    "forecast is only worth having if it beats that line, so both are shown."
)

_bt = load_backtest_results()
_city_bt = (_bt or {}).get("cities", {}).get(forecast_city, {})

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_forecast(city, days, synthetic):
    """Fitting one model per horizon is not free -- cache per (city, days)."""
    return forecast_with_baseline(city, days=days, use_synthetic=synthetic)


with st.spinner(
    f"Fitting {horizon_days} per-horizon models for {forecast_city} "
    "(cached afterwards)..."
):
    result = _cached_forecast(forecast_city, horizon_days, use_synthetic)

if result is None:
    st.warning(
        f"Not enough history for {forecast_city} to forecast responsibly. "
        "Cities with large monitoring gaps are excluded rather than given a "
        "confident-looking number."
    )
else:
    fdf = result["forecast"]

    fig, ax = plt.subplots(figsize=(14, 5))
    actual = df[df["city"] == forecast_city].dropna(subset=["aqi"]).tail(60)
    ax.plot(actual["date"], actual["aqi"], "o-", color="steelblue",
            alpha=0.7, label="Observed AQI", markersize=3)
    ax.plot(pd.to_datetime(fdf["date"]), fdf["prediction"], "s-",
            color="crimson", label="Model forecast", markersize=4, linewidth=2)
    ax.plot(pd.to_datetime(fdf["date"]), fdf["persistence"], "--",
            color="gray", label="Persistence baseline", linewidth=2)
    ax.axvline(x=actual["date"].max(), color="gray", linestyle=":", alpha=0.6)
    ax.set_title(f"{forecast_city}: {horizon_days}-day AQI forecast vs baseline")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)
    st.caption(
        f"Blue is what actually happened over the last 60 days. Red is the "
        f"forecast for the next {horizon_days} days; grey is the baseline. "
        "Where red sits close to grey, the model is adding little."
    )

    st.subheader("Measured accuracy, by how far ahead")
    if _city_bt.get("horizons"):
        rows = []
        for h_str, m in sorted(_city_bt["horizons"].items(), key=lambda kv: int(kv[0])):
            skill = m.get("skill_vs_persistence_pct")
            rows.append({
                "Days ahead": int(h_str),
                "Model error (MAPE)": f"{m['model_mape']:.1f}%" if m.get("model_mape") else "—",
                "Persistence error": f"{m['persistence_mape']:.1f}%" if m.get("persistence_mape") else "—",
                "Model vs baseline": (f"{skill:+.1f}%" if skill is not None else "—"),
                "Better method": "Model" if (skill or 0) > 0 else "Persistence",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Days ahead"),
                     use_container_width=True)
        st.caption(
            "MAPE is average error as a percentage of the true AQI — lower is "
            "better. These come from a rolling backtest on held-out periods "
            f"({_city_bt.get('n_folds', 0)} folds, hundreds of predictions per "
            "horizon), not from data the models trained on. Where the last "
            "column says *Persistence*, the machine learning model is not "
            "earning its place at that range."
        )
    else:
        st.info("No backtest on record for this city. Run `python scripts/backtest.py`.")

    st.subheader("Forecast values")
    show = fdf.copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(
        show.rename(columns={
            "date": "Date", "horizon": "Days ahead",
            "prediction": "Model AQI", "persistence": "Baseline AQI",
        }).set_index("Date"),
        use_container_width=True,
    )

    for _, row in fdf.iterrows():
        pred = row["prediction"]
        day = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        if pred > 200:
            st.warning(f"⚠️ **{day}**: AQI {pred:.0f} — poor air quality expected")
        elif pred > 100:
            st.info(f"ℹ️ **{day}**: AQI {pred:.0f} — moderate air quality expected")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "India Air Quality Analytics | CPCB daily 2015-01–2020-07 (26 cities) "
    "· hourly 2019-01–2020-07 (6 cities) | "
    f"Freshness: {freshness['last_ingested_at'] or 'N/A'}"
    "</div>", unsafe_allow_html=True
)
