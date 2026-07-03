#!/usr/bin/env python3
"""Unified India AQI Analytics Dashboard.

Pages:
1. Executive Summary — National snapshot, KPI cards, city ranking
2. Historical Trends — Multi-city trends, seasonal analysis
3. Pollutant Drill-Down — Per-pollutant analysis, correlations
4. City Deep-Dive — Single city analysis, diurnal patterns
5. Data Quality — Missing data, coverage warnings
"""

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
def load_pollutant_data(cities):
    all_data = []
    for city in cities:
        df = load_city_pollutants(engine, city, use_synthetic=use_synthetic)
        if not df.empty:
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


@st.cache_data(ttl=300)
def load_hourly_data(cities):
    from sqlalchemy import text as sa_text
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
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


with st.spinner("Loading data..."):
    df = load_pollutant_data(selected_cities)

if df.empty:
    st.warning("No data loaded. Try selecting different cities or enabling synthetic data.")
    st.stop()

df_hourly = load_hourly_data(selected_cities)

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

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    dist = aqi_distribution(df)
    plot_aqi_category_bars(ax, dist, top_n=min(26, len(selected_cities)))
    st.pyplot(fig)

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

with tab_e:
    corr_df = correlation_matrix(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_correlation_heatmap(ax, corr_df)
    st.pyplot(fig)

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
        st.info("Hourly data available for selected cities. Run `ingest_hourly.py` if not loaded.")

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

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "India Air Quality Analytics | CPCB Data 2015–2020 | "
    f"Freshness: {freshness['last_ingested_at'] or 'N/A'}"
    "</div>", unsafe_allow_html=True
)
