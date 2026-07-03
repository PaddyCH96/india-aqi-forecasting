"""Reusable EDA analysis functions for the India AQI dataset.

All functions work with DataFrames returned by lib.db query functions.
"""

import pandas as pd
import numpy as np


def city_ranking(df: pd.DataFrame, metric: str = "aqi") -> pd.DataFrame:
    """Rank cities by mean pollutant value."""
    ranking = (
        df.groupby("city")[metric]
        .agg(["mean", "std", "min", "max", "count"])
        .round(1)
        .sort_values("mean", ascending=False)
    )
    ranking.columns = ["Mean", "Std", "Min", "Max", "Obs"]
    ranking.index.name = "city"
    return ranking.reset_index()


def aqi_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AQI bucket distribution per city."""
    buckets = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    dist = pd.crosstab(df["city"], df["aqi_bucket"])
    for b in buckets:
        if b not in dist.columns:
            dist[b] = 0
    dist = dist[buckets]
    dist["Total"] = dist.sum(axis=1)
    for b in buckets:
        dist[f"{b}_pct"] = (dist[b] / dist["Total"] * 100).round(1)
    return dist.reset_index()


def monthly_trends(df: pd.DataFrame, pollutant: str = "aqi") -> pd.DataFrame:
    """Compute monthly averages for a pollutant across cities."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly = (
        df.groupby(["city", "year", "month"])[pollutant]
        .mean()
        .round(1)
        .reset_index()
    )
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str)
    )
    return monthly


def weekday_analysis(df: pd.DataFrame, pollutant: str = "aqi") -> pd.DataFrame:
    """Compute average pollutant by day of week per city."""
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.day_name()
    dow = (
        df.groupby(["city", "dow", "day_name"])[pollutant]
        .mean()
        .round(1)
        .reset_index()
        .sort_values(["city", "dow"])
    )
    return dow


def missing_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing percentage per city per pollutant."""
    pollutants = ["pm2_5", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3", "aqi"]
    available = [p for p in pollutants if p in df.columns]
    miss = df.groupby("city")[available].apply(lambda x: (x.isna().mean() * 100).round(1))
    return miss.reset_index()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation of pollutants."""
    pollutants = ["pm2_5", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3", "aqi"]
    available = [p for p in pollutants if p in df.columns]
    return df[available].corr().round(3)


def trend_decomposition(df: pd.DataFrame, pollutant: str = "aqi") -> pd.DataFrame:
    """Simple trend decomposition using moving averages."""
    df = df.copy().sort_values("date")
    df["trend"] = df[pollutant].rolling(90, min_periods=30, center=True).mean()
    df["seasonal_30"] = df[pollutant] - df["trend"]
    df["seasonal_avg"] = df.groupby(df["date"].dt.dayofyear)[pollutant].transform("mean")
    return df


def summer_winter_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare summer (Apr-Jun) vs winter (Nov-Feb) averages per city."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    conditions = [
        (df["month"].isin([4, 5, 6])),
        (df["month"].isin([11, 12, 1, 2])),
    ]
    choices = ["summer", "winter"]
    df["season"] = np.select(conditions, choices, default="other")
    seasonal = df[df["season"].isin(["summer", "winter"])].groupby(
        ["city", "season"]
    )["aqi"].agg(["mean", "std", "count"]).round(1).reset_index()
    pivot = seasonal.pivot(index="city", columns="season", values="mean")
    pivot["winter_summer_ratio"] = (pivot["winter"] / pivot["summer"]).round(2)
    return pivot.reset_index()


def top_n_cities(df: pd.DataFrame, n: int = 10, metric: str = "aqi") -> list:
    """Return top N cities by mean pollutant value."""
    return list(df.groupby("city")[metric].mean().sort_values(ascending=False).head(n).index)


def year_over_year(df: pd.DataFrame, pollutant: str = "aqi") -> pd.DataFrame:
    """Compute yearly averages per city."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    return (
        df.groupby(["city", "year"])[pollutant]
        .agg(["mean", "std", "min", "max", "count"])
        .round(1)
        .reset_index()
    )


def worst_best_cities(df: pd.DataFrame, year: int = 2020) -> dict:
    """Return best and worst city by mean AQI for a given year."""
    df = df.copy()
    y = df[df["date"].dt.year == year].groupby("city")["aqi"].mean()
    if y.empty:
        return {"worst": None, "best": None, "worst_aqi": None, "best_aqi": None}
    return {
        "worst": y.idxmax(),
        "best": y.idxmin(),
        "worst_aqi": round(y.max(), 1),
        "best_aqi": round(y.min(), 1),
    }


def hourly_diurnal(df: pd.DataFrame, pollutant: str = "aqi") -> pd.DataFrame:
    """Compute average hourly pattern per city."""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    return (
        df.groupby(["city", "hour"])[pollutant]
        .mean()
        .round(1)
        .reset_index()
    )


def pollutant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Overall pollutant summary statistics."""
    pollutants = ["pm2_5", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3", "aqi"]
    available = [p for p in pollutants if p in df.columns]
    stats = df[available].describe().round(1)
    stats.loc["missing_pct"] = (df[available].isna().mean() * 100).round(1)
    return stats
