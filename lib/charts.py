import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from lib.config import AQI_THRESHOLDS, MONTH_NAMES


def plot_history(
    ax: Axes,
    df: pd.DataFrame,
    city: str = "",
    show_covid: bool = False,
) -> Axes:
    df = df.copy()
    df["year"] = df["ds"].dt.year
    yearly_stats = df.groupby("year")["y"].agg("mean")

    ax.plot(df["ds"], df["y"], color="steelblue", alpha=0.3, linewidth=0.5, label="Daily AQI")

    years = yearly_stats.index
    yearly_dates = [pd.Timestamp(f"{y}-06-15") for y in years]
    ax.plot(yearly_dates, yearly_stats.values,
            color="crimson", linewidth=2.5, marker="o", markersize=7,
            label="Yearly Average", zorder=5)

    if show_covid:
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"),
                   alpha=0.1, color="gray", label="COVID Period")

    ax.axhline(y=AQI_THRESHOLDS["moderate"], color="orange", linestyle="--", alpha=0.5)
    ax.axhline(y=AQI_THRESHOLDS["poor"], color="red", linestyle="--", alpha=0.5)

    ax.set_title(f"{city} AQI History (2015-2024)" if city else "AQI History")
    ax.set_xlabel("Year")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_forecast(
    ax: Axes,
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    city: str = "",
) -> Axes:
    ax.plot(df["ds"], df["y"], color="steelblue", alpha=0.4, linewidth=1, label="Historical (2015-2024)")

    future_mask = forecast["ds"] > df["ds"].max()
    ax.plot(forecast["ds"][future_mask], forecast["yhat"][future_mask],
            color="crimson", linewidth=2.5, label="Forecast (2025-2030)")

    ax.fill_between(forecast["ds"][future_mask],
                    forecast["yhat_lower"][future_mask],
                    forecast["yhat_upper"][future_mask],
                    color="crimson", alpha=0.2, label="95% Confidence Interval")

    ax.axhline(y=AQI_THRESHOLDS["moderate"], color="orange", linestyle="--", alpha=0.7, label="Moderate (100)")
    ax.axhline(y=AQI_THRESHOLDS["poor"], color="red", linestyle="--", alpha=0.7, label="Poor (200)")
    ax.axvline(x=pd.Timestamp("2025-01-01"), color="green", linestyle=":", alpha=0.5, label="Forecast Start")

    ax.set_title(f"{city}: AQI Forecast 2025-2030" if city else "AQI Forecast", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("AQI")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df["ds"].min(), forecast["ds"].max())
    return ax


def plot_monthly_breakdown(
    ax: Axes,
    forecast: pd.DataFrame,
    year: int = 2030,
) -> Axes:
    pred = forecast[forecast["ds"].dt.year == year].copy()
    if pred.empty:
        ax.set_title(f"No forecast data for {year}")
        return ax

    pred["month"] = pred["ds"].dt.month
    monthly_avg = pred.groupby("month")["yhat"].mean()
    monthly_lower = pred.groupby("month")["yhat_lower"].mean()
    monthly_upper = pred.groupby("month")["yhat_upper"].mean()

    x_pos = range(12)
    ax.bar(x_pos, [monthly_avg.get(i + 1, 0) for i in range(12)],
           color="steelblue", alpha=0.7, edgecolor="black")

    ax.errorbar(x_pos, [monthly_avg.get(i + 1, 0) for i in range(12)],
                yerr=[[monthly_avg.get(i + 1, 0) - monthly_lower.get(i + 1, 0) for i in range(12)],
                      [monthly_upper.get(i + 1, 0) - monthly_avg.get(i + 1, 0) for i in range(12)]],
                fmt="none", color="black", alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_title(f"Predicted Monthly AQI for {year}")
    ax.set_ylabel("AQI")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=AQI_THRESHOLDS["moderate"], color="orange", linestyle="--", alpha=0.5)
    return ax


def plot_validation(
    ax: Axes,
    results: pd.DataFrame,
    mape: float,
    city: str = "",
) -> Axes:
    ax.plot(results["ds"], results["y"], color="steelblue", linewidth=2,
            label="Actual AQI (2023-2024)", marker="o", markersize=3)
    ax.plot(results["ds"], results["yhat"], color="crimson", linewidth=2,
            label=f"Predicted (MAPE: {mape:.1f}%)", alpha=0.8)

    ax.fill_between(results["ds"], results["yhat"] * 0.8, results["yhat"] * 1.2,
                    alpha=0.1, color="red", label="±20% Error Band")

    title = "Model Validation: Actual vs Predicted"
    if city:
        title += f" for {city}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_scatter_accuracy(
    ax: Axes,
    results: pd.DataFrame,
    mape: float,
) -> Axes:
    ax.scatter(results["y"], results["yhat"], alpha=0.5, edgecolors="black", linewidth=0.5)

    min_val = min(results["y"].min(), results["yhat"].min())
    max_val = max(results["y"].max(), results["yhat"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title(f"Prediction Accuracy (MAPE: {mape:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_comparison_barh(
    ax: Axes,
    comp_df: pd.DataFrame,
) -> Axes:
    colors = ["green" if x < 20 else "orange" if x < 30 else "red" for x in comp_df["MAPE (%)"]]
    ax.barh(comp_df["City"], comp_df["MAPE (%)"], color=colors)
    ax.set_xlabel("MAPE (%) - Lower is Better")
    ax.set_title("Forecast Accuracy Comparison Across Cities")
    ax.axvline(x=20, color="orange", linestyle="--", alpha=0.5, label="Good Threshold")
    ax.legend()
    return ax


def plot_multi_city_trends(
    ax: Axes,
    df: pd.DataFrame,
    cities: list[str],
    pollutant: str = "aqi",
    freq: str = "ME",
) -> Axes:
    """Plot smoothed trends for multiple cities on one axis."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(cities)))
    for city, color in zip(cities, colors):
        cdf = df[df["city"] == city].copy().sort_values("date")
        cdf = cdf.set_index("date")[pollutant].resample(freq).mean()
        ax.plot(cdf.index, cdf.values, color=color, linewidth=2, alpha=0.8, label=city)
    ax.set_title(f"{pollutant.upper()} Trends by City")
    ax.set_xlabel("Date")
    ax.set_ylabel(pollutant.upper())
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_missing_heatmap(
    ax: Axes,
    missing_df: pd.DataFrame,
    title: str = "Missing Data (%) by City and Pollutant",
) -> Axes:
    """Heatmap of missing percentages. missing_df has city column + pollutant columns."""
    plot_df = missing_df.set_index("city")
    im = ax.imshow(plot_df.values, aspect="auto", cmap="RdYlGn_r", norm=Normalize(0, 100))
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=8)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label="Missing %")
    for i in range(len(plot_df.index)):
        for j in range(len(plot_df.columns)):
            val = plot_df.values[i, j]
            color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)
    return ax


def plot_correlation_heatmap(
    ax: Axes,
    corr_df: pd.DataFrame,
    title: str = "Pollutant Correlation Matrix",
) -> Axes:
    """Heatmap of correlation matrix."""
    im = ax.imshow(corr_df.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index, fontsize=9)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = corr_df.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    return ax


def plot_pollutant_distribution(
    ax: Axes,
    df: pd.DataFrame,
    pollutant: str = "aqi",
    city: str | None = None,
    bins: int = 40,
) -> Axes:
    """Histogram + KDE for a pollutant."""
    data = df[df["city"] == city][pollutant].dropna() if city else df[pollutant].dropna()
    ax.hist(data, bins=bins, density=True, alpha=0.6, color="steelblue", edgecolor="black")
    kde = pd.Series(data).plot.kde(ax=ax, color="crimson", linewidth=2, legend=True)
    kde.set_label("Density")
    title = f"{pollutant.upper()} Distribution" + (f" - {city}" if city else "")
    ax.set_title(title)
    ax.set_xlabel(pollutant.upper())
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    return ax


def plot_aqi_category_bars(
    ax: Axes,
    dist_df: pd.DataFrame,
    top_n: int = 10,
) -> Axes:
    """Stacked horizontal bar of AQI bucket distribution per city."""
    buckets = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    colors = ["green", "lightgreen", "yellow", "orange", "red", "darkred"]
    plot_df = dist_df.head(top_n).set_index("city")
    bottom = None
    for bucket, color in zip(buckets, colors):
        if bucket in plot_df.columns:
            vals = plot_df[bucket]
            ax.barh(plot_df.index, vals, left=bottom, color=color, label=bucket, edgecolor="white", linewidth=0.5)
            bottom = vals if bottom is None else bottom + vals
    ax.set_xlabel("Days")
    ax.set_title("AQI Category Distribution by City")
    ax.legend(loc="lower right", fontsize=7)
    return ax


def plot_diurnal_pattern(
    ax: Axes,
    df: pd.DataFrame,
    city: str,
    pollutant: str = "aqi",
) -> Axes:
    """Average hourly pattern for a city."""
    cdf = df[df["city"] == city].copy()
    cdf["hour"] = cdf["datetime"].dt.hour
    hourly = cdf.groupby("hour")[pollutant].agg(["mean", "std"])
    ax.plot(hourly.index, hourly["mean"], color="steelblue", linewidth=2.5, marker="o")
    ax.fill_between(hourly.index,
                    hourly["mean"] - hourly["std"],
                    hourly["mean"] + hourly["std"],
                    alpha=0.2, color="steelblue")
    ax.set_title(f"Diurnal {pollutant.upper()} Pattern - {city}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(pollutant.upper())
    ax.set_xticks(range(0, 24, 3))
    ax.grid(True, alpha=0.3)
    return ax


def plot_city_ranking(
    ax: Axes,
    ranking_df: pd.DataFrame,
    metric: str = "Mean",
    title: str = "City Ranking by Mean AQI",
) -> Axes:
    """Horizontal bar chart of city ranking."""
    plot_df = ranking_df.sort_values(metric)
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(plot_df)))
    ax.barh(plot_df["city"], plot_df[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    return ax


def plot_seasonal_box(
    ax: Axes,
    df: pd.DataFrame,
    city: str,
    pollutant: str = "aqi",
) -> Axes:
    """Box plot by season for a city."""
    cdf = df[df["city"] == city].copy()
    cdf["month"] = cdf["date"].dt.month
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Monsoon", 10: "Monsoon", 11: "Monsoon"}
    cdf["season"] = cdf["month"].map(season_map)
    seasons = ["Winter", "Spring", "Summer", "Monsoon"]
    data = [cdf[cdf["season"] == s][pollutant].dropna().values for s in seasons]
    bp = ax.boxplot(data, labels=seasons, patch_artist=True)
    season_colors = ["lightblue", "lightgreen", "orange", "gray"]
    for patch, color in zip(bp["boxes"], season_colors):
        patch.set_facecolor(color)
    ax.set_title(f"Seasonal {pollutant.upper()} Distribution - {city}")
    ax.set_ylabel(pollutant.upper())
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_time_series(
    ax: Axes,
    df: pd.DataFrame,
    pollutant: str = "aqi",
    city: str | None = None,
    rolling_window: int = 0,
) -> Axes:
    """Generic time series plot with optional rolling average."""
    data = df[df["city"] == city] if city else df
    col = "datetime" if "datetime" in data.columns else "date"
    ax.plot(data[col], data[pollutant], alpha=0.3, linewidth=0.5, color="steelblue")
    if rolling_window > 0:
        roll = data.set_index(col)[pollutant].rolling(rolling_window, min_periods=1).mean()
        ax.plot(roll.index, roll.values, color="crimson", linewidth=2, label=f"{rolling_window}-day avg")
        ax.legend()
    title = f"{pollutant.upper()} Over Time" + (f" - {city}" if city else "")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(pollutant.upper())
    ax.grid(True, alpha=0.3)
    return ax
