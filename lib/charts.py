import pandas as pd
from matplotlib.axes import Axes

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
