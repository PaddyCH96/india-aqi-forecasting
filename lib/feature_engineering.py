"""Feature engineering for time-series air quality prediction.

All functions return DataFrames with engineered features added.
Designed for daily-frequency data from load_city_pollutants().
"""

import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged values of specified columns.

    Args:
        df: Must be sorted by date per city.
        columns: Pollutant columns to lag. Default: ['aqi', 'pm2_5', 'no2', 'co', 'o3'].
        lags: Lag periods. Default: [1, 2, 3, 7].

    Returns:
        DataFrame with added '{col}_lag{n}' columns.
    """
    if columns is None:
        columns = ["aqi", "pm2_5", "no2", "co", "o3"]
    if lags is None:
        lags = [1, 2, 3, 7]
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        for lag in lags:
            if lag == 1:
                result[f"{col}_lag1"] = result.groupby("city")[col].shift(1)
            else:
                result[f"{col}_lag{lag}"] = result.groupby("city")[col].shift(lag)
    return result


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling window statistics.

    Args:
        df: Must be sorted by date per city.
        columns: Pollutant columns. Default: ['aqi', 'pm2_5'].
        windows: Window sizes in days. Default: [3, 7, 30].

    Returns:
        DataFrame with added '{col}_roll{window}_mean' and '_std' columns.
    """
    if columns is None:
        columns = ["aqi", "pm2_5"]
    if windows is None:
        windows = [3, 7, 30]
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            continue
        for w in windows:
            rolled = result.groupby("city")[col].rolling(w, min_periods=1)
            result[f"{col}_roll{w}_mean"] = rolled.mean().reset_index(level=0, drop=True)
            result[f"{col}_roll{w}_std"] = rolled.std().fillna(0).reset_index(level=0, drop=True)
            result[f"{col}_roll{w}_max"] = rolled.max().reset_index(level=0, drop=True)
    return result


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical temporal features.

    Adds: hour, day_of_week, month, quarter, is_weekend,
          month_sin, month_cos, dow_sin, dow_cos.
    """
    result = df.copy()
    col = "datetime" if "datetime" in result.columns else "date"
    dt = pd.to_datetime(result[col])
    result["hour"] = dt.dt.hour
    result["day_of_week"] = dt.dt.dayofweek
    result["month"] = dt.dt.month
    result["quarter"] = dt.dt.quarter
    result["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    return result


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pollutant interaction terms.

    Adds: pm25_pm10_ratio, no2_co_product, primary_secondary.
    """
    result = df.copy()
    if "pm2_5" in result.columns and "pm10" in result.columns:
        result["pm25_pm10_ratio"] = (
            result["pm2_5"] / result["pm10"].replace(0, np.nan)
        )
    if "no2" in result.columns and "co" in result.columns:
        result["no2_co_product"] = result["no2"] * result["co"]
    if "pm2_5" in result.columns and "o3" in result.columns:
        result["primary_secondary"] = result["pm2_5"] * result["o3"]
    if "no2" in result.columns and "so2" in result.columns:
        result["no2_so2_sum"] = result["no2"] + result["so2"]
    return result


def add_city_normalization(
    df: pd.DataFrame,
    target: str = "aqi",
) -> pd.DataFrame:
    """Add city-relative features.

    Adds: {target}_city_zscore, {target}_city_pctile.
    """
    result = df.copy()
    means = result.groupby("city")[target].transform("mean")
    stds = result.groupby("city")[target].transform("std").replace(0, 1)
    result[f"{target}_city_zscore"] = ((result[target] - means) / stds).round(3)
    return result


def build_feature_pipeline(
    df: pd.DataFrame,
    target: str = "aqi",
    include_lags: bool = True,
    include_rolling: bool = True,
    include_seasonal: bool = True,
    include_interactions: bool = True,
    include_normalization: bool = True,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Args:
        df: DataFrame from load_city_pollutants() sorted by date.
        target: Target variable name.

    Returns:
        DataFrame with all engineered features. Rows with NaN
        from lag/rolling transformations are preserved (model handles them).
    """
    result = df.sort_values(["city", "date"]).copy()
    if include_seasonal:
        result = add_seasonal_features(result)
    if include_lags:
        result = add_lag_features(result, columns=[target, "pm2_5", "no2", "co"])
    if include_rolling:
        result = add_rolling_features(result, columns=[target, "pm2_5"])
    if include_interactions:
        result = add_interaction_features(result)
    if include_normalization:
        result = add_city_normalization(result, target=target)
    return result


def feature_column_groups(df: pd.DataFrame) -> dict:
    """Return dictionary of feature column names by category."""
    cols = set(df.columns)
    return {
        "temporal": [c for c in ["month_sin", "month_cos", "dow_sin", "dow_cos",
                                  "day_of_week", "month", "quarter", "is_weekend"]
                     if c in cols],
        "lags": sorted([c for c in cols if "_lag" in c]),
        "rolling": sorted([c for c in cols if "_roll" in c]),
        "interactions": sorted([c for c in ["pm25_pm10_ratio", "no2_co_product",
                                             "primary_secondary", "no2_so2_sum"]
                                if c in cols]),
        "normalization": sorted([c for c in cols if "zscore" in c]),
    }
