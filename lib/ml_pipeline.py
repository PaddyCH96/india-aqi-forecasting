"""ML dataset builder and train/test split logic.

Provides reusable functions to build training datasets from the database
with proper time-based splits and per-city support.
"""

import pandas as pd

from lib.db import get_engine, load_city_pollutants
from lib.feature_engineering import build_feature_pipeline


def build_ml_dataset(
    city: str,
    target: str = "aqi",
    use_synthetic: bool = False,
    engine=None,
) -> pd.DataFrame:
    """Build a complete ML-ready dataset for a single city.

    Loads from DB, runs full feature engineering.
    Returns DataFrame with features + target, sorted by date.
    """
    if engine is None:
        engine = get_engine()
    df = load_city_pollutants(engine, city, use_synthetic=use_synthetic)
    if df.empty:
        return df
    df["city"] = city
    result = build_feature_pipeline(df, target=target)
    return result.sort_values("date").reset_index(drop=True)


def build_multi_city_dataset(
    cities: list[str],
    target: str = "aqi",
    use_synthetic: bool = False,
    engine=None,
) -> pd.DataFrame:
    """Build ML dataset for multiple cities, stacked vertically."""
    if engine is None:
        engine = get_engine()
    dfs = []
    for city in cities:
        df = build_ml_dataset(city, target=target, use_synthetic=use_synthetic, engine=engine)
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def time_based_split(
    df: pd.DataFrame,
    cutoff_date: str = "2019-01-01",
    target: str = "aqi",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-based train/test split.

    No random shuffle — preserves temporal order.
    Trains on data BEFORE cutoff_date, tests on AFTER (inclusive).

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    df = df.sort_values("date").reset_index(drop=True)
    train_mask = df["date"] < pd.Timestamp(cutoff_date)
    test_mask = df["date"] >= pd.Timestamp(cutoff_date)

    exclude_cols = {target, "date", "datetime", "city", "aqi_bucket",
                    "is_synthetic", "data_source", "ds"}
    feature_cols = [c for c in df.columns
                    if c not in exclude_cols and df[c].dtype in ("float64", "int64", "float32")]

    train = df[train_mask].copy()
    test = df[test_mask].copy()

    X_train = train[feature_cols]
    y_train = train[target]
    X_test = test[feature_cols]
    y_test = test[target]

    return X_train, X_test, y_train, y_test


def expanding_window_split(
    df: pd.DataFrame,
    min_train_days: int = 365,
    step_days: int = 180,
    test_days: int = 90,
    target: str = "aqi",
):
    """Generate expanding window train/test splits for time-series CV.

    Yields (X_train, X_test, y_train, y_test, fold_start, fold_end) tuples.
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].values
    max_date = dates.max()

    start = dates[0] + pd.Timedelta(days=min_train_days)
    while start + pd.Timedelta(days=test_days) <= max_date:
        train_end = start
        test_start = start
        test_end = start + pd.Timedelta(days=test_days)

        train_mask = df["date"] < pd.Timestamp(train_end)
        test_mask = (df["date"] >= pd.Timestamp(test_start)) & (df["date"] < pd.Timestamp(test_end))

        if test_mask.sum() < 10:
            start += pd.Timedelta(days=step_days)
            continue

        exclude_cols = {target, "date", "datetime", "city", "aqi_bucket",
                        "is_synthetic", "data_source", "ds"}
        feature_cols = [c for c in df.columns
                        if c not in exclude_cols and df[c].dtype in ("float64", "int64", "float32")]

        X_train = df.loc[train_mask.values, feature_cols]
        y_train = df.loc[train_mask.values, target]
        X_test = df.loc[test_mask.values, feature_cols]
        y_test = df.loc[test_mask.values, target]

        yield X_train, X_test, y_train, y_test, train_end, test_end
        start += pd.Timedelta(days=step_days)


def get_feature_names(df: pd.DataFrame, target: str = "aqi") -> list[str]:
    """Get list of feature column names (excludes target/id columns)."""
    exclude = {target, "date", "datetime", "city", "aqi_bucket",
               "is_synthetic", "data_source", "ds"}
    return [c for c in df.columns
            if c not in exclude and df[c].dtype in ("float64", "int64", "float32")]


def prepare_ml_data(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    max_nan_frac: float = 0.8,
):
    """Prepare ML data by dropping sparse features and imputing NaN.

    1. Drops rows where target is NaN.
    2. Drops features with >max_nan_frac NaN in training set.
    3. Imputes remaining NaN with training-set median.
    4. Returns (X_train, X_test, y_train, y_test) with no NaN.

    This is critical because cities like Mumbai have sparse pollutant
    data that would otherwise drop all rows.
    """
    train_valid_y = y_train.notna()
    test_valid_y = y_test.notna()
    X_train = X_train[train_valid_y]
    y_train = y_train[train_valid_y]
    X_test = X_test[test_valid_y]
    y_test = y_test[test_valid_y]

    train_nan_frac = X_train.isna().mean()
    keep_cols = train_nan_frac[train_nan_frac <= max_nan_frac].index.tolist()

    X_train_clean = X_train[keep_cols].copy()
    X_test_clean = X_test[keep_cols].copy()

    medians = X_train_clean.median()
    X_train_clean = X_train_clean.fillna(medians)
    X_test_clean = X_test_clean.fillna(medians)

    return X_train_clean, X_test_clean, y_train, y_test
