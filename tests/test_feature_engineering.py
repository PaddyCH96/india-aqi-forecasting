import pytest
import pandas as pd
import numpy as np
from lib.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_seasonal_features,
    add_interaction_features,
    add_city_normalization,
    build_feature_pipeline,
    feature_column_groups,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "city": ["Delhi"] * n,
        "date": dates,
        "aqi": np.random.randint(50, 300, n).astype(float),
        "pm2_5": np.random.randint(20, 200, n).astype(float),
        "pm10": np.random.randint(40, 400, n).astype(float),
        "no2": np.random.randint(10, 80, n).astype(float),
        "co": np.random.uniform(0.5, 5, n),
        "o3": np.random.randint(10, 100, n).astype(float),
        "so2": np.random.randint(5, 40, n).astype(float),
    }).sort_values(["city", "date"]).reset_index(drop=True)


class TestAddLagFeatures:
    def test_adds_lag_columns(self, sample_df):
        result = add_lag_features(sample_df)
        assert "aqi_lag1" in result.columns
        assert "aqi_lag7" in result.columns
        assert result["aqi_lag1"].iloc[0] != result["aqi_lag1"].iloc[0]

    def test_first_rows_are_nan(self, sample_df):
        result = add_lag_features(sample_df, lags=[1, 3])
        assert pd.isna(result["aqi_lag1"].iloc[0])
        assert pd.isna(result["aqi_lag3"].iloc[2])

    def test_custom_columns(self, sample_df):
        result = add_lag_features(sample_df, columns=["pm2_5"], lags=[1])
        assert "pm2_5_lag1" in result.columns
        assert "aqi_lag1" not in result.columns

    def test_unknown_column_skipped(self, sample_df):
        result = add_lag_features(sample_df, columns=["nonexistent"])
        assert "nonexistent_lag1" not in result.columns


class TestAddRollingFeatures:
    def test_adds_rolling_columns(self, sample_df):
        result = add_rolling_features(sample_df)
        assert "aqi_roll3_mean" in result.columns
        assert "aqi_roll7_mean" in result.columns

    def test_rolling_values_reasonable(self, sample_df):
        result = add_rolling_features(sample_df, columns=["aqi"], windows=[3])
        assert result["aqi_roll3_mean"].notna().sum() > 0


class TestAddSeasonalFeatures:
    def test_adds_seasonal_columns(self, sample_df):
        result = add_seasonal_features(sample_df)
        for col in ["month_sin", "month_cos", "dow_sin", "dow_cos",
                     "day_of_week", "month", "is_weekend"]:
            assert col in result.columns

    def test_seasonal_values_in_range(self, sample_df):
        result = add_seasonal_features(sample_df)
        assert result["month_sin"].between(-1, 1).all()
        assert result["is_weekend"].isin([0, 1]).all()


class TestAddInteractionFeatures:
    def test_adds_interaction_columns(self, sample_df):
        result = add_interaction_features(sample_df)
        assert "pm25_pm10_ratio" in result.columns

    def test_interaction_not_empty(self, sample_df):
        result = add_interaction_features(sample_df)
        assert result["pm25_pm10_ratio"].notna().sum() > 0


class TestAddCityNormalization:
    def test_adds_normalization_columns(self, sample_df):
        result = add_city_normalization(sample_df)
        assert "aqi_city_zscore" in result.columns

    def test_zscore_mean_near_zero(self, sample_df):
        result = add_city_normalization(sample_df)
        assert abs(result["aqi_city_zscore"].mean()) < 0.1


class TestBuildFeaturePipeline:
    def test_full_pipeline_runs(self, sample_df):
        result = build_feature_pipeline(sample_df)
        assert len(result) == len(sample_df)
        assert len(result.columns) > len(sample_df.columns)

    def test_preserves_date_column(self, sample_df):
        result = build_feature_pipeline(sample_df)
        assert "date" in result.columns

    def test_partial_pipeline(self, sample_df):
        result = build_feature_pipeline(sample_df, include_lags=False, include_rolling=False)
        assert not any("_lag" in c for c in result.columns)
        assert not any("_roll" in c for c in result.columns)
        assert any("_sin" in c for c in result.columns)


class TestFeatureColumnGroups:
    def test_returns_dict_of_lists(self, sample_df):
        df = build_feature_pipeline(sample_df)
        groups = feature_column_groups(df)
        assert isinstance(groups, dict)
        assert "lags" in groups
        assert "rolling" in groups
        assert "temporal" in groups
