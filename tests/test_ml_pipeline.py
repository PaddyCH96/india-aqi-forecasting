import pytest
import pandas as pd
import numpy as np
from lib.ml_pipeline import (
    time_based_split,
    get_feature_names,
    prepare_ml_data,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "city": ["Delhi"] * n,
        "date": dates,
        "aqi": np.random.randint(50, 300, n).astype(float),
        "pm2_5": np.random.randint(20, 200, n).astype(float),
        "month_sin": np.sin(2 * np.pi * dates.month / 12),
        "month_cos": np.cos(2 * np.pi * dates.month / 12),
        "aqi_lag1": np.random.randint(50, 300, n).astype(float),
        "aqi_roll7_mean": np.random.randint(50, 300, n).astype(float),
    })
    df.loc[0, "aqi_lag1"] = np.nan
    return df


class TestTimeBasedSplit:
    def test_splits_by_date(self, sample_df):
        X_train, X_test, y_train, y_test = time_based_split(
            sample_df, cutoff_date="2018-07-01"
        )
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) == len(sample_df)

    def test_no_leakage(self, sample_df):
        X_train, X_test, y_train, y_test = time_based_split(
            sample_df, cutoff_date="2019-01-01"
        )
        X_train_dates = sample_df.loc[X_train.index, "date"]
        X_test_dates = sample_df.loc[X_test.index, "date"]
        assert X_train_dates.max() < X_test_dates.min()

    def test_target_column_excluded(self, sample_df):
        X_train, X_test, y_train, y_test = time_based_split(
            sample_df, target="aqi"
        )
        assert "aqi" not in X_train.columns

    def test_city_column_excluded(self, sample_df):
        X_train, X_test, y_train, y_test = time_based_split(
            sample_df, target="aqi"
        )
        assert "city" not in X_train.columns


class TestPrepareMLData:
    def test_drops_nan_target_rows(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "aqi"] = np.nan
        df.loc[1, "aqi"] = np.nan
        X_train, X_test, y_train, y_test = time_based_split(df, target="aqi")
        X_tr, X_te, y_tr, y_te = prepare_ml_data(X_train, y_train, X_test, y_test)
        assert len(y_tr) <= len(X_train)
        assert not y_tr.isna().any()
        assert not y_te.isna().any()

    def test_imputes_feature_nan(self, sample_df):
        X_train, X_test, y_train, y_test = time_based_split(sample_df, target="aqi")
        X_tr, X_te, y_tr, y_te = prepare_ml_data(X_train, y_train, X_test, y_test)
        assert X_tr.isna().sum().sum() == 0
        assert X_te.isna().sum().sum() == 0

    def test_drops_sparse_features(self, sample_df):
        df = sample_df.copy()
        for i in range(400):
            df.loc[i, "pm2_5"] = np.nan
        df.loc[0, "aqi"] = np.nan
        X_train, X_test, y_train, y_test = time_based_split(df, target="aqi")
        X_tr, X_te, y_tr, y_te = prepare_ml_data(X_train, y_train, X_test, y_test, max_nan_frac=0.5)
        assert "pm2_5" not in X_tr.columns


class TestGetFeatureNames:
    def test_excludes_target(self, sample_df):
        features = get_feature_names(sample_df, target="aqi")
        assert "aqi" not in features

    def test_excludes_id_columns(self, sample_df):
        features = get_feature_names(sample_df)
        for col in ["city", "date", "aqi_bucket"]:
            assert col not in features

    def test_returns_numeric_only(self, sample_df):
        features = get_feature_names(sample_df)
        assert all(sample_df[c].dtype in ("float64", "int64", "float32") for c in features)
