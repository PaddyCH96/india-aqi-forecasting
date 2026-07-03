import pytest
import pandas as pd
import numpy as np
from lib.model_training import (
    train_moving_average,
    train_seasonal_naive,
    train_xgboost,
    train_random_forest,
)


@pytest.fixture
def train_test_data():
    np.random.seed(42)
    n = 365
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "aqi": np.random.randint(50, 300, n).astype(float),
    })
    train = df.iloc[:300]
    test = df.iloc[300:]
    return train, test


@pytest.fixture
def ml_data():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        "aqi_lag1": np.random.randn(n),
        "aqi_roll7_mean": np.random.randn(n),
        "month_sin": np.sin(2 * np.pi * np.arange(n) / 365),
        "pm2_5": np.random.randn(n) * 50 + 100,
    })
    y = pd.Series(X["pm2_5"] * 0.8 + np.random.randn(n) * 10 + 50)
    return X.iloc[:400], X.iloc[400:], y.iloc[:400], y.iloc[400:]


class TestTrainMovingAverage:
    def test_returns_metrics(self, train_test_data):
        train, test = train_test_data
        result = train_moving_average(train, test)
        assert "metrics" in result
        assert "mape" in result["metrics"]
        assert "rmse" in result["metrics"]

    def test_model_type(self, train_test_data):
        train, test = train_test_data
        result = train_moving_average(train, test)
        assert result["model_type"] == "Moving Average"

    def test_predictions_have_correct_length(self, train_test_data):
        train, test = train_test_data
        result = train_moving_average(train, test)
        assert len(result["predictions"]) == len(result["actuals"])


class TestTrainSeasonalNaive:
    def test_returns_metrics(self, train_test_data):
        train, test = train_test_data
        result = train_seasonal_naive(train, test, period=30)
        assert "metrics" in result
        assert "mape" in result["metrics"]

    def test_model_type(self, train_test_data):
        train, test = train_test_data
        result = train_seasonal_naive(train, test)
        assert result["model_type"] == "Seasonal Naive"


class TestTrainXGBoost:
    def test_returns_metrics(self, ml_data):
        X_tr, X_te, y_tr, y_te = ml_data
        result = train_xgboost(X_tr, y_tr, X_te, y_te)
        assert "metrics" in result
        assert "mape" in result["metrics"]

    def test_returns_model(self, ml_data):
        X_tr, X_te, y_tr, y_te = ml_data
        result = train_xgboost(X_tr, y_tr, X_te, y_te)
        assert result["model"] is not None

    def test_predictions_length_matches_test(self, ml_data):
        X_tr, X_te, y_tr, y_te = ml_data
        result = train_xgboost(X_tr, y_tr, X_te, y_te)
        assert len(result["predictions"]) == len(y_te)

    def test_feature_importance_returned(self, ml_data):
        X_tr, X_te, y_tr, y_te = ml_data
        result = train_xgboost(X_tr, y_tr, X_te, y_te)
        assert result["feature_importance"] is not None


class TestTrainRandomForest:
    def test_returns_metrics(self, ml_data):
        X_tr, X_te, y_tr, y_te = ml_data
        result = train_random_forest(X_tr, y_tr, X_te, y_te)
        assert "metrics" in result
        assert "mape" in result["metrics"]

    def test_returns_model(self, ml_data):
        X_tr, X_te, y_tr, y_te = ml_data
        result = train_random_forest(X_tr, y_tr, X_te, y_te)
        assert result["model"] is not None
