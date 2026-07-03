import pandas as pd
import pytest
from lib.models import create_model, train_and_forecast, train_and_validate
from lib.config import PROPHET_PARAMS, FORECAST_PERIODS, FORECAST_YEARS


class TestCreateModel:
    def test_returns_prophet_instance(self):
        from prophet import Prophet
        model = create_model()
        assert isinstance(model, Prophet)

    def test_applies_default_params(self):
        model = create_model()
        assert model.yearly_seasonality == PROPHET_PARAMS["yearly_seasonality"]
        assert model.changepoint_prior_scale == PROPHET_PARAMS["changepoint_prior_scale"]

    def test_overrides_default_params(self):
        model = create_model(changepoint_prior_scale=0.1)
        assert model.changepoint_prior_scale == 0.1
        assert model.yearly_seasonality == PROPHET_PARAMS["yearly_seasonality"]


class TestTrainAndForecast:
    @pytest.fixture
    def sample_df(self):
        n = 100
        return pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=n, freq="D"),
            "y": range(100, 100 + n),
        })

    def test_returns_model_and_forecast(self, sample_df):
        model, forecast = train_and_forecast(sample_df, periods=30)
        from prophet import Prophet
        assert isinstance(model, Prophet)
        assert isinstance(forecast, pd.DataFrame)
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns
        assert "yhat_lower" in forecast.columns
        assert "yhat_upper" in forecast.columns

    def test_forecast_has_correct_periods(self, sample_df):
        _, forecast = train_and_forecast(sample_df, periods=30)
        future = forecast[forecast["ds"] > sample_df["ds"].max()]
        assert len(future) == 30

    def test_default_periods_equals_config(self, sample_df):
        _, forecast = train_and_forecast(sample_df)
        future = forecast[forecast["ds"] > sample_df["ds"].max()]
        assert len(future) == FORECAST_PERIODS

    def test_forecast_values_are_reasonable(self, sample_df):
        _, forecast = train_and_forecast(sample_df, periods=10)
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            assert forecast[col].notna().all()
        assert (forecast["yhat_upper"] >= forecast["yhat_lower"]).all()

    def test_overrides_passed_to_model(self, sample_df):
        model, _ = train_and_forecast(sample_df, periods=10, changepoint_prior_scale=0.2)
        assert model.changepoint_prior_scale == 0.2


class TestTrainAndValidate:
    @pytest.fixture
    def split_data(self):
        n = 200
        df = pd.DataFrame({
            "ds": pd.date_range("2023-01-01", periods=n, freq="D"),
            "y": range(100, 100 + n),
        })
        train = df.iloc[:180]
        test = df.iloc[180:]
        return train, test

    def test_returns_model_forecast_results(self, split_data):
        train, test = split_data
        model, forecast, results = train_and_validate(train, test)
        from prophet import Prophet
        assert isinstance(model, Prophet)
        assert isinstance(forecast, pd.DataFrame)
        assert isinstance(results, pd.DataFrame)

    def test_results_match_test_period(self, split_data):
        train, test = split_data
        _, _, results = train_and_validate(train, test)
        assert len(results) == len(test)
        assert "y" in results.columns
        assert "yhat" in results.columns

    def test_forecast_covers_test_dates(self, split_data):
        train, test = split_data
        _, forecast, results = train_and_validate(train, test)
        test_dates = set(test["ds"].dt.date)
        forecast_dates = set(forecast["ds"].dt.date)
        assert test_dates.issubset(forecast_dates)
