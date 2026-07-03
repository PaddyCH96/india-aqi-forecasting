from prophet import Prophet
import pandas as pd

from lib.config import PROPHET_PARAMS, FORECAST_PERIODS


def create_model(**overrides) -> Prophet:
    params = {**PROPHET_PARAMS, **overrides}
    return Prophet(**params)


def train_and_forecast(
    df: pd.DataFrame,
    periods: int = FORECAST_PERIODS,
    freq: str = "D",
    **overrides,
) -> tuple[Prophet, pd.DataFrame]:
    model = create_model(**overrides)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast


def train_and_validate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    freq: str = "D",
    **overrides,
) -> tuple[Prophet, pd.DataFrame, pd.DataFrame]:
    model = create_model(**overrides)
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq=freq)
    forecast = model.predict(future)
    predictions = forecast[forecast["ds"].isin(test["ds"])][["ds", "yhat"]]
    results = test.merge(predictions, on="ds")
    return model, forecast, results
