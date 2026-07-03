import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_actual_predicted():
    actual = pd.Series([100.0, 110.0, 120.0, 95.0, 105.0])
    predicted = pd.Series([95.0, 115.0, 118.0, 100.0, 102.0])
    return actual, predicted


@pytest.fixture
def sample_results():
    return pd.DataFrame({
        "ds": pd.date_range("2023-01-01", periods=5, freq="D"),
        "y": [100.0, 110.0, 120.0, 95.0, 105.0],
        "yhat": [95.0, 115.0, 118.0, 100.0, 102.0],
    })


@pytest.fixture
def sample_dates():
    return pd.date_range("2020-01-01", "2020-12-31", freq="D")
