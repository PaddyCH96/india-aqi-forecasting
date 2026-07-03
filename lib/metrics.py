import numpy as np
import pandas as pd


def calc_mape(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def calc_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calc_mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def evaluate_forecast(
    results: pd.DataFrame,
) -> dict:
    mape = calc_mape(results["y"], results["yhat"])
    rmse = calc_rmse(results["y"], results["yhat"])
    mae = calc_mae(results["y"], results["yhat"])
    return {"mape": mape, "rmse": rmse, "mae": mae}


def classify_model_quality(mape: float) -> tuple[str, str]:
    if mape < 15:
        return "Excellent", "green"
    elif mape < 20:
        return "Good", "blue"
    elif mape < 30:
        return "Moderate", "orange"
    else:
        return "Poor", "red"
