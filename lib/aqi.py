import numpy as np
import pandas as pd


def pm25_to_aqi(pm25: float) -> float | None:
    if pd.isna(pm25):
        return None
    if pm25 <= 30:
        return pm25 * 1.67
    elif pm25 <= 60:
        return 50 + (pm25 - 30) * 1.67
    elif pm25 <= 90:
        return 100 + (pm25 - 60) * 3.33
    elif pm25 <= 120:
        return 200 + (pm25 - 90) * 3.33
    elif pm25 <= 250:
        return 300 + (pm25 - 120) * 0.77
    else:
        return min(400 + (pm25 - 250) * 0.4, 500)


def generate_synthetic_aqi(
    city: str,
    dates: pd.DatetimeIndex,
    base_aqi: float,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(dates)
    seasonal = np.sin(np.arange(n) * 2 * np.pi / 365.25 - np.pi / 2) * 40
    trend = np.linspace(-30, 20, n)
    weekly = np.array([0 if d.weekday() < 5 else -10 for d in dates])
    noise = rng.normal(0, 15, n)
    aqi = base_aqi + seasonal + trend + weekly + noise
    return np.clip(aqi, 30, 400)
