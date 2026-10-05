"""Builds the model inputs for one prediction, keeping antecedent and target-day
information strictly separate (the leakage-prevention design of the thesis).

The sequence input uses only days BEFORE the target day. The static input carries
the target day's forecast rainfall, the reservoir's known state, and calendar features.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import CATCHMENTS, DOW_COLUMNS, STATIC_ORDER, WINDOW


class FeatureError(Exception):
    """Raised when the history cannot support a valid prediction."""


def build_model_inputs(window, target_date, forecast):
    """Return (sequence[30,10], static[25]) for the given target date.

    window: DataFrame of exactly WINDOW consecutive days ending the day before target_date.
    forecast: dict of forecast rainfall (mm) for the target day, keyed by catchment.
    """
    if len(window) != WINDOW:
        raise FeatureError(f"Need {WINDOW} days of history, found {len(window)}.")
    dates = pd.to_datetime(window["date"]).reset_index(drop=True)
    if (dates.diff().dropna() != pd.Timedelta(days=1)).any():
        raise FeatureError("The 30-day history has missing days.")
    if dates.iloc[-1].date() != target_date - timedelta(days=1):
        raise FeatureError("The history must end on the day before the target date.")
    missing = [c for c in CATCHMENTS if c not in forecast]
    if missing:
        raise FeatureError("Forecast rainfall is missing for: " + ", ".join(missing))

    sequence = window[CATCHMENTS].to_numpy(dtype=np.float32)  # (30, 10), days before target only

    lag1 = float(window["storage_mcm"].iloc[-1])   # storage on day d-1
    lag2 = float(window["storage_mcm"].iloc[-2])   # storage on day d-2
    month = target_date.month
    day_of_year = target_date.timetuple().tm_yday

    features = {name: float(forecast[name]) for name in CATCHMENTS}
    features.update({
        "rain_total_target_day": float(sum(forecast[name] for name in CATCHMENTS)),
        "storage_lag1": lag1,
        "storage_lag2": lag2,
        "storage_trend": lag1 - lag2,
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "doy_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
        "data_era": 1.0,  # all operational dates are after the 2017 resolution change
    })
    weekday = target_date.weekday()  # Monday = 0 ... Sunday = 6
    for index, column in enumerate(DOW_COLUMNS):
        features[column] = 1.0 if weekday == index else 0.0

    static = np.array([features[name] for name in STATIC_ORDER], dtype=np.float32)
    return sequence, static
