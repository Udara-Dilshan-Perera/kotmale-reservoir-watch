"""Central configuration for the Kotmale Reservoir Watch application.

Most values can be overridden with environment variables (see README.md).
"""
import os
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- catchments
# Order matters: the sequence branch of the LSTM was trained with this column order.
CATCHMENTS = [
    "ambewela_rain_mm",
    "campion_estate_rain_mm",
    "hellboda_north_rain_mm",
    "holmwood_estate_rain_mm",
    "hope_estate_rain_mm",
    "nuwara_eliya_rain_mm",
    "oonoogaloya_estate_rain_mm",
    "sandrigham_rain_mm",
    "talawakelle_rain_mm",
    "watawala_rain_mm",
]

# ------------------------------------------------------------ model features
WINDOW = 30  # days of rainfall history fed to the LSTM sequence branch
DOW_COLUMNS = [
    "dow_monday", "dow_tuesday", "dow_wednesday",
    "dow_thursday", "dow_friday", "dow_saturday",
]  # Sunday is the reference category
STATIC_ORDER = (
    CATCHMENTS
    + ["rain_total_target_day", "storage_lag1", "storage_lag2", "storage_trend",
       "month_sin", "month_cos", "doy_sin", "doy_cos", "data_era"]
    + DOW_COLUMNS
)  # 25 static features, same order as training

# ------------------------------------------------------- data-entry limits
CAPACITY_MCM = 176.78            # gross capacity of Kotmale Reservoir
STORAGE_MIN_MCM = 0.0
STORAGE_MAX_MCM = 180.0          # capacity plus a small tolerance
RAIN_MAX_MM = 600.0              # plausible upper bound for one catchment, one day
LARGE_STORAGE_CHANGE_MCM = 20.0  # day-to-day change that needs explicit confirmation
TYPICAL_ABS_ERROR_MCM = 1.0      # mean absolute error, validation 0.93 / test 1.07

# ----------------------------------------------------------------- file paths
DB_PATH = Path(os.getenv("KOTMALE_DB", BASE_DIR / "kotmale.db"))
SEED_CSV_CANDIDATES = [BASE_DIR / "kotmale_clean.csv", BASE_DIR / "kotmale_features.csv"]
MODEL_PATH = BASE_DIR / "lstm_model.keras"
SEQ_SCALER_PATH = BASE_DIR / "seq_scaler.joblib"
STATIC_SCALER_PATH = BASE_DIR / "static_scaler.joblib"
Y_SCALER_PATH = BASE_DIR / "y_scaler.joblib"
COORDINATES_PATH = BASE_DIR / "catchment_coordinates.json"
STATIC_DIR = BASE_DIR / "static"

# ------------------------------------------------------------------- security
ENTRY_PIN = os.getenv("KOTMALE_ENTRY_PIN") or None  # optional PIN for changing records

# ----------------------------------------------------------------- Open-Meteo
OPEN_METEO_VARIABLE = os.getenv("OPEN_METEO_VARIABLE", "precipitation_sum")
OPEN_METEO_TIMEZONE = os.getenv("OPEN_METEO_TIMEZONE", "Asia/Colombo")


def get_system_date() -> date:
    """The date the application treats as 'today'.

    Normally the computer's date. For the viva demonstration it can be fixed with
    the KOTMALE_SYSTEM_DATE environment variable (for example 2026-01-01), because
    the historical dataset ends on 2025-12-31.
    """
    override = os.getenv("KOTMALE_SYSTEM_DATE")
    if override:
        try:
            return date.fromisoformat(override.strip())
        except ValueError:
            raise RuntimeError(
                "KOTMALE_SYSTEM_DATE must be a date in the format YYYY-MM-DD, got: " + override
            )
    return date.today()


def system_date_is_overridden() -> bool:
    return bool(os.getenv("KOTMALE_SYSTEM_DATE"))
