"""Application logic. Independent of the web framework so it can be tested directly."""
import hmac
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Optional

import config
import database
import openmeteo
import status_logic
import validation
from errors import ApiError
from features import FeatureError, build_model_inputs


@dataclass
class AppState:
    db_path: str
    predictor: Optional[object] = None
    predictor_error: Optional[str] = None
    seed_error: Optional[str] = None
    entry_pin: Optional[str] = None
    coordinates_path: str = str(config.COORDINATES_PATH)
    system_date_fn: Callable[[], date] = config.get_system_date
    overridden_fn: Callable[[], bool] = config.system_date_is_overridden
    real_today_fn: Callable[[], date] = date.today
    fetch_fn: Callable = openmeteo.request_json
    demo_clock: Optional[object] = None  # a DemoClock, only set in demonstration mode


def build_default_state():
    """Create the database, import the historical data if needed, and load the model."""
    database.init_db(config.DB_PATH)
    seed_error = None
    with database.connect(config.DB_PATH) as conn:
        try:
            database.seed_if_empty(conn, config.SEED_CSV_CANDIDATES)
        except database.DataError as exc:
            seed_error = str(exc)

    from predictor import LSTMPredictor  # imported here so TensorFlow loads only in real use
    predictor = LSTMPredictor(config.MODEL_PATH, config.SEQ_SCALER_PATH,
                              config.STATIC_SCALER_PATH, config.Y_SCALER_PATH)
    predictor_error = None
    try:
        predictor.load()
    except Exception as exc:  # missing files, incompatible TensorFlow, and so on
        predictor, predictor_error = None, f"{type(exc).__name__}: {exc}"

    state = AppState(db_path=str(config.DB_PATH), predictor=predictor,
                     predictor_error=predictor_error, seed_error=seed_error,
                     entry_pin=config.ENTRY_PIN)

    # A demo clock is only created when the system date was explicitly overridden,
    # so the real calendar date is never advanceable outside a deliberate demo run.
    if config.system_date_is_overridden():
        from demo_clock import DemoClock
        state.demo_clock = DemoClock(config.get_system_date())
        state.system_date_fn = state.demo_clock.today
    return state


def require_pin(state, pin):
    if state.entry_pin and not hmac.compare_digest(str(pin or ""), str(state.entry_pin)):
        raise ApiError(401, "PIN_REQUIRED", "A valid entry PIN is required to change records.")


# --------------------------------------------------------------------- status
def get_status(state):
    system_date = state.system_date_fn()
    with database.connect(state.db_path) as conn:
        latest = database.get_latest(conn)
        record_count = database.count(conn)
        chained_days = (database.count_source_in_window(conn, latest["date"], config.WINDOW, "predicted_chain")
                       if latest else 0)
    info = status_logic.compute_status(latest, system_date)
    if latest is None and state.seed_error:
        info["message"] = state.seed_error
    info.update({
        "system_date_overridden": state.overridden_fn(),
        "demo_chain_available": state.demo_clock is not None,
        "record_count": record_count,
        "chained_days_in_window": chained_days,
        "model_loaded": state.predictor is not None,
        "model_error": state.predictor_error,
        "entry_pin_required": bool(state.entry_pin),
        "capacity_mcm": config.CAPACITY_MCM,
        "typical_error_mcm": config.TYPICAL_ABS_ERROR_MCM,
        "last_recorded_storage_mcm": latest["storage_mcm"] if latest else None,
    })
    if info["level"] == "ok" and not info["model_loaded"]:
        info["level"] = "warn"
    return info


# ---------------------------------------------------------------------- reads
def list_history(state, days):
    with database.connect(state.db_path) as conn:
        return database.get_storage_history(conn, days)


def list_observations(state, days):
    with database.connect(state.db_path) as conn:
        records = database.get_recent(conn, days)
    for record in records:
        record["rain_total"] = round(sum(record["rainfall"].values()), 1)
    return records


# --------------------------------------------------------------------- writes
def add_observation(state, payload, pin=None):
    require_pin(state, pin)
    system_date = state.system_date_fn()
    with database.connect(state.db_path) as conn:
        latest = database.get_latest(conn)
        obs_date, storage, rain, source = validation.validate_new_observation(
            payload, latest, system_date, enforce_future_check=(state.demo_clock is None))
        database.insert_observation(conn, obs_date.isoformat(), storage, rain, source)
    if state.demo_clock is not None:
        # Keep the demo's pretend-today tracking forward with whatever was just
        # recorded, exactly as accept_prediction_as_real does, so the day after
        # this one is immediately recordable too, with no file editing needed.
        state.demo_clock.set(obs_date)
    return {"saved": True, "date": obs_date.isoformat(), "status": get_status(state)}


def correct_latest(state, payload, pin=None):
    require_pin(state, pin)
    with database.connect(state.db_path) as conn:
        latest = database.get_latest(conn)
        previous = database.get_previous(conn, latest["date"]) if latest else None
        obs_date, storage, rain, source = validation.validate_correction(payload, latest, previous)
        database.update_observation(conn, obs_date.isoformat(), storage, rain, source)
    return {"saved": True, "date": obs_date.isoformat(), "status": get_status(state)}


# ----------------------------------------------------------------- prediction
def extreme_rain_warning(total, p99, hist_max):
    if p99 is None:
        return None
    if total > hist_max:
        return (f"The forecast total rainfall ({total:.0f} mm) is higher than every day in the "
                f"historical record (maximum {hist_max:.0f} mm). The model has not seen conditions "
                "like this, so treat the prediction with strong caution.")
    if total > p99:
        return (f"The forecast total rainfall ({total:.0f} mm) is in the top 1% of the historical "
                "record. Model accuracy is lower during extreme rainfall, so treat this prediction "
                "with caution.")
    return None


def predict(state, payload):
    info = get_status(state)
    if not info["ready"]:
        raise ApiError(409, "DATA_NOT_CURRENT", info["message"],
                       extra={"next_record_date": info["next_record_date"]})
    if state.predictor is None:
        raise ApiError(503, "MODEL_UNAVAILABLE",
                       "The prediction model is not loaded. " + (state.predictor_error or ""))

    target = date.fromisoformat(info["prediction_target"])
    requested = payload.get("target_date")
    if requested and requested != info["prediction_target"]:
        raise ApiError(400, "WRONG_TARGET_DATE",
                       f"Predictions can only be made for {info['prediction_target']}, the day after "
                       f"the latest recorded data ({info['last_recorded_date']}).")
    forecast = validation.validate_rainfall(payload.get("rainfall_forecast"), "Forecast rainfall")

    with database.connect(state.db_path) as conn:
        window = database.get_window_df(conn, info["last_recorded_date"], config.WINDOW)
        p99, hist_max = database.rain_total_stats(conn)
    try:
        sequence, static = build_model_inputs(window, target, forecast)
    except FeatureError as exc:
        raise ApiError(409, "HISTORY_INCOMPLETE", str(exc))

    delta = float(state.predictor.predict_delta(sequence, static))
    lag1 = info["last_recorded_storage_mcm"]
    return {
        "target_date": target.isoformat(),
        "predicted_delta_storage_mcm": round(delta, 3),
        "storage_lag1_mcm": round(lag1, 3),
        "predicted_storage_mcm": round(lag1 + delta, 3),
        "typical_error_mcm": config.TYPICAL_ABS_ERROR_MCM,
        "warning": extreme_rain_warning(sum(forecast.values()), p99, hist_max),
    }


# ------------------------------------------------------- demo chained prediction
def accept_prediction_as_real(state, payload):
    """Record a just-shown prediction as if it were tomorrow's real observation, and
    advance the demo clock to that day, so the next day can immediately be predicted.

    Demonstration mode only. The recorded storage and rainfall are the model's own
    output and the forecast used to produce it, not a genuine observation, so the
    record is tagged 'predicted_chain' and never mistaken for real data elsewhere
    in the interface.
    """
    if state.demo_clock is None:
        raise ApiError(
            409, "NOT_DEMO_MODE",
            "Chaining a prediction forward is only available in demonstration mode "
            "(the system date must be set with KOTMALE_SYSTEM_DATE).",
        )
    info = get_status(state)
    if not info["ready"]:
        raise ApiError(409, "DATA_NOT_CURRENT", info["message"])

    target_date = date.fromisoformat(info["prediction_target"])
    storage = validation.validate_storage(payload.get("storage_mcm"))
    rain = validation.validate_rainfall(payload.get("rainfall"))

    state.demo_clock.set(target_date)  # "today" becomes the day we are about to record
    with database.connect(state.db_path) as conn:
        database.insert_observation(conn, target_date.isoformat(), storage, rain, "predicted_chain")
    return {"saved": True, "date": target_date.isoformat(), "status": get_status(state)}


# ---------------------------------------------------------------- Open-Meteo
def fetch_rainfall(state, date_str=None):
    if date_str is None:
        date_str = get_status(state)["next_record_date"]
    target = validation.parse_iso_date(date_str)
    try:
        coordinates = openmeteo.load_coordinates(state.coordinates_path)
    except (OSError, ValueError) as exc:
        raise ApiError(500, "COORDINATES_UNREADABLE",
                       f"catchment_coordinates.json could not be read: {exc}")
    result = openmeteo.fetch_daily_rainfall(
        target, coordinates, config.OPEN_METEO_VARIABLE, config.OPEN_METEO_TIMEZONE,
        real_today=state.real_today_fn(), fetch=state.fetch_fn,
    )
    if not result["rainfall"]:
        raise ApiError(
            502, "OPEN_METEO_UNAVAILABLE",
            "Rainfall could not be retrieved from Open-Meteo. Check the internet connection and the "
            "coordinates in catchment_coordinates.json, or enter the values manually.",
            extra={"unavailable": result["unavailable"]},
        )
    return {
        "date": target.isoformat(),
        "rainfall": {k: round(v, 1) for k, v in result["rainfall"].items()},
        "unavailable": result["unavailable"],
        "variable": config.OPEN_METEO_VARIABLE,
        "source": "open_meteo",
    }
