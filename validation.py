"""Validation rules for data entered through the application."""
import math
from datetime import date, timedelta

from config import (
    CATCHMENTS, STORAGE_MIN_MCM, STORAGE_MAX_MCM, RAIN_MAX_MM, LARGE_STORAGE_CHANGE_MCM,
)
from errors import ApiError

VALID_SOURCES = ("manual", "open_meteo")


def parse_iso_date(text):
    try:
        return date.fromisoformat(str(text))
    except (TypeError, ValueError):
        raise ApiError(400, "INVALID_DATE", f"'{text}' is not a valid date. Use the format YYYY-MM-DD.")


def _finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ApiError(400, "INVALID_NUMBER", f"{label} must be a number.")
    if not math.isfinite(number):
        raise ApiError(400, "INVALID_NUMBER", f"{label} must be a finite number.")
    return number


def validate_storage(value):
    storage = _finite(value, "Reservoir storage")
    if storage < STORAGE_MIN_MCM or storage > STORAGE_MAX_MCM:
        raise ApiError(
            400, "STORAGE_OUT_OF_RANGE",
            f"Reservoir storage ({storage:g} MCM) must be between {STORAGE_MIN_MCM:g} and "
            f"{STORAGE_MAX_MCM:g} MCM.",
        )
    return storage


def validate_rainfall(rain, label="Rainfall"):
    """Require exactly the ten catchments, each a non-negative, plausible number."""
    if not isinstance(rain, dict):
        raise ApiError(400, "INVALID_RAINFALL", f"{label} values are missing.")
    unknown = sorted(k for k in rain if k not in CATCHMENTS)
    if unknown:
        raise ApiError(400, "UNKNOWN_CATCHMENT", f"Unknown catchment name(s): {', '.join(unknown)}.")
    missing = [c for c in CATCHMENTS if c not in rain]
    if missing:
        raise ApiError(400, "MISSING_CATCHMENT", f"{label} is missing for: {', '.join(missing)}.")
    clean = {}
    for name in CATCHMENTS:
        value = _finite(rain[name], f"{label} for {name}")
        if value < 0:
            raise ApiError(400, "NEGATIVE_RAINFALL", f"{label} for {name} cannot be negative.")
        if value > RAIN_MAX_MM:
            raise ApiError(
                400, "RAINFALL_TOO_HIGH",
                f"{label} for {name} ({value:g} mm) is above the plausible daily maximum of {RAIN_MAX_MM:g} mm.",
            )
        clean[name] = value
    return clean


def _validate_source(payload):
    source = payload.get("source") or "manual"
    if source not in VALID_SOURCES:
        raise ApiError(400, "INVALID_SOURCE", f"Source must be one of: {', '.join(VALID_SOURCES)}.")
    return source


def _check_large_change(storage, reference, reference_date, confirmed):
    change = storage - reference
    if abs(change) > LARGE_STORAGE_CHANGE_MCM and not confirmed:
        raise ApiError(
            409, "NEEDS_CONFIRMATION",
            f"Storage changed by {change:+.1f} MCM since {reference_date}, which is larger than usual. "
            "Please confirm the value is correct.",
            extra={"change_mcm": round(change, 2)},
        )


def validate_new_observation(payload, latest, system_date, enforce_future_check=True):
    """Validate a new daily record. Returns (date, storage, rainfall, source).

    enforce_future_check is disabled in demonstration mode, where the system date is
    a user-controlled stand-in rather than a real clock, so the next sequential day
    is never genuinely "in the future" - it is only ever exactly the next expected
    day, already guaranteed by the two checks above this one.
    """
    if latest is None:
        raise ApiError(409, "NO_HISTORY", "The database has no records, so new days cannot be added.")
    obs_date = parse_iso_date(payload.get("date"))
    last_date = date.fromisoformat(latest["date"])
    expected = last_date + timedelta(days=1)

    if obs_date < expected:
        raise ApiError(
            409, "DATE_ALREADY_RECORDED",
            f"A record for {obs_date} already exists. Only the most recent record can be corrected.",
        )
    if obs_date > expected:
        raise ApiError(
            409, "DATE_OUT_OF_ORDER",
            f"The next date that can be recorded is {expected}. Records must be entered one day at a "
            "time, in order, so the 30-day history stays complete.",
        )
    if enforce_future_check and obs_date > system_date:
        raise ApiError(
            409, "FUTURE_DATE",
            f"{obs_date} is in the future. Records can only be entered up to the system date ({system_date}).",
        )

    storage = validate_storage(payload.get("storage_mcm"))
    rain = validate_rainfall(payload.get("rainfall"))
    source = _validate_source(payload)
    _check_large_change(storage, latest["storage_mcm"], last_date, payload.get("confirm_large_change"))
    return obs_date, storage, rain, source


def validate_correction(payload, latest, previous):
    """Validate a correction to the most recent record. Returns (date, storage, rainfall, source)."""
    if latest is None:
        raise ApiError(409, "NO_HISTORY", "The database has no records to correct.")
    obs_date = parse_iso_date(payload.get("date"))
    if obs_date.isoformat() != latest["date"]:
        raise ApiError(409, "NOT_LATEST_RECORD", "Only the most recent record can be corrected.")
    if latest["source"] == "historical_import":
        raise ApiError(
            409, "NOT_EDITABLE",
            "Records imported from the original historical dataset cannot be edited.",
        )

    storage = validate_storage(payload.get("storage_mcm"))
    rain = validate_rainfall(payload.get("rainfall"))
    source = _validate_source(payload)
    if previous is not None:
        _check_large_change(storage, previous["storage_mcm"], previous["date"], payload.get("confirm_large_change"))
    return obs_date, storage, rain, source
