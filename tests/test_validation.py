from datetime import date

import pytest

import config
import validation
from errors import ApiError
from helpers import new_record, rain

LATEST = {"date": "2025-12-31", "storage_mcm": 100.0, "rainfall": rain(), "source": "historical_import"}
TODAY = date(2026, 1, 1)


def _code(payload, latest=LATEST, system_date=TODAY):
    with pytest.raises(ApiError) as info:
        validation.validate_new_observation(payload, latest, system_date)
    return info.value.code


def test_accepts_the_next_day_with_valid_values():
    obs_date, storage, rainfall, source = validation.validate_new_observation(
        new_record(storage=101.5), LATEST, TODAY)
    assert obs_date == date(2026, 1, 1)
    assert storage == 101.5
    assert set(rainfall) == set(config.CATCHMENTS)
    assert source == "manual"


def test_rejects_a_date_that_skips_a_day():
    assert _code(new_record(day="2026-01-02"), system_date=date(2026, 1, 5)) == "DATE_OUT_OF_ORDER"


def test_rejects_a_date_that_is_already_recorded():
    assert _code(new_record(day="2025-12-31")) == "DATE_ALREADY_RECORDED"


def test_rejects_a_date_in_the_future():
    assert _code(new_record(day="2026-01-01"), system_date=date(2025, 12, 31)) == "FUTURE_DATE"


def test_rejects_an_invalid_date_string():
    assert _code(new_record(day="not-a-date")) == "INVALID_DATE"


def test_rejects_negative_rainfall():
    payload = new_record()
    payload["rainfall"][config.CATCHMENTS[0]] = -1
    assert _code(payload) == "NEGATIVE_RAINFALL"


def test_rejects_implausibly_high_rainfall():
    payload = new_record()
    payload["rainfall"][config.CATCHMENTS[0]] = config.RAIN_MAX_MM + 1
    assert _code(payload) == "RAINFALL_TOO_HIGH"


def test_rejects_missing_and_unknown_catchments():
    missing = new_record()
    del missing["rainfall"][config.CATCHMENTS[2]]
    assert _code(missing) == "MISSING_CATCHMENT"
    unknown = new_record()
    unknown["rainfall"]["atlantis_rain_mm"] = 1.0
    assert _code(unknown) == "UNKNOWN_CATCHMENT"


def test_rejects_storage_outside_the_plausible_range():
    assert _code(new_record(storage=-5)) == "STORAGE_OUT_OF_RANGE"
    assert _code(new_record(storage=config.STORAGE_MAX_MCM + 1)) == "STORAGE_OUT_OF_RANGE"


def test_rejects_non_numeric_storage():
    assert _code(new_record(storage="abc")) == "INVALID_NUMBER"


def test_a_large_storage_jump_needs_confirmation_and_then_passes():
    assert _code(new_record(storage=140.0)) == "NEEDS_CONFIRMATION"
    obs_date, storage, _, _ = validation.validate_new_observation(
        new_record(storage=140.0, confirm_large_change=True), LATEST, TODAY)
    assert storage == 140.0


def test_rejects_an_unknown_source():
    assert _code(new_record(source="telepathy")) == "INVALID_SOURCE"


def test_historical_records_cannot_be_corrected():
    with pytest.raises(ApiError) as info:
        validation.validate_correction(new_record(day="2025-12-31"), LATEST, None)
    assert info.value.code == "NOT_EDITABLE"


def test_only_the_latest_record_can_be_corrected():
    latest = dict(LATEST, source="manual")
    with pytest.raises(ApiError) as info:
        validation.validate_correction(new_record(day="2025-12-30"), latest, None)
    assert info.value.code == "NOT_LATEST_RECORD"


def test_a_manual_latest_record_can_be_corrected():
    latest = dict(LATEST, source="manual")
    obs_date, storage, _, _ = validation.validate_correction(
        new_record(day="2025-12-31", storage=101.0), latest, {"date": "2025-12-30", "storage_mcm": 100.5})
    assert storage == 101.0
