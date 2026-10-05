"""Tests the whole application logic (data entry, currency check, prediction) with a
stand-in model. Mirrors what happens at the viva: warn, add today's data, then predict."""
from datetime import date

import pytest

import config
import openmeteo
import service
from errors import ApiError
from helpers import FakePredictor, make_state, new_record, rain


def _code(func, *args):
    with pytest.raises(ApiError) as info:
        func(*args)
    return info.value.code


# ------------------------------------------------------------ the viva scenario
def test_prediction_is_blocked_until_todays_data_is_entered():
    state = make_state(predictor=FakePredictor())
    info = service.get_status(state)
    assert info["ready"] is False
    assert "Today's data (2026-01-01) has not been entered" in info["message"]
    assert _code(service.predict, state, {"rainfall_forecast": rain(5)}) == "DATA_NOT_CURRENT"


def test_adding_todays_data_unlocks_prediction_for_tomorrow():
    predictor = FakePredictor(delta=1.25)
    state = make_state(predictor=predictor)
    saved = service.add_observation(state, new_record(day="2026-01-01", storage=106.0, value=4.0))
    assert saved["status"]["ready"] is True
    assert saved["status"]["prediction_target"] == "2026-01-02"

    result = service.predict(state, {"target_date": "2026-01-02", "rainfall_forecast": rain(7)})
    assert result["target_date"] == "2026-01-02"
    assert result["storage_lag1_mcm"] == 106.0
    assert result["predicted_delta_storage_mcm"] == 1.25
    assert result["predicted_storage_mcm"] == 107.25


def test_the_newly_entered_day_is_part_of_the_models_history():
    predictor = FakePredictor()
    state = make_state(predictor=predictor)
    service.add_observation(state, new_record(day="2026-01-01", storage=106.0, value=4.0))
    service.predict(state, {"rainfall_forecast": rain(7)})
    sequence, static = predictor.calls[0]
    assert list(sequence[-1]) == [4.0] * 10            # today's real rainfall is the last day
    assert static[config.STATIC_ORDER.index("storage_lag1")] == 106.0
    assert list(static[:10]) == [7.0] * 10              # the forecast is only in the static input


def test_adding_data_does_not_touch_the_model():
    predictor = FakePredictor()
    state = make_state(predictor=predictor)
    service.add_observation(state, new_record())
    assert predictor.calls == []  # no prediction, and no retraining hook exists


# ---------------------------------------------------------- prediction errors
def test_predicting_a_different_date_is_rejected():
    state = make_state(predictor=FakePredictor(), last_date="2025-12-31", system_date="2025-12-31")
    assert _code(service.predict, state, {"target_date": "2026-01-04", "rainfall_forecast": rain(5)}) == "WRONG_TARGET_DATE"


def test_missing_model_gives_a_clear_error():
    state = make_state(predictor=None, system_date="2025-12-31")
    assert _code(service.predict, state, {"rainfall_forecast": rain(5)}) == "MODEL_UNAVAILABLE"


def test_forecast_must_include_every_catchment():
    state = make_state(predictor=FakePredictor(), system_date="2025-12-31")
    forecast = rain(5)
    del forecast[config.CATCHMENTS[0]]
    assert _code(service.predict, state, {"rainfall_forecast": forecast}) == "MISSING_CATCHMENT"


def test_extreme_rainfall_produces_a_warning():
    state = make_state(predictor=FakePredictor(), system_date="2025-12-31")
    result = service.predict(state, {"rainfall_forecast": rain(60)})
    assert "higher than every day" in result["warning"]


def test_ordinary_rainfall_produces_no_warning():
    state = make_state(predictor=FakePredictor(), system_date="2025-12-31")
    assert service.predict(state, {"rainfall_forecast": rain(1)})["warning"] is None


def test_warning_levels():
    assert service.extreme_rain_warning(30, 40, 60) is None
    assert "top 1%" in service.extreme_rain_warning(50, 40, 60)
    assert "higher than every day" in service.extreme_rain_warning(70, 40, 60)


# ------------------------------------------------------------------ data entry
def test_entering_a_day_out_of_order_is_rejected():
    state = make_state(system_date="2026-01-05")
    assert _code(service.add_observation, state, new_record(day="2026-01-03")) == "DATE_OUT_OF_ORDER"


def test_entering_tomorrow_is_rejected():
    # Real-deployment behaviour specifically: outside demo mode, a day that has not
    # really happened yet is rejected. In demo mode this check is intentionally
    # relaxed (see test_demo_chain.py), since there is no real "future" to violate.
    state = make_state(demo=False, system_date="2025-12-31")
    assert _code(service.add_observation, state, new_record(day="2026-01-01")) == "FUTURE_DATE"


def test_large_change_requires_confirmation_and_is_then_saved():
    state = make_state()
    assert _code(service.add_observation, state, new_record(storage=150.0)) == "NEEDS_CONFIRMATION"
    saved = service.add_observation(state, new_record(storage=150.0, confirm_large_change=True))
    assert saved["saved"] is True


def test_failed_validation_does_not_change_the_database():
    state = make_state()
    before = service.get_status(state)["record_count"]
    _code(service.add_observation, state, new_record(storage=-1))
    assert service.get_status(state)["record_count"] == before


def test_the_latest_manual_record_can_be_corrected():
    state = make_state()
    service.add_observation(state, new_record(storage=106.0))
    service.correct_latest(state, new_record(storage=105.5))
    assert service.list_observations(state, 1)[0]["storage_mcm"] == 105.5


def test_imported_history_cannot_be_corrected():
    state = make_state()
    assert _code(service.correct_latest, state, new_record(day="2025-12-31")) == "NOT_EDITABLE"


def test_recent_records_include_total_rainfall_and_source():
    state = make_state()
    service.add_observation(state, new_record(value=2.0))
    newest = service.list_observations(state, 3)[0]
    assert newest["date"] == "2026-01-01"
    assert newest["rain_total"] == 20.0
    assert newest["source"] == "manual"


# ------------------------------------------------------------------------ PIN
def test_entry_pin_is_enforced_when_configured():
    state = make_state(pin="1234")
    assert _code(service.add_observation, state, new_record(), None) == "PIN_REQUIRED"
    assert _code(service.add_observation, state, new_record(), "0000") == "PIN_REQUIRED"
    assert service.add_observation(state, new_record(), "1234")["saved"] is True


def test_prediction_does_not_need_a_pin():
    state = make_state(pin="1234", predictor=FakePredictor(), system_date="2025-12-31")
    assert service.predict(state, {"rainfall_forecast": rain(2)})["target_date"] == "2026-01-01"


# ------------------------------------------------------------------ Open-Meteo
def _good_response(url, timeout=10):
    return {"daily": {"time": ["2026-01-01"], "precipitation_sum": [3.44]}}


def test_fetch_rainfall_fills_every_catchment():
    state = make_state(fetch_fn=_good_response)
    result = service.fetch_rainfall(state, "2026-01-01")
    assert result["rainfall"] == {c: 3.4 for c in config.CATCHMENTS}
    assert result["unavailable"] == {}
    assert result["source"] == "open_meteo"


def test_fetch_rainfall_reports_a_catchment_without_coordinates():
    state = make_state(fetch_fn=_good_response, complete_coords=False)
    result = service.fetch_rainfall(state, "2026-01-01")
    assert config.CATCHMENTS[0] in result["unavailable"]
    assert config.CATCHMENTS[0] not in result["rainfall"]


def test_fetch_rainfall_fails_clearly_when_offline():
    state = make_state()  # default fetch function raises OSError
    assert _code(service.fetch_rainfall, state, "2026-01-01") == "OPEN_METEO_UNAVAILABLE"


def test_old_dates_use_the_archive_endpoint():
    url = openmeteo.build_url(7.1, 80.7, date(2026, 1, 1), date(2026, 9, 28), "precipitation_sum", "Asia/Colombo")
    assert url.startswith(openmeteo.ARCHIVE_URL)
    assert "start_date=2026-01-01" in url


def test_recent_dates_use_the_forecast_endpoint_with_past_days():
    url = openmeteo.build_url(7.1, 80.7, date(2026, 9, 25), date(2026, 9, 28), "precipitation_sum", "Asia/Colombo")
    assert url.startswith(openmeteo.FORECAST_URL)
    assert "past_days=3" in url and "forecast_days=1" in url


def test_future_dates_use_the_forecast_endpoint():
    url = openmeteo.build_url(7.1, 80.7, date(2026, 9, 30), date(2026, 9, 28), "precipitation_sum", "Asia/Colombo")
    assert "past_days=0" in url and "forecast_days=3" in url


def test_dates_beyond_the_forecast_horizon_are_rejected():
    with pytest.raises(openmeteo.OpenMeteoError):
        openmeteo.build_url(7.1, 80.7, date(2026, 10, 30), date(2026, 9, 28), "precipitation_sum", "Asia/Colombo")


def test_a_missing_value_from_open_meteo_is_an_error():
    data = {"daily": {"time": ["2026-01-01"], "precipitation_sum": [None]}}
    with pytest.raises(openmeteo.OpenMeteoError):
        openmeteo.extract_value(data, date(2026, 1, 1), "precipitation_sum")
