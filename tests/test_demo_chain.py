"""Tests the demo clock and the 'accept prediction as real, advance one day' action
that lets a multi-day chained demo run with a single repeated button click."""
import pytest

import config
import service
from demo_clock import DemoClock
from errors import ApiError
from helpers import FakePredictor, make_state, rain


def _code(func, *args):
    with pytest.raises(ApiError) as info:
        func(*args)
    return info.value.code


# --------------------------------------------------------------------- DemoClock
def test_demo_clock_advances_by_exactly_one_day():
    from datetime import date
    clock = DemoClock(date(2026, 1, 1))
    assert clock.advance() == date(2026, 1, 2)
    assert clock.today() == date(2026, 1, 2)


def test_demo_clock_set_jumps_directly():
    from datetime import date
    clock = DemoClock(date(2026, 1, 1))
    clock.set(date(2026, 3, 1))
    assert clock.today() == date(2026, 3, 1)


# ---------------------------------------------------------------- guard rails
def test_blocked_outside_demo_mode():
    state = make_state(demo=False)
    assert _code(service.accept_prediction_as_real, state,
                 {"storage_mcm": 100.0, "rainfall": rain(1)}) == "NOT_DEMO_MODE"


def test_blocked_when_todays_data_is_not_yet_entered():
    state = make_state()  # 2026-01-01 has not been entered
    assert _code(service.accept_prediction_as_real, state,
                 {"storage_mcm": 100.0, "rainfall": rain(1)}) == "DATA_NOT_CURRENT"


def test_invalid_values_are_rejected_like_any_other_entry():
    state = make_state()
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 106.0, "rainfall": rain(4)})
    assert _code(service.accept_prediction_as_real, state,
                 {"storage_mcm": -5, "rainfall": rain(1)}) == "STORAGE_OUT_OF_RANGE"


# --------------------------------------------------------------- the chain itself
def test_accepting_a_prediction_advances_the_clock_and_records_it():
    state = make_state(predictor=FakePredictor(delta=2.0))
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 106.0, "rainfall": rain(4)})
    result = service.predict(state, {"rainfall_forecast": rain(7)})
    assert result["target_date"] == "2026-01-02"

    accepted = service.accept_prediction_as_real(
        state, {"storage_mcm": result["predicted_storage_mcm"], "rainfall": rain(7)})
    assert accepted["date"] == "2026-01-02"
    status = accepted["status"]
    assert status["last_recorded_date"] == "2026-01-02"
    assert status["system_date"] == "2026-01-02"          # the clock moved forward
    assert status["ready"] is True
    assert status["prediction_target"] == "2026-01-03"     # immediately ready for the next step


def test_three_consecutive_chained_predictions_advance_one_day_at_a_time():
    state = make_state(predictor=FakePredictor(delta=1.0))
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 100.0, "rainfall": rain(4)})

    expected_targets = ["2026-01-02", "2026-01-03", "2026-01-04"]
    for target in expected_targets:
        result = service.predict(state, {"rainfall_forecast": rain(5)})
        assert result["target_date"] == target
        service.accept_prediction_as_real(state, {"storage_mcm": result["predicted_storage_mcm"], "rainfall": rain(5)})

    status = service.get_status(state)
    assert status["last_recorded_date"] == "2026-01-04"
    assert status["prediction_target"] == "2026-01-05"


def test_the_chained_value_becomes_part_of_the_next_predictions_history():
    predictor = FakePredictor(delta=3.0)
    state = make_state(predictor=predictor)
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 100.0, "rainfall": rain(4)})

    first = service.predict(state, {"rainfall_forecast": rain(9)})
    service.accept_prediction_as_real(state, {"storage_mcm": first["predicted_storage_mcm"], "rainfall": rain(9)})
    service.predict(state, {"rainfall_forecast": rain(1)})

    sequence, static = predictor.calls[-1]
    assert list(sequence[-1]) == [9.0] * 10  # yesterday's forecast rainfall is now "real" history
    assert static[config.STATIC_ORDER.index("storage_lag1")] == first["predicted_storage_mcm"]


def test_chained_records_are_tagged_predicted_chain_not_manual():
    state = make_state(predictor=FakePredictor(delta=1.0))
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 100.0, "rainfall": rain(4)})
    result = service.predict(state, {"rainfall_forecast": rain(5)})
    service.accept_prediction_as_real(state, {"storage_mcm": result["predicted_storage_mcm"], "rainfall": rain(5)})
    record = service.list_observations(state, 1)[0]
    assert record["source"] == "predicted_chain"


def test_status_reports_how_many_recent_days_are_chained_not_real():
    state = make_state(predictor=FakePredictor(delta=1.0))
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 100.0, "rainfall": rain(4)})
    assert service.get_status(state)["chained_days_in_window"] == 0

    for _ in range(3):
        result = service.predict(state, {"rainfall_forecast": rain(5)})
        service.accept_prediction_as_real(state, {"storage_mcm": result["predicted_storage_mcm"], "rainfall": rain(5)})
    assert service.get_status(state)["chained_days_in_window"] == 3


def test_a_chained_record_can_still_be_corrected_like_a_manual_one():
    state = make_state(predictor=FakePredictor(delta=1.0))
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 100.0, "rainfall": rain(4)})
    result = service.predict(state, {"rainfall_forecast": rain(5)})
    service.accept_prediction_as_real(state, {"storage_mcm": result["predicted_storage_mcm"], "rainfall": rain(5)})

    service.correct_latest(state, {"date": "2026-01-02", "storage_mcm": 103.5, "rainfall": rain(5)})
    assert service.list_observations(state, 1)[0]["storage_mcm"] == 103.5


def test_manual_entry_still_works_for_the_day_after_a_chained_record():
    """Regression test: chaining must not leave manual entry of the following day
    blocked. This mirrors a real bug where the entry form disabled itself after a
    chain step, even though saving that next day was always valid at the service
    layer - the bug was in the frontend's display logic, not here, but this test
    guards the capability the frontend depends on."""
    state = make_state(predictor=FakePredictor(delta=1.0))
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 100.0, "rainfall": rain(4)})
    result = service.predict(state, {"rainfall_forecast": rain(5)})
    service.accept_prediction_as_real(state, {"storage_mcm": result["predicted_storage_mcm"], "rainfall": rain(5)})

    status = service.get_status(state)
    assert status["next_record_date"] == "2026-01-03"
    saved = service.add_observation(state, {"date": "2026-01-03", "storage_mcm": 108.0, "rainfall": rain(2)})
    assert saved["saved"] is True
    assert service.list_observations(state, 1)[0]["source"] == "manual"


def test_manual_entry_keeps_working_for_several_consecutive_days_after_chaining():
    """The scenario this bug was found in: chain once, then switch to typing in
    several real days manually, one after another, with no file editing."""
    state = make_state(predictor=FakePredictor(delta=1.0))
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 100.0, "rainfall": rain(4)})
    result = service.predict(state, {"rainfall_forecast": rain(5)})
    service.accept_prediction_as_real(state, {"storage_mcm": result["predicted_storage_mcm"], "rainfall": rain(5)})

    for day, storage in [("2026-01-03", 108.0), ("2026-01-04", 109.5), ("2026-01-05", 111.0)]:
        saved = service.add_observation(state, {"date": day, "storage_mcm": storage, "rainfall": rain(2)})
        assert saved["saved"] is True
        assert saved["status"]["next_record_date"] is not None

    assert service.get_status(state)["last_recorded_date"] == "2026-01-05"


def test_manual_entry_and_chaining_can_alternate_freely():
    state = make_state(predictor=FakePredictor(delta=1.0))
    service.add_observation(state, {"date": "2026-01-01", "storage_mcm": 100.0, "rainfall": rain(4)})
    service.add_observation(state, {"date": "2026-01-02", "storage_mcm": 101.0, "rainfall": rain(3)})
    result = service.predict(state, {"rainfall_forecast": rain(5)})
    service.accept_prediction_as_real(state, {"storage_mcm": result["predicted_storage_mcm"], "rainfall": rain(5)})
    saved = service.add_observation(state, {"date": "2026-01-04", "storage_mcm": 105.0, "rainfall": rain(1)})
    assert saved["saved"] is True


def test_real_deployment_still_rejects_a_genuinely_future_date():
    """The protection this fix must NOT remove: outside demo mode, a day that
    hasn't really happened yet is still rejected."""
    state = make_state(demo=False, system_date="2025-12-31")
    assert _code(service.add_observation, state,
                 {"date": "2026-01-01", "storage_mcm": 101.0, "rainfall": rain(3)}) == "FUTURE_DATE"


def test_status_flags_demo_chain_availability():
    assert service.get_status(make_state(predictor=FakePredictor()))["demo_chain_available"] is True
    assert service.get_status(make_state(demo=False))["demo_chain_available"] is False
