from datetime import date

from status_logic import compute_status
from helpers import rain

LATEST = {"date": "2025-12-31", "storage_mcm": 100.0, "rainfall": rain(), "source": "historical_import"}


def test_todays_data_missing_gives_the_exact_warning():
    info = compute_status(LATEST, date(2026, 1, 1))
    assert info["ready"] is False
    assert info["level"] == "warn"
    assert info["message"] == ("Today's data (2026-01-01) has not been entered. "
                               "Add today's storage and rainfall before predicting tomorrow.")
    assert info["next_record_date"] == "2026-01-01"
    assert info["can_record"] is True


def test_several_missing_days_are_counted():
    info = compute_status(LATEST, date(2026, 1, 4))
    assert info["ready"] is False
    assert info["missing_days"] == 4
    assert "4 days" in info["message"]


def test_data_current_through_today_is_ready_for_tomorrow():
    info = compute_status(LATEST, date(2025, 12, 31))
    assert info["ready"] is True
    assert info["prediction_target"] == "2026-01-01"
    assert info["can_record"] is False  # tomorrow cannot be recorded yet


def test_empty_database_is_an_error_state():
    info = compute_status(None, date(2026, 1, 1))
    assert info["ready"] is False
    assert info["level"] == "error"
