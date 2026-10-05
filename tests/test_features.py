from datetime import date

import pytest

import config
from features import FeatureError, build_model_inputs
from helpers import make_window, rain


def _static_value(static, name):
    return float(static[config.STATIC_ORDER.index(name)])


def test_static_vector_has_25_features_in_training_order():
    _, static = build_model_inputs(make_window(), date(2026, 1, 1), rain(5))
    assert len(config.STATIC_ORDER) == 25
    assert static.shape == (25,)
    assert list(static[:10]) == [5.0] * 10  # forecast rainfall comes first


def test_sequence_is_the_30_days_before_the_target_day():
    window = make_window()
    sequence, _ = build_model_inputs(window, date(2026, 1, 1), rain(5))
    assert sequence.shape == (30, 10)
    assert (sequence == window[config.CATCHMENTS].to_numpy()).all()


def test_storage_lags_and_trend():
    window = make_window(start_storage=100.0)  # storage 100 ... 129
    _, static = build_model_inputs(window, date(2026, 1, 1), rain(5))
    assert _static_value(static, "storage_lag1") == 129.0
    assert _static_value(static, "storage_lag2") == 128.0
    assert _static_value(static, "storage_trend") == 1.0


def test_rain_total_is_the_sum_of_the_forecast():
    _, static = build_model_inputs(make_window(), date(2026, 1, 1), rain(3))
    assert _static_value(static, "rain_total_target_day") == 30.0


def test_friday_target_sets_only_the_friday_indicator():
    window = make_window(end="2026-01-01")
    _, static = build_model_inputs(window, date(2026, 1, 2), rain(1))  # 2 Jan 2026 is a Friday
    for column in config.DOW_COLUMNS:
        expected = 1.0 if column == "dow_friday" else 0.0
        assert _static_value(static, column) == expected


def test_sunday_target_leaves_all_indicators_at_zero():
    window = make_window(end="2026-01-03")
    _, static = build_model_inputs(window, date(2026, 1, 4), rain(1))  # 4 Jan 2026 is a Sunday
    assert all(_static_value(static, c) == 0.0 for c in config.DOW_COLUMNS)


def test_january_cyclical_month_encoding():
    _, static = build_model_inputs(make_window(), date(2026, 1, 1), rain(1))
    assert abs(_static_value(static, "month_sin") - 0.5) < 1e-6
    assert abs(_static_value(static, "month_cos") - 0.8660254) < 1e-6


def test_data_era_flag_is_one_for_operational_dates():
    _, static = build_model_inputs(make_window(), date(2026, 1, 1), rain(1))
    assert _static_value(static, "data_era") == 1.0


def test_target_day_forecast_never_enters_the_sequence():
    window = make_window()
    seq_a, _ = build_model_inputs(window, date(2026, 1, 1), rain(0))
    seq_b, _ = build_model_inputs(window, date(2026, 1, 1), rain(200))
    assert (seq_a == seq_b).all()


def test_changing_the_forecast_changes_only_forecast_features():
    window = make_window()
    _, static_a = build_model_inputs(window, date(2026, 1, 1), rain(1))
    _, static_b = build_model_inputs(window, date(2026, 1, 1), rain(9))
    forecast_features = set(config.CATCHMENTS) | {"rain_total_target_day"}
    for index, name in enumerate(config.STATIC_ORDER):
        if name in forecast_features:
            assert static_a[index] != static_b[index]
        else:
            assert static_a[index] == static_b[index]


def test_rejects_a_short_history():
    with pytest.raises(FeatureError):
        build_model_inputs(make_window().iloc[5:], date(2026, 1, 1), rain(1))


def test_rejects_a_history_with_a_missing_day():
    window = make_window(n=31).drop(index=10).reset_index(drop=True)  # 30 rows, one day missing
    with pytest.raises(FeatureError):
        build_model_inputs(window, date(2026, 1, 1), rain(1))


def test_rejects_a_history_that_does_not_end_the_day_before_the_target():
    with pytest.raises(FeatureError):
        build_model_inputs(make_window(end="2025-12-31"), date(2026, 1, 3), rain(1))


def test_rejects_a_forecast_with_a_missing_catchment():
    forecast = rain(1)
    del forecast[config.CATCHMENTS[3]]
    with pytest.raises(FeatureError):
        build_model_inputs(make_window(), date(2026, 1, 1), forecast)
