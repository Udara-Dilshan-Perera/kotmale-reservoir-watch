import os
import tempfile
from datetime import date, timedelta

import pandas as pd
import pytest

import config
import database
from helpers import make_state, rain


def _csv(path, days=40, storage_col="storage_mcm", rename_oonoogaloya=False, skip_day=None):
    end = date(2025, 12, 31)
    rows = []
    for i in range(days):
        if skip_day is not None and i == skip_day:
            continue
        d = end - timedelta(days=days - 1 - i)
        row = {"date": d.isoformat(), storage_col: 100 + i * 0.1}
        row.update({c: float(i % 5) for c in config.CATCHMENTS})
        rows.append(row)
    df = pd.DataFrame(rows)
    if rename_oonoogaloya:
        df = df.rename(columns={"oonoogaloya_estate_rain_mm": "oonogaloya_estate_rain_mm"})
    df.to_csv(path, index=False)


def _fresh_db():
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    database.init_db(path)
    return path


def test_seeding_imports_every_row_as_historical():
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "clean.csv")
    _csv(csv_path)
    db = _fresh_db()
    with database.connect(db) as conn:
        assert database.seed_from_csv(conn, csv_path) == 40
        latest = database.get_latest(conn)
    assert latest["date"] == "2025-12-31"
    assert latest["source"] == "historical_import"


def test_seeding_accepts_the_misspelled_catchment_column():
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "clean.csv")
    _csv(csv_path, rename_oonoogaloya=True)
    with database.connect(_fresh_db()) as conn:
        assert database.seed_from_csv(conn, csv_path) == 40


def test_seeding_accepts_the_features_file_layout():
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "features.csv")
    _csv(csv_path, storage_col="storage_actual")
    with database.connect(_fresh_db()) as conn:
        assert database.seed_from_csv(conn, csv_path) == 40


def test_seeding_rejects_a_dataset_with_a_missing_day():
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "gap.csv")
    _csv(csv_path, skip_day=10)
    with database.connect(_fresh_db()) as conn:
        with pytest.raises(database.DataError):
            database.seed_from_csv(conn, csv_path)


def test_seed_if_empty_does_not_import_twice():
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "clean.csv")
    _csv(csv_path)
    with database.connect(_fresh_db()) as conn:
        assert database.seed_if_empty(conn, [csv_path]) == 40
        assert database.seed_if_empty(conn, [csv_path]) == 0
        assert database.count(conn) == 40


def test_window_never_includes_rows_after_the_end_date():
    state = make_state(days=60)
    with database.connect(state.db_path) as conn:
        database.insert_observation(conn, "2026-01-01", 200.0, rain(9), "manual")
        window = database.get_window_df(conn, "2025-12-31", 30)
    assert len(window) == 30
    assert window["date"].iloc[-1] == pd.Timestamp("2025-12-31")
    assert window["storage_mcm"].max() < 200.0


def test_window_is_ordered_oldest_first():
    state = make_state(days=60)
    with database.connect(state.db_path) as conn:
        window = database.get_window_df(conn, "2025-12-31", 30)
    assert window["date"].is_monotonic_increasing


def test_rain_total_stats_returns_percentile_and_maximum():
    state = make_state(days=60)
    with database.connect(state.db_path) as conn:
        p99, maximum = database.rain_total_stats(conn)
    assert 0 < p99 <= maximum


def test_update_changes_only_the_target_row():
    state = make_state(days=5)
    with database.connect(state.db_path) as conn:
        before = database.get_recent(conn, 5)
        database.update_observation(conn, "2025-12-31", 150.0, rain(1), "manual")
        after = database.get_recent(conn, 5)
    assert after[0]["storage_mcm"] == 150.0
    assert after[0]["source"] == "manual"
    assert [r["storage_mcm"] for r in after[1:]] == [r["storage_mcm"] for r in before[1:]]
