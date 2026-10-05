"""Shared test helpers: a temporary database, a stand-in model, and sample data."""
import json
import os
import tempfile
from datetime import date, timedelta

import pandas as pd

import config
import database
from service import AppState


def rain(value=5.0):
    return {c: float(value) for c in config.CATCHMENTS}


class FakePredictor:
    """Stands in for the LSTM so the application logic can be tested without TensorFlow."""

    def __init__(self, delta=1.25):
        self.delta = delta
        self.calls = []

    def predict_delta(self, sequence, static):
        self.calls.append((sequence.copy(), static.copy()))
        return self.delta


def make_window(end="2025-12-31", n=30, start_storage=100.0):
    end_date = date.fromisoformat(end)
    rows = []
    for i in range(n):
        d = end_date - timedelta(days=n - 1 - i)
        row = {"date": pd.Timestamp(d), "storage_mcm": start_storage + i}
        row.update({c: float((i + j) % 7) for j, c in enumerate(config.CATCHMENTS)})
        rows.append(row)
    return pd.DataFrame(rows)


def write_coordinates(directory, complete=True):
    path = os.path.join(directory, "coords.json")
    data = {c: {"lat": 7.0 + i * 0.01, "lon": 80.7 + i * 0.01} for i, c in enumerate(config.CATCHMENTS)}
    if not complete:
        data[config.CATCHMENTS[0]] = {"lat": None, "lon": None}
    with open(path, "w") as handle:
        json.dump(data, handle)
    return path


def _no_network(url, timeout=10):
    raise OSError("network disabled in tests")


def make_state(last_date="2025-12-31", system_date="2026-01-01", days=60, predictor=None,
               pin=None, fetch_fn=None, complete_coords=True, demo=True):
    """An AppState backed by a fresh temporary database of `days` historical records."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    database.init_db(db_path)
    end = date.fromisoformat(last_date)
    with database.connect(db_path) as conn:
        for i in range(days):
            d = end - timedelta(days=days - 1 - i)
            rainfall = {c: float((i + j) % 7) for j, c in enumerate(config.CATCHMENTS)}
            database.insert_observation(conn, d.isoformat(), 100.0 + i * 0.1, rainfall, "historical_import")
    clock = None
    if demo:
        from demo_clock import DemoClock
        clock = DemoClock(date.fromisoformat(system_date))
    return AppState(
        db_path=db_path, predictor=predictor, entry_pin=pin,
        coordinates_path=write_coordinates(tmp, complete_coords),
        system_date_fn=(clock.today if clock else lambda: date.fromisoformat(system_date)),
        overridden_fn=lambda: True,
        real_today_fn=lambda: date(2026, 9, 28), fetch_fn=fetch_fn or _no_network,
        demo_clock=clock,
    )


def new_record(day="2026-01-01", storage=106.0, value=4.0, **extra):
    payload = {"date": day, "storage_mcm": storage, "rainfall": rain(value)}
    payload.update(extra)
    return payload
