"""SQLite storage for daily reservoir observations.

One table, one row per day: reservoir storage plus rainfall for the ten catchments.
The historical dataset is imported once (source = 'historical_import'). New days
entered through the application are stored with source 'manual' or 'open_meteo'.
"""
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import CATCHMENTS


class DataError(Exception):
    """Raised when the source data is missing or unusable."""


def _schema_sql():
    rain_cols = ",\n    ".join(f"{c} REAL NOT NULL" for c in CATCHMENTS)
    return (
        "CREATE TABLE IF NOT EXISTS observations (\n"
        "    obs_date TEXT PRIMARY KEY,\n"
        "    storage_mcm REAL NOT NULL,\n"
        f"    {rain_cols},\n"
        "    source TEXT NOT NULL,\n"
        "    entered_at TEXT NOT NULL\n"
        ")"
    )


_COLUMNS = ["obs_date", "storage_mcm"] + CATCHMENTS + ["source", "entered_at"]
_INSERT_SQL = (
    "INSERT INTO observations (" + ", ".join(_COLUMNS) + ") VALUES ("
    + ", ".join("?" * len(_COLUMNS)) + ")"
)
_UPDATE_SQL = (
    "UPDATE observations SET storage_mcm = ?, "
    + ", ".join(f"{c} = ?" for c in CATCHMENTS)
    + ", source = ?, entered_at = ? WHERE obs_date = ?"
)


@contextmanager
def connect(db_path):
    """Open a connection; commit on success, roll back on any error."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path):
    with connect(db_path) as conn:
        conn.execute(_schema_sql())


def _now_iso():
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def _row_to_record(row):
    return {
        "date": row["obs_date"],
        "storage_mcm": row["storage_mcm"],
        "rainfall": {c: row[c] for c in CATCHMENTS},
        "source": row["source"],
        "entered_at": row["entered_at"],
    }


# ----------------------------------------------------------------------- reads
def count(conn):
    return conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]


def get_latest(conn):
    row = conn.execute("SELECT * FROM observations ORDER BY obs_date DESC LIMIT 1").fetchone()
    return _row_to_record(row) if row else None


def get_previous(conn, date_iso):
    """The most recent record strictly before date_iso."""
    row = conn.execute(
        "SELECT * FROM observations WHERE obs_date < ? ORDER BY obs_date DESC LIMIT 1",
        (date_iso,),
    ).fetchone()
    return _row_to_record(row) if row else None


def get_recent(conn, n):
    """The n most recent records, newest first."""
    rows = conn.execute(
        "SELECT * FROM observations ORDER BY obs_date DESC LIMIT ?", (n,)
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def get_window_df(conn, end_date_iso, n):
    """The n days ending on end_date_iso (inclusive), oldest first, as a DataFrame.

    Only rows dated on or before end_date_iso are ever returned, so nothing from
    the target day or later can leak into the model input.
    """
    cols = "obs_date, storage_mcm, " + ", ".join(CATCHMENTS)
    rows = conn.execute(
        f"SELECT {cols} FROM observations WHERE obs_date <= ? ORDER BY obs_date DESC LIMIT ?",
        (end_date_iso, n),
    ).fetchall()
    rows = list(reversed(rows))
    if not rows:
        return pd.DataFrame(columns=["date", "storage_mcm"] + CATCHMENTS)
    df = pd.DataFrame([dict(r) for r in rows]).rename(columns={"obs_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_storage_history(conn, days):
    rows = conn.execute(
        "SELECT obs_date, storage_mcm FROM observations ORDER BY obs_date DESC LIMIT ?",
        (days,),
    ).fetchall()
    return [{"date": r["obs_date"], "storage_mcm": r["storage_mcm"]} for r in reversed(rows)]


def count_source_in_window(conn, end_date_iso, n, source):
    """How many of the n most recent days (ending end_date_iso) came from `source`."""
    rows = conn.execute(
        "SELECT source FROM observations WHERE obs_date <= ? ORDER BY obs_date DESC LIMIT ?",
        (end_date_iso, n),
    ).fetchall()
    return sum(1 for r in rows if r["source"] == source)


def rain_total_stats(conn):
    """(99th percentile, maximum) of total daily rainfall across all catchments."""
    expr = " + ".join(CATCHMENTS)
    totals = np.array([r[0] for r in conn.execute(f"SELECT {expr} FROM observations")], dtype=float)
    if totals.size == 0:
        return None, None
    return float(np.percentile(totals, 99)), float(totals.max())


# ---------------------------------------------------------------------- writes
def insert_observation(conn, obs_date_iso, storage_mcm, rainfall, source):
    conn.execute(
        _INSERT_SQL,
        (obs_date_iso, storage_mcm, *[rainfall[c] for c in CATCHMENTS], source, _now_iso()),
    )


def update_observation(conn, obs_date_iso, storage_mcm, rainfall, source):
    conn.execute(
        _UPDATE_SQL,
        (storage_mcm, *[rainfall[c] for c in CATCHMENTS], source, _now_iso(), obs_date_iso),
    )


# ------------------------------------------------------------------ CSV import
def _collapse(name):
    """Lower-case and merge repeated letters, so 'oonogaloya' matches 'oonoogaloya'."""
    return re.sub(r"(.)\1+", r"\1", name.strip().lower())


def canonicalise_columns(columns):
    """Map CSV column names onto the canonical catchment names."""
    canon = {_collapse(c): c for c in CATCHMENTS}
    return {col: canon[_collapse(col)] for col in columns if _collapse(col) in canon}


def seed_from_csv(conn, csv_path):
    """Import the historical dataset. Accepts kotmale_clean.csv or kotmale_features.csv."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=canonicalise_columns(df.columns))

    if "storage_mcm" in df.columns:
        storage_col = "storage_mcm"
    elif "storage_actual" in df.columns:
        storage_col = "storage_actual"
    else:
        raise DataError(f"{Path(csv_path).name} has no storage column (storage_mcm or storage_actual).")
    if "date" not in df.columns:
        raise DataError(f"{Path(csv_path).name} has no 'date' column.")
    missing = [c for c in CATCHMENTS if c not in df.columns]
    if missing:
        raise DataError(f"{Path(csv_path).name} is missing rainfall columns: {', '.join(missing)}.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df[[storage_col] + CATCHMENTS].isnull().any().any():
        raise DataError(f"{Path(csv_path).name} contains empty values.")
    if not (df["date"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise DataError(f"{Path(csv_path).name} has missing or duplicated dates; the record must be one row per day.")

    now = _now_iso()
    rows = []
    for _, r in df.iterrows():
        rows.append((
            r["date"].strftime("%Y-%m-%d"), float(r[storage_col]),
            *[float(r[c]) for c in CATCHMENTS], "historical_import", now,
        ))
    conn.executemany(_INSERT_SQL, rows)
    return len(rows)


def seed_if_empty(conn, candidates):
    """Seed the database from the first available CSV, only if it has no records yet."""
    if count(conn) > 0:
        return 0
    for path in candidates:
        if Path(path).exists():
            return seed_from_csv(conn, path)
    raise DataError(
        "No historical dataset was found. Place kotmale_clean.csv (or kotmale_features.csv) "
        "in the application folder and restart."
    )
