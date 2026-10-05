"""Decides whether the stored data is current enough to predict tomorrow."""
from datetime import date, timedelta


def compute_status(latest, system_date):
    """Describe the state of the data relative to the system date.

    Predicting tomorrow requires today's real storage and rainfall, so the data is
    'ready' only when the latest record is dated on or after the system date.
    """
    base = {"system_date": system_date.isoformat()}
    if latest is None:
        base.update({
            "last_recorded_date": None, "prediction_target": None, "next_record_date": None,
            "can_record": False, "ready": False, "missing_days": None, "level": "error",
            "message": "No records were found in the database, so predictions are not possible.",
        })
        return base

    last = date.fromisoformat(latest["date"])
    next_day = last + timedelta(days=1)
    base.update({
        "last_recorded_date": last.isoformat(),
        "prediction_target": next_day.isoformat(),
        "next_record_date": next_day.isoformat(),
        "can_record": next_day <= system_date,
    })

    if last >= system_date:
        base.update({
            "ready": True, "missing_days": 0, "level": "ok",
            "message": f"Data is current through {last}. Ready to predict {next_day}.",
        })
        return base

    missing = (system_date - last).days
    if missing == 1:
        message = (f"Today's data ({system_date}) has not been entered. "
                   "Add today's storage and rainfall before predicting tomorrow.")
    else:
        message = (f"Data has not been entered for {missing} days ({next_day} to {system_date}). "
                   "Add these records, in order, before predicting tomorrow.")
    base.update({"ready": False, "missing_days": missing, "level": "warn", "message": message})
    return base
