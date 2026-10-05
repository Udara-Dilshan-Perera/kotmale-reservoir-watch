"""Retrieves daily rainfall for the ten catchments from Open-Meteo.

Open-Meteo's archive endpoint lags real time by several days, so recent dates are
requested from the forecast endpoint (which can also return past days) instead.
"""
import json
import urllib.parse
import urllib.request
from datetime import date

from config import CATCHMENTS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 7      # dates older than this come from the archive endpoint
MAX_PAST_DAYS = 92        # limit of the forecast endpoint's past_days option
MAX_FORECAST_DAYS = 16    # limit of the forecast endpoint's forecast horizon


class OpenMeteoError(Exception):
    """Raised when a request cannot be built or the response cannot be used."""


def load_coordinates(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def request_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_url(lat, lon, target, real_today, variable, timezone):
    """Choose the right endpoint for the target date and build the request URL."""
    common = {"latitude": lat, "longitude": lon, "daily": variable, "timezone": timezone}
    age_days = (real_today - target).days
    if age_days > ARCHIVE_LAG_DAYS:
        params = dict(common, start_date=target.isoformat(), end_date=target.isoformat())
        return ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    if age_days > 0:
        if age_days > MAX_PAST_DAYS:
            raise OpenMeteoError("The date is too far in the past for the forecast endpoint.")
        params = dict(common, past_days=age_days, forecast_days=1)
    else:
        ahead = -age_days
        if ahead > MAX_FORECAST_DAYS:
            raise OpenMeteoError(f"Open-Meteo forecasts extend at most {MAX_FORECAST_DAYS} days ahead.")
        params = dict(common, past_days=0, forecast_days=ahead + 1)
    return FORECAST_URL + "?" + urllib.parse.urlencode(params)


def extract_value(data, target, variable):
    """Pull the target date's value out of an Open-Meteo daily response."""
    daily = data.get("daily") if isinstance(data, dict) else None
    if not daily or "time" not in daily or variable not in daily:
        raise OpenMeteoError("Unexpected response from Open-Meteo.")
    iso = target.isoformat()
    if iso not in daily["time"]:
        raise OpenMeteoError(f"Open-Meteo returned no data for {iso}.")
    value = daily[variable][daily["time"].index(iso)]
    if value is None:
        raise OpenMeteoError(f"Open-Meteo has no {variable} value for {iso} yet.")
    return float(value)


def fetch_daily_rainfall(target, coordinates, variable, timezone,
                         real_today=None, fetch=request_json, timeout=10):
    """Fetch the target day's rainfall for every catchment.

    Returns {"rainfall": {catchment: mm}, "unavailable": {catchment: reason}}.
    """
    real_today = real_today or date.today()
    rainfall, unavailable = {}, {}
    for name in CATCHMENTS:
        point = coordinates.get(name) or {}
        lat, lon = point.get("lat"), point.get("lon")
        if lat is None or lon is None:
            unavailable[name] = "no coordinates configured"
            continue
        try:
            url = build_url(lat, lon, target, real_today, variable, timezone)
            rainfall[name] = extract_value(fetch(url, timeout), target, variable)
        except Exception as exc:  # network errors, bad responses, missing values
            unavailable[name] = str(exc)
    return {"rainfall": rainfall, "unavailable": unavailable}
