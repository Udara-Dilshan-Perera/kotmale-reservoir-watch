"""A movable stand-in for 'today', used only in demonstration mode
(KOTMALE_SYSTEM_DATE set). Lets a chained multi-day demo advance one day at a
time from the interface, with no restart and no file editing.

Never used in a real deployment: build_default_state() only creates one when
the system date has been explicitly overridden, so the real calendar date is
never advanceable.
"""
from datetime import date, timedelta


class DemoClock:
    def __init__(self, start_date):
        self._current = start_date

    def today(self):
        return self._current

    def set(self, new_date):
        self._current = new_date

    def advance(self):
        self._current += timedelta(days=1)
        return self._current
