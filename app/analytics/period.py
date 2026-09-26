"""
Analytics periods in the store's local time.

today : local midnight -> now; previous = yesterday, same clock window
7d/30d: the last N local days incl. today (today partial); previous = the N days
        before, same length, so comparisons are fair
custom: from/to local dates (inclusive), max 90 days, not in the future
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

COUNTRY_TZ = {"EG": "Africa/Cairo", "SA": "Asia/Riyadh", "AE": "Asia/Dubai", "KW": "Asia/Kuwait",
              "QA": "Asia/Qatar", "BH": "Asia/Bahrain", "OM": "Asia/Muscat", "JO": "Asia/Amman"}
MAX_CUSTOM_DAYS = 90


class PeriodError(ValueError):
    pass


@dataclass(frozen=True)
class Period:
    tz: ZoneInfo
    start: datetime          # UTC, inclusive
    end: datetime            # UTC, exclusive
    prev_start: datetime
    prev_end: datetime
    first_day: date          # local
    last_day: date           # local

    def days(self) -> list[date]:
        n = (self.last_day - self.first_day).days + 1
        return [self.first_day + timedelta(days=i) for i in range(n)]

    def local_day(self, moment: datetime) -> date:
        return moment.astimezone(self.tz).date()

    def describe(self) -> dict:
        prev_last = (self.prev_end - timedelta(microseconds=1)).astimezone(self.tz).date()
        return {"from": self.first_day.isoformat(), "to": self.last_day.isoformat(), "timezone": self.tz.key,
                "previous_from": self.prev_start.astimezone(self.tz).date().isoformat(),
                "previous_to": prev_last.isoformat()}


def store_timezone(country: str | None) -> ZoneInfo:
    return ZoneInfo(COUNTRY_TZ.get((country or "").upper(), "UTC"))


def _midnight(day: date, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, time.min, tzinfo=tz).astimezone(timezone.utc)


def resolve_period(period: str, from_: date | None, to: date | None, country: str | None,
                   now: datetime | None = None) -> Period:
    tz = store_timezone(country)
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    today = now.astimezone(tz).date()

    if period == "custom":
        if from_ is None or to is None:
            raise PeriodError("Custom periods need 'from' and 'to' dates.")
        if from_ > to:
            raise PeriodError("'from' must be on or before 'to'.")
        if to > today:
            raise PeriodError("'to' can't be in the future.")
        if (to - from_).days + 1 > MAX_CUSTOM_DAYS:
            raise PeriodError(f"Custom periods can cover at most {MAX_CUSTOM_DAYS} days.")
        first, last = from_, to
    else:
        days = {"today": 1, "7d": 7, "30d": 30}.get(period)
        if days is None:
            raise PeriodError("period must be today, 7d, 30d or custom.")
        first, last = today - timedelta(days=days - 1), today

    start = _midnight(first, tz)
    end = min(_midnight(last + timedelta(days=1), tz), now)
    span = _midnight(last + timedelta(days=1), tz) - start           # whole-day span
    prev_start = start - span
    prev_end = prev_start + (end - start)                              # same length as the current window
    return Period(tz, start, end, prev_start, prev_end, first, last)