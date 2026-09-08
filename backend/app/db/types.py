"""Custom column types.

Two SQLite/SQLAlchemy defaults are wrong for this project, and both fail
silently rather than raising:

`UTCDateTime` exists because SQLite's DATETIME affinity discards the UTC offset,
so a plain `DateTime(timezone=True)` round-trips as a *naive* datetime. That is
not a cosmetic problem here: this system computes time intervals (the monitoring
observation window, `HistoricalChange.interval_days`, days-since-onset), and
comparing a naive value read from the database against `datetime.now(timezone.utc)`
raises `TypeError: can't compare offset-naive and offset-aware datetimes`.

Storing ISO-8601 text with an explicit offset and always returning an aware UTC
datetime makes that class of bug unrepresentable.

`JSONNoneAsNull` exists because SQLAlchemy's `JSON` defaults to
`none_as_null=False`, which stores a Python `None` as the four-character JSON
document `'null'` instead of SQL NULL. Reading the row back still yields `None`,
so the difference is invisible until something filters on it: `WHERE col IS NULL`
matches nothing and `WHERE col IS NOT NULL` matches every row. Several columns
here are legitimately absent for some rows (`normalised` on a lab result,
`final_assessment` on a pending run), so that distinction has to be real.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, String
from sqlalchemy.types import TypeDecorator


class JSONNoneAsNull(JSON):
    """JSON column where a Python `None` is stored as SQL NULL, not `'null'`."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["none_as_null"] = True
        super().__init__(*args, **kwargs)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC timestamp that survives a SQLite round-trip."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> str | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"UTCDateTime requires a datetime, got {type(value).__name__}")
        # Assume naive input is already UTC rather than local time: every
        # producer in this codebase writes UTC, and silently applying a local
        # offset would shift stored timestamps by the machine's timezone.
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def process_result_value(self, value: str | datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
