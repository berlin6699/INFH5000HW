"""Time-series maths for the Monitoring Agent.

Pure functions over sequences of timestamps and floats. No database, no schemas,
and — deliberately — **no clinical knowledge**. This module reports what the
numbers did; `monitoring_rules.py` decides, against `thresholds.py`, whether
what they did is adverse. Splitting those two is what lets the direction of
"bad" be changed in one place instead of being baked into the arithmetic.

Two traps this module exists to avoid:

**A trend from two points is not a trend.** An ordinary least-squares line
through two points passes through both, so r² is 1.0 by construction. Reporting
that as a confident slope with a perfect fit is the easiest way for this agent
to be wrong and look right, so `linear_trend` refuses to fit fewer than two
points and returns no r² when the series has no variance to explain.

**A baseline that includes the current value cannot show deterioration.** If the
latest reading is part of its own baseline, every delta shrinks toward zero by
construction. The baseline is therefore always drawn from samples strictly
before the last one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

SECONDS_PER_DAY = 86400.0

NO_SAMPLES = "no samples in the observation window"
SINGLE_SAMPLE = (
    "single sample; a personal baseline cannot be derived from one observation"
)


@dataclass(frozen=True)
class LinearTrend:
    """An ordinary least-squares fit of value against time."""

    slope_per_day: float
    r_squared: float | None
    n: int

    @property
    def is_well_fitted(self) -> bool:
        """False when r² is absent — a constant series has no variance to
        explain, so the fit is undefined rather than perfect."""
        return self.r_squared is not None


@dataclass(frozen=True)
class SeriesStats:
    """Everything the monitoring rules need about one metric's series.

    Optional fields are `None` when the series cannot support them. That is
    load-bearing: `slope_per_day=None` means "not enough data to know", which is
    a different statement from `slope_per_day=0.0`, meaning "measured and flat".
    Collapsing the two would report an unobserved trend as a stable one.
    """

    count: int
    first_at: datetime | None = None
    last_at: datetime | None = None
    span_days: float = 0.0
    window_days: int = 0

    current: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None

    baseline: float | None = None
    baseline_samples: int = 0
    baseline_method: str = NO_SAMPLES

    delta: float | None = None
    delta_pct: float | None = None

    slope_per_day: float | None = None
    r_squared: float | None = None

    coverage: float = 0.0

    @property
    def has_trend(self) -> bool:
        return self.slope_per_day is not None

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None


def _days_from(first: datetime, when: datetime) -> float:
    return (when - first).total_seconds() / SECONDS_PER_DAY


def linear_trend(
    times: Sequence[datetime], values: Sequence[float]
) -> LinearTrend | None:
    """Ordinary least-squares slope in units per day, with goodness of fit.

    Returns None when the series cannot be fitted at all — fewer than two
    points, or every point at the same instant, where the slope is undefined
    rather than zero.

    r² is None when the series has no variance (every value identical). The
    slope is still reported, because a flat series genuinely is flat; the fit
    quality is undefined, because there was nothing to explain.
    """
    n = min(len(times), len(values))
    if n < 2:
        return None

    pairs = sorted(zip(times[:n], values[:n], strict=True), key=lambda p: p[0])
    xs = [_days_from(pairs[0][0], t) for t, _ in pairs]
    ys = [v for _, v in pairs]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:
        return None

    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    syy = sum((y - y_mean) ** 2 for y in ys)

    return LinearTrend(
        slope_per_day=sxy / sxx,
        # r² = Sxy² / (Sxx·Syy). Undefined when Syy is 0, not 1.0 — a constant
        # series is not a perfectly explained trend, it is a series with no
        # variation in it.
        r_squared=(sxy * sxy) / (sxx * syy) if syy > 0 else None,
        n=n,
    )


def derive_baseline(
    values: Sequence[float], max_window: int
) -> tuple[float | None, int, str]:
    """The patient's own baseline, as `(value, samples_used, method)`.

    `max_window` is clamped to `len(values) - 1` so the latest observation can
    never be part of the baseline it is compared against. With two samples that
    leaves one, and the method string says so rather than silently claiming a
    two-sample mean.

    Returns `(None, 0, reason)` for a series too short to have a baseline,
    because "baseline equals current value" would report every patient as
    unchanged.
    """
    n = len(values)
    if n == 0:
        return None, 0, NO_SAMPLES
    if n == 1:
        return None, 0, SINGLE_SAMPLE

    window = max(1, min(max_window, n - 1))
    earliest = values[:window]
    baseline = sum(earliest) / len(earliest)

    if window == 1:
        method = (
            "earliest sample in the window (only one observation precedes the latest)"
            if n == 2
            else f"earliest sample in the window (baseline window capped at {window} of {n})"
        )
    else:
        method = f"mean of the earliest {window} samples in the window"
    return baseline, window, method


def consecutive_adverse_run(values: Sequence[float], adverse_direction: int) -> int:
    """Consecutive adverse transitions ending at the latest sample.

    `adverse_direction` is -1 when falling is adverse (SpO₂, sleep, activity)
    and +1 when rising is adverse (heart rate, temperature, respiratory rate).
    The direction is supplied by the caller because it is clinical knowledge,
    and this module deliberately has none.

    Counts *transitions*, which equals days when sampling is daily. A flat step
    breaks the run: a value that stops moving has stopped worsening, and
    treating "unchanged" as "still deteriorating" would keep escalating a
    patient who has plateaued.
    """
    if adverse_direction not in (-1, 1):
        raise ValueError(f"adverse_direction must be -1 or 1, got {adverse_direction!r}")
    if len(values) < 2:
        return 0

    run = 0
    for previous, current in zip(reversed(values[:-1]), reversed(values[1:]), strict=True):
        if (current - previous) * adverse_direction > 0:
            run += 1
        else:
            break
    return run


def coverage_fraction(count: int, window_days: int, expected_per_day: float) -> float:
    """Observed samples as a fraction of what the window should contain.

    Capped at 1.0: a device reporting hourly should not read as 2400% complete,
    and coverage answers "are there gaps?", not "how busy is the device?".
    """
    if window_days <= 0 or expected_per_day <= 0:
        return 1.0 if count > 0 else 0.0
    expected = window_days * expected_per_day
    return min(1.0, count / expected)


def describe_series(
    times: Sequence[datetime],
    values: Sequence[float],
    baseline_window: int,
    expected_per_day: float,
) -> SeriesStats:
    """Summarise one metric's series. Empty or ragged input yields an all-None
    `SeriesStats` rather than raising, because a metric the device never reported
    is a normal outcome, not an error."""
    n = min(len(times), len(values))
    if n == 0:
        return SeriesStats(count=0)

    pairs = sorted(zip(times[:n], values[:n], strict=True), key=lambda p: p[0])
    ordered_times = [t for t, _ in pairs]
    ordered_values = [v for _, v in pairs]

    first_at, last_at = ordered_times[0], ordered_times[-1]
    span_days = _days_from(first_at, last_at)
    # Calendar days touched, so a 5-day daily series spanning 4.0 days reports a
    # 5-day window rather than a 4-day one.
    window_days = int(span_days) + 1

    baseline, baseline_samples, baseline_method = derive_baseline(
        ordered_values, baseline_window
    )
    current = ordered_values[-1]
    delta = current - baseline if baseline is not None else None
    # Guarded: a zero baseline makes a percentage meaningless rather than
    # infinite, and None is the honest answer.
    delta_pct = (
        (delta / abs(baseline)) * 100.0
        if delta is not None and baseline not in (None, 0)
        else None
    )

    trend = linear_trend(ordered_times, ordered_values)

    return SeriesStats(
        count=n,
        first_at=first_at,
        last_at=last_at,
        span_days=span_days,
        window_days=window_days,
        current=current,
        minimum=min(ordered_values),
        maximum=max(ordered_values),
        mean=sum(ordered_values) / n,
        baseline=baseline,
        baseline_samples=baseline_samples,
        baseline_method=baseline_method,
        delta=delta,
        delta_pct=delta_pct,
        slope_per_day=trend.slope_per_day if trend else None,
        r_squared=trend.r_squared if trend else None,
        coverage=coverage_fraction(n, window_days, expected_per_day),
    )
