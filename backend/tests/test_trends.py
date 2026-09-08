"""Tests for the pure time-series maths in `reasoning/trends.py`.

The module documents two traps — a two-point fit reporting r² = 1.0 by
construction, and a baseline that includes the current value — and both get
explicit tests here, because the whole Monitoring Agent's claim to judge
"current value + trend + personal baseline" rests on these functions not
quietly lying.

Expected slopes and r² values are hand-computed from the ordinary least-squares
normal equations rather than recorded from a passing run, so a regression in the
arithmetic fails against the mathematics rather than against itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.reasoning.trends import (
    NO_SAMPLES,
    SINGLE_SAMPLE,
    consecutive_adverse_run,
    coverage_fraction,
    derive_baseline,
    describe_series,
    linear_trend,
)

START = datetime(2026, 9, 4, 8, tzinfo=timezone.utc)


def _days(n: int, start: datetime = START) -> list[datetime]:
    return [start + timedelta(days=i) for i in range(n)]


# --- Linear trend --------------------------------------------------------------


class TestLinearTrend:
    def test_perfect_linear_fall_has_exact_slope_and_unit_fit(self):
        # y = 10 - 2x over x = 0..4: Sxy = -20, Sxx = 10, Syy = 40.
        trend = linear_trend(_days(5), [10, 8, 6, 4, 2])
        assert trend.slope_per_day == pytest.approx(-2.0)
        assert trend.r_squared == pytest.approx(1.0)
        assert trend.n == 5

    def test_demo_spo2_series_matches_hand_computed_fit(self):
        # 98,97,95,93,91: Sxy = -18, Sxx = 10, Syy = 32.8 -> r^2 = 324/328.
        trend = linear_trend(_days(5), [98, 97, 95, 93, 91])
        assert trend.slope_per_day == pytest.approx(-1.8)
        assert trend.r_squared == pytest.approx(324 / 328)

    def test_two_points_fit_exactly_and_that_is_the_trap(self):
        """r² = 1.0 here is an artefact of there being two points, not evidence
        of a trend. The maths reports it honestly; `MIN_SAMPLES_FOR_TREND` in
        the rules layer is what refuses to act on it."""
        trend = linear_trend(_days(2), [10, 8])
        assert trend.slope_per_day == pytest.approx(-2.0)
        assert trend.r_squared == pytest.approx(1.0)
        assert trend.n == 2

    def test_fewer_than_two_points_cannot_be_fitted(self):
        assert linear_trend([], []) is None
        assert linear_trend(_days(1), [10]) is None

    def test_identical_timestamps_have_no_defined_slope(self):
        """Sxx = 0: the slope is undefined rather than zero, and returning 0.0
        would report a vertical series as flat."""
        assert linear_trend([START, START], [1, 2]) is None

    def test_constant_series_has_zero_slope_but_undefined_fit(self):
        """Syy = 0: perfectly flat, with nothing for the fit to explain. r² is
        None here, not 1.0 — the series was never trending."""
        trend = linear_trend(_days(4), [5, 5, 5, 5])
        assert trend.slope_per_day == pytest.approx(0.0)
        assert trend.r_squared is None
        assert not trend.is_well_fitted

    def test_scattered_series_has_a_low_r_squared(self):
        # 1,5,2,6,3: Sxy = 5, Sxx = 10, Syy = 17.2 -> r^2 = 25/172.
        trend = linear_trend(_days(5), [1, 5, 2, 6, 3])
        assert trend.slope_per_day == pytest.approx(0.5)
        assert trend.r_squared == pytest.approx(25 / 172)
        assert trend.is_well_fitted  # defined, just poor

    def test_slope_is_per_day_regardless_of_sampling_interval(self):
        # Two samples 12 hours apart, rising by 1 -> 2 per day.
        trend = linear_trend([START, START + timedelta(hours=12)], [1, 2])
        assert trend.slope_per_day == pytest.approx(2.0)

    def test_input_order_does_not_matter(self):
        times = _days(5)
        values = [98, 97, 95, 93, 91]
        shuffled = list(reversed(list(zip(times, values, strict=True))))
        trend = linear_trend(
            [t for t, _ in shuffled], [v for _, v in shuffled]
        )
        assert trend.slope_per_day == pytest.approx(-1.8)

    def test_ragged_input_uses_the_shorter_length(self):
        trend = linear_trend(_days(5), [10, 8, 6])
        assert trend.n == 3
        assert trend.slope_per_day == pytest.approx(-2.0)


# --- Personal baseline ----------------------------------------------------------


class TestDeriveBaseline:
    def test_demo_series_baseline_is_the_earliest_two_samples(self):
        baseline, used, method = derive_baseline([98, 97, 95, 93, 91], 2)
        assert baseline == pytest.approx(97.5)
        assert used == 2
        assert "earliest 2" in method

    def test_the_current_value_is_never_part_of_its_own_baseline(self):
        """The load-bearing invariant. Including the last value would shrink
        every delta toward zero by construction: a series ending in a 100-point
        jump would report a baseline of 20 instead of 0."""
        baseline, used, _ = derive_baseline([0, 0, 0, 0, 100], 2)
        assert baseline == pytest.approx(0.0)
        assert used == 2

    def test_window_is_clamped_to_leave_one_sample_out(self):
        baseline, used, method = derive_baseline([98, 97, 95, 93, 91], 10)
        assert used == 4  # min(10, n-1)
        assert baseline == pytest.approx((98 + 97 + 95 + 93) / 4)
        assert method == "mean of the earliest 4 samples in the window"

    def test_window_of_one_takes_the_earliest_sample(self):
        baseline, used, method = derive_baseline([98, 97, 95, 93, 91], 1)
        assert baseline == pytest.approx(98)
        assert used == 1
        assert "capped at 1 of 5" in method

    def test_two_samples_leave_exactly_one_predecessor(self):
        baseline, used, method = derive_baseline([10, 4], 2)
        assert baseline == pytest.approx(10)
        assert used == 1
        assert "only one observation precedes" in method

    def test_a_single_observation_cannot_be_its_own_baseline(self):
        assert derive_baseline([7], 2) == (None, 0, SINGLE_SAMPLE)

    def test_no_observations_no_baseline(self):
        assert derive_baseline([], 2) == (None, 0, NO_SAMPLES)


# --- Adverse runs ----------------------------------------------------------------


class TestConsecutiveAdverseRun:
    def test_monotonic_fall_counts_every_transition(self):
        assert consecutive_adverse_run([98, 97, 95, 93, 91], -1) == 4

    def test_monotonic_rise_counts_every_transition(self):
        assert consecutive_adverse_run([72, 75, 82, 94, 103], 1) == 4

    def test_a_flat_step_breaks_the_run(self):
        """Counted from the end: 95->93 and 97->95 are adverse, then 97->97 is
        flat. A value that has stopped moving has stopped worsening, and
        treating a plateau as continued deterioration would keep escalating a
        patient who has stabilised."""
        assert consecutive_adverse_run([98, 97, 97, 95, 93], -1) == 2

    def test_flat_series_has_no_run(self):
        assert consecutive_adverse_run([3, 3], -1) == 0

    def test_a_favourable_latest_step_resets_the_run_to_zero(self):
        assert consecutive_adverse_run([95, 93, 94], -1) == 0

    def test_favourable_movement_never_counts_as_adverse(self):
        assert consecutive_adverse_run([1, 2, 3], -1) == 0
        assert consecutive_adverse_run([3, 2, 1], 1) == 0

    def test_runs_count_from_the_end_only(self):
        # An early adverse run that was interrupted does not count.
        assert consecutive_adverse_run([98, 96, 97, 96, 95], -1) == 2

    def test_short_series(self):
        assert consecutive_adverse_run([5], -1) == 0
        assert consecutive_adverse_run([], 1) == 0

    @pytest.mark.parametrize("direction", [0, 2, -2])
    def test_direction_outside_the_vocabulary_raises(self, direction):
        with pytest.raises(ValueError):
            consecutive_adverse_run([1, 2], direction)


# --- Coverage ---------------------------------------------------------------------


class TestCoverageFraction:
    def test_full_daily_coverage(self):
        assert coverage_fraction(5, 5, 1) == pytest.approx(1.0)

    def test_gappy_window(self):
        assert coverage_fraction(2, 5, 1) == pytest.approx(0.4)

    def test_coverage_is_capped_at_one(self):
        """An hourly device should not read as 2400% complete; coverage answers
        'are there gaps?', not 'how busy is the device?'."""
        assert coverage_fraction(24, 1, 1) == pytest.approx(1.0)

    def test_no_samples(self):
        assert coverage_fraction(0, 5, 1) == pytest.approx(0.0)

    def test_degenerate_windows_do_not_divide_by_zero(self):
        assert coverage_fraction(3, 0, 1) == pytest.approx(1.0)
        assert coverage_fraction(0, 0, 1) == pytest.approx(0.0)
        assert coverage_fraction(3, 5, 0) == pytest.approx(1.0)


# --- Series summary ----------------------------------------------------------------


class TestDescribeSeries:
    def test_demo_spo2_series_end_to_end(self):
        stats = describe_series(_days(5), [98, 97, 95, 93, 91], 2, 1)
        assert stats.count == 5
        assert stats.first_at == START
        assert stats.last_at == START + timedelta(days=4)
        assert stats.span_days == pytest.approx(4.0)
        assert stats.window_days == 5  # calendar days touched, not elapsed days
        assert stats.current == pytest.approx(91)
        assert stats.minimum == pytest.approx(91)
        assert stats.maximum == pytest.approx(98)
        assert stats.mean == pytest.approx(94.8)
        assert stats.baseline == pytest.approx(97.5)
        assert stats.baseline_samples == 2
        assert stats.delta == pytest.approx(-6.5)
        assert stats.delta_pct == pytest.approx(-6.5 / 97.5 * 100)
        assert stats.slope_per_day == pytest.approx(-1.8)
        assert stats.r_squared == pytest.approx(324 / 328)
        assert stats.coverage == pytest.approx(1.0)
        assert stats.has_trend and stats.has_baseline

    def test_empty_series_is_all_none_rather_than_an_error(self):
        """A metric the device never reported is a normal outcome. Raising here
        would turn a sparse row into a failed agent run."""
        stats = describe_series([], [], 2, 1)
        assert stats.count == 0
        assert stats.current is None
        assert stats.baseline is None
        assert stats.baseline_method == NO_SAMPLES
        assert stats.slope_per_day is None
        assert stats.r_squared is None
        assert stats.window_days == 0
        assert stats.coverage == pytest.approx(0.0)
        assert not stats.has_trend and not stats.has_baseline

    def test_single_sample_reports_a_value_but_no_trend_and_no_baseline(self):
        stats = describe_series([START], [91], 2, 1)
        assert stats.count == 1
        assert stats.current == pytest.approx(91)
        assert stats.baseline is None
        assert stats.baseline_method == SINGLE_SAMPLE
        assert stats.delta is None
        assert stats.delta_pct is None
        assert stats.slope_per_day is None
        assert stats.window_days == 1
        assert stats.coverage == pytest.approx(1.0)
        assert not stats.has_trend and not stats.has_baseline

    def test_zero_baseline_yields_no_percentage(self):
        """delta_pct is guarded: a percentage of zero is meaningless, not
        infinite."""
        stats = describe_series(_days(2), [0.0, 5.0], 2, 1)
        assert stats.baseline == pytest.approx(0.0)
        assert stats.delta == pytest.approx(5.0)
        assert stats.delta_pct is None

    def test_window_is_elapsed_days_plus_one(self):
        """The window counts the days the series spans, +1 so a 5-day daily
        series spanning 4.0 elapsed days reports 5. Two samples 2 hours apart
        therefore report a 1-day window even when they cross midnight UTC —
        coverage is about elapsed time, not calendar dates touched."""
        across_midnight = [
            datetime(2026, 1, 1, 23, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 1, tzinfo=timezone.utc),
        ]
        stats = describe_series(across_midnight, [1, 2], 2, 1)
        assert stats.span_days == pytest.approx(2 / 24)
        assert stats.window_days == 1

        stats = describe_series(_days(5), [98, 97, 95, 93, 91], 2, 1)
        assert stats.span_days == pytest.approx(4.0)
        assert stats.window_days == 5

    def test_unsorted_input_is_sorted_before_analysis(self):
        times = _days(3)
        values = [95, 93, 91]
        reversed_pairs = list(reversed(list(zip(times, values, strict=True))))
        stats = describe_series(
            [t for t, _ in reversed_pairs], [v for _, v in reversed_pairs], 2, 1
        )
        assert stats.current == pytest.approx(91)  # latest by time, not by position
        assert stats.baseline == pytest.approx(94)  # mean of the earliest two by time
        assert stats.slope_per_day == pytest.approx(-2.0)

    def test_improvement_shows_as_a_positive_delta(self):
        stats = describe_series(_days(2), [91, 98], 2, 1)
        assert stats.delta == pytest.approx(7.0)
        assert stats.delta_pct == pytest.approx(7 / 91 * 100)
