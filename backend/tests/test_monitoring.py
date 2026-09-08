"""Monitoring Agent and its rule engine.

Two layers, tested separately because they fail differently:

* `app.reasoning.monitoring_rules` is pure — tested from constructed
  `VitalSample`s with no database, so a threshold change shows up as a specific
  rule assertion rather than as a diff in a seeded assessment.
* `app.agents.monitoring` owns the observation window, the anchor date and the
  trace — tested against the seeded temporary database from conftest.

The reverse cases carry as much weight as the demo scenario. This agent's whole
claim is that it does not conclude from a single time point, so the tests assert
that a lone abnormal reading still fires the absolute-threshold rule (91% is
91%) but is *not* reported as deterioration, that a flat abnormal series is
stable rather than worsening, and that two samples never produce a slope. A
suite that only checked the demo patient would pass just as happily if those
distinctions were deleted.

Numeric expectations are hand-computed from the published NEWS2 table and from
ordinary least squares over the demo series, not captured from a run, so a
regression in the maths fails a test rather than silently becoming the new
expected value.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from app.agents.monitoring import load_samples, run_monitoring
from app.db.models import MonitoringRecordRow, PatientRow
from app.reasoning.monitoring_rules import (
    MONITORING_RULE_CATEGORY,
    NEWS2_TRIGGER_RULE_ID,
    QUALITY_RULE_ID,
    RAPID_RULE_ID,
    SPEC_BY_METRIC,
    VITAL_SPECS,
    MetricAnalysis,
    MonitoringDerivation,
    analyse_metric,
    assess_monitoring,
    classify_direction,
    composite_anomaly,
    series_for,
)
from app.reasoning.thresholds import (
    ANOMALY_WEIGHT_BASELINE,
    ANOMALY_WEIGHT_RUN,
    ANOMALY_WEIGHT_SLOPE,
    ANOMALY_WEIGHT_THRESHOLD,
    MIN_R_SQUARED_FOR_TREND,
    MIN_SAMPLES_FOR_TREND,
    NEWS2_BANDS,
    NEWS2_PARTIAL_SCORE_CAVEAT,
    NEWS2_SOURCE,
    RAPID_DETERIORATION_MIN_RUN,
    SUPPORTING_SIGNAL_WEIGHT,
    news2_bands,
    news2_max_parameter_score,
    news2_score,
)
from app.reasoning.trends import SeriesStats
from app.safety import audit_prose
from app.schemas import (
    ClinicalDimension,
    MonitoringAssessment,
    TrendDirection,
    TrendFeature,
    VitalMetric,
    VitalSample,
    utcnow,
)

DEMO_ID = "PT-DEMO-001"
ANCHOR = date(2026, 9, 8)
START = datetime(2026, 9, 4, 8, tzinfo=timezone.utc)

SPO2 = SPEC_BY_METRIC[VitalMetric.SPO2]
HEART_RATE = SPEC_BY_METRIC[VitalMetric.HEART_RATE]

DEMO_SERIES: dict[str, tuple[float, ...]] = {
    "spo2": (98, 97, 95, 93, 91),
    "heart_rate": (72, 75, 82, 94, 103),
    "temperature_c": (36.8, 37.4, 38.1, 38.5, 38.7),
    "respiratory_rate": (14, 16, 18, 21, 24),
    "sleep_hours": (7.2, 6.8, 5.9, 5.1, 4.2),
    "activity_steps": (8200, 6100, 3400, 1500, 600),
}
"""The demo patient's five days, transcribed here so tests can build windows the
fixture does not contain. A test asserts this matches `data/synthetic/
monitoring_5d.csv`, because a hand-copy that drifts from the shipped fixture
would make every pinned number below describe a patient nobody can demo."""


def _day(n: int) -> datetime:
    return START + timedelta(days=n)


def _sample(n: int, **values: float) -> VitalSample:
    return VitalSample(recorded_at=_day(n), **values)


def _series(field: str, values: tuple[float, ...]) -> list[VitalSample]:
    """One metric reported daily, nothing else — the sparse-device case."""
    return [_sample(i, **{field: v}) for i, v in enumerate(values)]


def _demo_samples() -> list[VitalSample]:
    return [_sample(i, **{f: vals[i] for f, vals in DEMO_SERIES.items()}) for i in range(5)]


def _analysis(
    *,
    spec=SPO2,
    direction: TrendDirection = TrendDirection.WORSENING,
    slope: float | None = -2.0,
    r_squared: float | None = 1.0,
    run: int = 4,
    count: int = 5,
    current: float = 91.0,
) -> MetricAnalysis:
    """A `MetricAnalysis` assembled by hand, for testing `is_rapid`'s four gates.

    The real construction path is covered by `TestAnalyseMetric`; here the point
    is to move one gate at a time, and no sample series can hold, say, a run of 4
    with an r² of 0.49.
    """
    feature = TrendFeature(
        metric=spec.metric,
        unit=spec.unit,
        dimension=spec.dimension,
        sample_count=count,
        current=current,
        min_value=current,
        max_value=current,
        direction=direction,
        slope_per_day=slope,
        r_squared=r_squared,
        consecutive_worsening_days=run,
    )
    return MetricAnalysis(spec=spec, stats=SeriesStats(count=count), band=None, feature=feature)


# --- The vital-spec table ------------------------------------------------------


class TestVitalSpecTable:
    def test_every_metric_is_covered_exactly_once(self):
        """A metric added to `VitalMetric` and forgotten here would simply never
        be analysed, and nothing else would fail."""
        assert [s.metric for s in VITAL_SPECS] == list(VitalMetric)

    def test_spec_lookup_matches_the_table(self):
        assert set(SPEC_BY_METRIC) == set(VitalMetric)
        assert all(SPEC_BY_METRIC[s.metric] is s for s in VITAL_SPECS)

    def test_adverse_direction_is_the_only_encoding_of_which_way_is_bad(self):
        assert {s.metric: s.adverse_direction for s in VITAL_SPECS} == {
            VitalMetric.SPO2: -1,
            VitalMetric.HEART_RATE: 1,
            VitalMetric.TEMPERATURE_C: 1,
            VitalMetric.RESPIRATORY_RATE: 1,
            VitalMetric.SLEEP_HOURS: -1,
            VitalMetric.ACTIVITY_STEPS: -1,
        }

    def test_rule_codes_are_unique(self):
        codes = [s.rule_code for s in VITAL_SPECS]
        assert len(set(codes)) == len(codes), "two metrics would share a rule id"

    def test_units_and_dimensions_are_pinned(self):
        assert {s.metric: s.unit for s in VITAL_SPECS} == {
            VitalMetric.SPO2: "%",
            VitalMetric.HEART_RATE: "bpm",
            VitalMetric.TEMPERATURE_C: "C",
            VitalMetric.RESPIRATORY_RATE: "breaths/min",
            VitalMetric.SLEEP_HOURS: "hours",
            VitalMetric.ACTIVITY_STEPS: "steps",
        }
        assert {s.metric: s.dimension for s in VITAL_SPECS} == {
            VitalMetric.SPO2: ClinicalDimension.OXYGENATION,
            VitalMetric.HEART_RATE: ClinicalDimension.CARDIAC_RATE,
            VitalMetric.TEMPERATURE_C: ClinicalDimension.TEMPERATURE,
            VitalMetric.RESPIRATORY_RATE: ClinicalDimension.RESPIRATORY_RATE,
            VitalMetric.SLEEP_HOURS: ClinicalDimension.FUNCTIONAL_STATUS,
            VitalMetric.ACTIVITY_STEPS: ClinicalDimension.FUNCTIONAL_STATUS,
        }

    def test_wearable_signals_are_not_clinical_measurements(self):
        assert {s.metric for s in VITAL_SPECS if s.is_supporting_signal} == {
            VitalMetric.SLEEP_HOURS,
            VitalMetric.ACTIVITY_STEPS,
        }

    def test_every_news2_parameter_is_a_clinical_measurement(self):
        """One direction only. The converse is deliberately not asserted: a
        device-derived NEWS2 parameter would be scoreable and still not be a
        clinical measurement, which is why `is_supporting_signal` is not derived
        from NEWS2 coverage."""
        for spec in VITAL_SPECS:
            if news2_bands(spec.metric.value):
                assert spec.clinical_measurement, spec.metric

    def test_temperature_has_no_relative_baseline_form(self):
        """Celsius has an arbitrary zero, so '2.7% above baseline' means
        something different depending on whether the baseline was 35.0 or 37.0."""
        spec = SPEC_BY_METRIC[VitalMetric.TEMPERATURE_C]
        assert spec.baseline_delta is not None
        assert spec.baseline_delta_pct is None

    def test_activity_has_only_the_relative_form(self):
        """A fixed drop of 3000 steps is a rounding error for a very active
        patient and most of the day for a sedentary one."""
        spec = SPEC_BY_METRIC[VitalMetric.ACTIVITY_STEPS]
        assert spec.baseline_delta is None
        assert spec.baseline_delta_pct is not None

    def test_every_spec_carries_an_adverse_slope(self):
        for spec in VITAL_SPECS:
            assert spec.adverse_slope.value > 0, spec.metric
            assert spec.adverse_slope.unit.endswith("/day"), spec.metric


class TestSeriesFor:
    def test_unreported_metric_yields_an_empty_series(self):
        assert series_for([_sample(0, heart_rate=80)], SPO2) == ([], [])

    def test_sparse_rows_are_filtered_per_metric(self):
        samples = [_sample(0, spo2=98), _sample(1, heart_rate=80), _sample(2, spo2=96)]
        assert series_for(samples, SPO2) == ([_day(0), _day(2)], [98.0, 96.0])

    def test_input_order_does_not_matter(self):
        samples = [_sample(2, spo2=96), _sample(0, spo2=98), _sample(1, spo2=97)]
        times, values = series_for(samples, SPO2)
        assert times == [_day(0), _day(1), _day(2)]
        assert values == [98.0, 97.0, 96.0]

    def test_values_are_floats(self):
        """`activity_steps` is an int on the wire; the maths must not depend on
        which numeric type the source happened to use."""
        _, values = series_for(_series("activity_steps", (8200, 6100)), SPEC_BY_METRIC[VitalMetric.ACTIVITY_STEPS])
        assert all(isinstance(v, float) for v in values)


# --- The published band table --------------------------------------------------


def _expected_spo2(v: float) -> int:
    if v <= 91:
        return 3
    if v <= 93:
        return 2
    if v <= 95:
        return 1
    return 0


def _expected_heart_rate(v: float) -> int:
    if v <= 40:
        return 3
    if v <= 50:
        return 1
    if v <= 90:
        return 0
    if v <= 110:
        return 1
    if v <= 130:
        return 2
    return 3


def _expected_temperature(v: float) -> int:
    if v <= 35.0:
        return 3
    if v <= 36.0:
        return 1
    if v <= 38.0:
        return 0
    if v <= 39.0:
        return 1
    return 2


def _expected_respiratory_rate(v: float) -> int:
    if v <= 8:
        return 3
    if v <= 11:
        return 1
    if v <= 20:
        return 0
    if v <= 24:
        return 2
    return 3


class TestNews2Bands:
    """A transcription check against the published table (RCP, 2017).

    These expectations are written out independently of `NEWS2_BANDS` on purpose.
    A test that derived them from the table would keep passing if a band were
    edited into something nobody published.
    """

    @pytest.mark.parametrize("v", range(80, 101))
    def test_spo2_table(self, v):
        assert news2_score("spo2", v) == _expected_spo2(v)

    @pytest.mark.parametrize("v", range(30, 151))
    def test_heart_rate_table(self, v):
        assert news2_score("heart_rate", v) == _expected_heart_rate(v)

    @pytest.mark.parametrize("v", [round(34.0 + 0.1 * i, 1) for i in range(61)])
    def test_temperature_table(self, v):
        assert news2_score("temperature_c", v) == _expected_temperature(v)

    @pytest.mark.parametrize("v", range(5, 36))
    def test_respiratory_rate_table(self, v):
        assert news2_score("respiratory_rate", v) == _expected_respiratory_rate(v)

    def test_bands_tile_the_line(self):
        for metric, bands in NEWS2_BANDS.items():
            assert bands[0].low is None, metric
            assert bands[-1].high is None, metric
            for lower, upper in zip(bands, bands[1:]):
                assert lower.high == upper.low, f"{metric}: gap or overlap at {lower.high}"
            assert all(0 <= b.score <= 3 for b in bands), metric

    def test_every_value_lands_in_exactly_one_band(self):
        for metric, bands in NEWS2_BANDS.items():
            for i in range(400):
                value = 20 + i * 0.5
                assert sum(b.contains(value) for b in bands) == 1, f"{metric} at {value}"

    def test_fractions_resolve_toward_the_more_abnormal_band_when_falling_is_adverse(self):
        assert news2_score("spo2", 91.5) == 3
        assert news2_score("spo2", 93.5) == 2
        assert news2_score("spo2", 95.5) == 1
        assert news2_score("spo2", 96.0) == 0

    def test_fractions_resolve_the_other_way_when_rising_is_adverse(self):
        """The asymmetry `ScoreBand`'s docstring records, pinned so a change to
        it is a deliberate decision rather than a side effect."""
        assert news2_score("heart_rate", 130.5) == 2
        assert news2_score("temperature_c", 38.05) == 0
        assert news2_score("respiratory_rate", 20.5) == 0

    def test_metrics_outside_news2_are_not_scored(self):
        assert news2_score("sleep_hours", 4.2) is None
        assert news2_score("activity_steps", 600) is None
        assert news2_bands("sleep_hours") == ()

    def test_a_missing_value_is_not_a_score_of_zero(self):
        """Zero is a real score meaning normal; None means not scoreable."""
        assert news2_score("spo2", None) is None

    def test_max_parameter_score_is_derived_from_the_table(self):
        assert all(news2_max_parameter_score(m) == 3 for m in NEWS2_BANDS)
        assert news2_max_parameter_score("sleep_hours") is None


# --- The anomaly score ---------------------------------------------------------


class TestCompositeAnomaly:
    def test_no_components_is_zero_not_an_error(self):
        assert composite_anomaly([], supporting_signal=False) == 0.0

    def test_a_single_component_is_its_own_value(self):
        assert composite_anomaly([(0.5, 0.25)], supporting_signal=False) == 0.25

    def test_weights_are_renormalised_over_the_components_that_apply(self):
        """Without this a metric NEWS2 does not score could never exceed the sum
        of its three remaining weights, and could never rank first."""
        assert composite_anomaly(
            [(ANOMALY_WEIGHT_BASELINE.value, 1.0), (ANOMALY_WEIGHT_SLOPE.value, 1.0),
             (ANOMALY_WEIGHT_RUN.value, 1.0)],
            supporting_signal=False,
        ) == 1.0

    def test_supporting_signals_are_scaled_back_down(self):
        raw = composite_anomaly([(0.25, 1.0), (0.20, 1.0), (0.15, 1.0)], supporting_signal=False)
        scaled = composite_anomaly([(0.25, 1.0), (0.20, 1.0), (0.15, 1.0)], supporting_signal=True)
        assert scaled == pytest.approx(raw * SUPPORTING_SIGNAL_WEIGHT.value)
        assert scaled == 0.6

    def test_result_is_clamped_to_the_unit_interval(self):
        assert composite_anomaly([(0.4, 5.0)], supporting_signal=False) == 1.0
        assert composite_anomaly([(0.4, -5.0)], supporting_signal=False) == 0.0

    def test_rounded_to_four_places(self):
        assert composite_anomaly([(0.4, 1 / 3)], supporting_signal=False) == 0.3333

    def test_component_weights_sum_to_one(self):
        """Asserted because a score normalised to [0, 1] silently stops being
        normalised the moment one weight is retuned."""
        weights = [
            ANOMALY_WEIGHT_THRESHOLD.value,
            ANOMALY_WEIGHT_BASELINE.value,
            ANOMALY_WEIGHT_SLOPE.value,
            ANOMALY_WEIGHT_RUN.value,
        ]
        assert sum(weights) == 1.0


# --- Direction -----------------------------------------------------------------


class TestClassifyDirection:
    def test_no_slope_is_unknown_not_stable(self):
        assert classify_direction(SPO2, None) is TrendDirection.UNKNOWN

    def test_movement_inside_the_noise_band_is_stable(self):
        assert classify_direction(SPO2, -0.49) is TrendDirection.STABLE
        assert classify_direction(SPO2, 0.49) is TrendDirection.STABLE
        assert classify_direction(HEART_RATE, 2.9) is TrendDirection.STABLE

    def test_the_noise_band_is_inclusive_at_its_edge(self):
        assert classify_direction(SPO2, -0.5) is TrendDirection.WORSENING
        assert classify_direction(HEART_RATE, 3.0) is TrendDirection.WORSENING

    def test_a_flat_fit_is_stable(self):
        assert classify_direction(SPO2, 0.0) is TrendDirection.STABLE

    def test_direction_depends_on_the_metric_not_on_the_sign(self):
        assert classify_direction(SPO2, -2.0) is TrendDirection.WORSENING
        assert classify_direction(SPO2, 2.0) is TrendDirection.IMPROVING
        assert classify_direction(HEART_RATE, 8.1) is TrendDirection.WORSENING
        assert classify_direction(HEART_RATE, -8.1) is TrendDirection.IMPROVING


# --- Per-metric analysis -------------------------------------------------------


class TestAnalyseMetric:
    def test_a_metric_nobody_reported_is_none(self):
        assert analyse_metric([_sample(0, heart_rate=80)], SPO2) is None

    def test_a_lone_abnormal_reading_is_abnormal_but_not_deteriorating(self):
        """The reverse case this agent exists for. 91% is 91% and fires the
        absolute-threshold rule; with one observation there is no trajectory to
        claim, so no slope, no baseline and no deterioration."""
        a = analyse_metric([_sample(0, spo2=91)], SPO2)
        assert a is not None
        f = a.feature
        assert f.current == 91
        assert f.news2_score == 3
        assert f.absolute_threshold_breach is True
        assert f.threshold_value == 92
        assert f.baseline is None
        assert f.slope_per_day is None
        assert f.r_squared is None
        assert f.direction is TrendDirection.UNKNOWN
        assert f.consecutive_worsening_days == 0
        assert f.baseline_deviation_breach is False
        assert a.is_rapid is False
        assert [r.rule_id for r in a.fired_rules] == ["R-MON-SPO2-ABS-01"]
        # threshold 0.40 x 1.0 and run 0.15 x 0 are the only applicable
        # components: 0.40 / 0.55 = 0.7273.
        assert f.anomaly_score == pytest.approx(0.4 / 0.55, abs=1e-4)
        assert a.abnormality == "SpO2 91% — NEWS2 band <= 91%, parameter score 3"

    def test_two_samples_never_produce_a_slope(self):
        """`linear_trend` would happily fit two points and return r² = 1.0, which
        on a dashboard reads as a confident trend line."""
        a = analyse_metric([_sample(0, spo2=98), _sample(1, spo2=95)], SPO2)
        assert a is not None
        f = a.feature
        assert f.slope_per_day is None
        assert f.r_squared is None
        assert f.direction is TrendDirection.UNKNOWN
        assert a.is_trendable is False
        assert a.is_rapid is False
        # The point-in-time components still stand on their own.
        assert f.baseline == 98
        assert f.delta == -3
        assert f.baseline_deviation_breach is True
        assert f.news2_score == 1
        assert [r.rule_id for r in a.fired_rules] == [
            "R-MON-SPO2-ABS-01",
            "R-MON-SPO2-DELTA-01",
        ]
        assert any("below the 3 required" in g for g in a.gaps)

    def test_a_flat_normal_series_reports_nothing(self):
        a = analyse_metric(_series("spo2", (98, 98, 98, 98, 98)), SPO2)
        assert a is not None
        f = a.feature
        assert f.slope_per_day == 0.0, "measured and flat is not the same as unfittable"
        assert f.r_squared is None, "a constant series has no variance to explain"
        assert f.direction is TrendDirection.STABLE
        assert f.news2_score == 0
        assert f.absolute_threshold_breach is False
        assert f.anomaly_score == 0.0
        assert a.fired_rules == ()
        assert a.gaps == ()
        assert a.abnormality is None

    def test_a_flat_abnormal_series_is_stable_not_worsening(self):
        """The key reverse case: an unchanging 91% is an abnormality, not a
        deterioration, and must not set `rapid_deterioration`."""
        a = analyse_metric(_series("spo2", (91, 91, 91, 91, 91)), SPO2)
        assert a is not None
        f = a.feature
        assert f.news2_score == 3
        assert f.absolute_threshold_breach is True
        assert f.direction is TrendDirection.STABLE
        assert f.consecutive_worsening_days == 0
        assert f.baseline_deviation_breach is False
        assert f.anomaly_score == 0.4, "the threshold component alone"
        assert a.is_rapid is False
        assert [r.rule_id for r in a.fired_rules] == ["R-MON-SPO2-ABS-01"]

    def test_an_improving_series_fires_no_rules(self):
        a = analyse_metric(_series("spo2", (91, 93, 95, 97, 98)), SPO2)
        assert a is not None
        f = a.feature
        assert f.slope_per_day == pytest.approx(1.8)
        assert f.direction is TrendDirection.IMPROVING
        assert f.consecutive_worsening_days == 0
        assert f.anomaly_score == 0.0
        assert a.fired_rules == ()
        assert a.abnormality is None

    def test_a_steep_slope_through_scattered_points_is_flagged_as_a_poor_fit(self):
        a = analyse_metric(_series("spo2", (98, 93, 97, 92, 96)), SPO2)
        assert a is not None
        f = a.feature
        assert f.slope_per_day == pytest.approx(-0.5)
        assert f.r_squared == pytest.approx(0.0933, abs=1e-4)
        assert f.direction is TrendDirection.WORSENING
        assert a.is_rapid is False, "r² below the minimum is noise, not deterioration"
        assert [r.rule_id for r in a.fired_rules] == ["R-MON-SPO2-TREND-01"]
        assert any("poor fit" in g for g in a.gaps)

    def test_a_still_normal_reading_can_breach_the_personal_baseline(self):
        """95% is inside the normal band and would be unremarkable in isolation;
        in a patient whose own baseline was 98% it is not."""
        a = analyse_metric(_series("spo2", (98, 98, 97, 96, 95)), SPO2)
        assert a is not None
        f = a.feature
        assert f.news2_score == 1
        assert f.baseline == 98
        assert f.baseline_deviation_breach is True
        assert "personal baseline" in (a.abnormality or "")

    def test_the_cannot_show_deterioration_property_holds_for_the_baseline(self):
        """A series that ends at its own baseline has a delta of zero by
        construction, which is why the latest sample is excluded from it."""
        a = analyse_metric(_series("spo2", (98, 98, 91, 91, 91)), SPO2)
        assert a is not None
        assert a.feature.baseline == 98
        assert a.feature.delta == -7

    def test_gaps_name_the_metric_they_belong_to(self):
        a = analyse_metric([_sample(0, spo2=91)], SPO2)
        assert a is not None
        assert all(g.startswith("SpO2:") for g in a.gaps)
        assert all(g.endswith(".") for g in a.gaps)

    def test_sparse_coverage_is_reported_as_a_gap(self):
        a = analyse_metric([_sample(0, spo2=97), _sample(4, spo2=95), _sample(8, spo2=93)], SPO2)
        assert a is not None
        assert a.stats.coverage == pytest.approx(1 / 3)
        assert any("cover 33% of the 9-day window" in g for g in a.gaps)


# --- The rapid-deterioration gate ----------------------------------------------


class TestRapidDeteriorationGate:
    def test_all_four_conditions_met(self):
        assert _analysis().is_rapid is True

    @pytest.mark.parametrize("r_squared", [None, 0.0, 0.49])
    def test_a_poor_or_undefined_fit_is_not_deterioration(self, r_squared):
        assert _analysis(r_squared=r_squared).is_rapid is False

    def test_the_fit_requirement_is_inclusive_at_its_edge(self):
        assert _analysis(r_squared=MIN_R_SQUARED_FOR_TREND.value).is_rapid is True

    @pytest.mark.parametrize("run", [0, 1, 2])
    def test_a_short_run_is_a_jump_not_a_trajectory(self, run):
        assert _analysis(run=run).is_rapid is False

    def test_the_run_requirement_is_inclusive_at_its_edge(self):
        assert _analysis(run=int(RAPID_DETERIORATION_MIN_RUN.value)).is_rapid is True

    @pytest.mark.parametrize("count", [0, 1, 2])
    def test_too_few_samples_is_not_deterioration(self, count):
        """Unreachable through `analyse_metric`, which suppresses the slope below
        three samples; pinned because the gate is the last line of defence if
        that suppression is ever relaxed."""
        assert _analysis(count=count).is_rapid is False

    def test_no_slope_is_not_deterioration(self):
        assert _analysis(slope=None, direction=TrendDirection.UNKNOWN).is_rapid is False

    def test_a_stable_or_improving_metric_is_not_deterioration(self):
        assert _analysis(direction=TrendDirection.STABLE, slope=0.0).is_rapid is False
        assert _analysis(direction=TrendDirection.IMPROVING, slope=2.0).is_rapid is False

    def test_the_sample_floor_comes_from_the_trend_constant(self):
        assert int(MIN_SAMPLES_FOR_TREND.value) == 3


# --- Assessment-level behaviour ------------------------------------------------


class TestAssessMonitoring:
    def test_an_empty_window_states_the_absence(self):
        """Reporting normal vitals for a patient nobody measured would be the
        dangerous version of this outcome."""
        d = assess_monitoring([])
        a = d.assessment
        assert a.sample_count == 0
        assert a.window_days == 0
        assert a.first_recorded_at is None
        assert a.trends == []
        assert a.worst_metric is None
        assert a.current_abnormalities == []
        assert a.rapid_deterioration is False
        assert a.news2_partial_total is None
        assert a.data_quality == "insufficient"
        assert a.baseline_method == ""
        assert len(a.data_gaps) == len(VITAL_SPECS) + 1
        assert any(g.startswith("No monitoring samples at all") for g in a.data_gaps)
        assert [r.rule_id for r in d.fired_rules] == [QUALITY_RULE_ID]
        assert a.summary.startswith("No monitoring samples in the observation window")

    def test_an_entirely_normal_window_names_no_worst_metric(self):
        samples = [
            _sample(i, spo2=98, heart_rate=72, temperature_c=36.8, respiratory_rate=14,
                    sleep_hours=7.5, activity_steps=8000)
            for i in range(5)
        ]
        a = assess_monitoring(samples).assessment
        assert len(a.trends) == len(VITAL_SPECS)
        assert all(t.anomaly_score == 0.0 for t in a.trends)
        assert a.worst_metric is None, "naming a 'worst' out of an all-normal window invites misreading"
        assert a.data_quality == "complete"
        assert a.data_gaps == []
        assert a.news2_partial_total == 0, "a real total, not None: every parameter was scoreable"

    def test_a_tie_resolves_to_the_metric_declared_first(self):
        """`max` keeps the first of equals and `trends` is in `VitalSpec` order,
        so a tie is deterministic rather than dependent on dict ordering."""
        a = assess_monitoring([_sample(i, spo2=91, heart_rate=131) for i in range(5)]).assessment
        by_metric = {t.metric: t.anomaly_score for t in a.trends}
        assert by_metric[VitalMetric.SPO2] == by_metric[VitalMetric.HEART_RATE] == 0.4
        assert a.worst_metric is VitalMetric.SPO2

    def test_input_order_does_not_change_the_assessment(self):
        samples = _demo_samples()
        forward = assess_monitoring(samples).assessment.model_dump(mode="json")
        backward = assess_monitoring(list(reversed(samples))).assessment.model_dump(mode="json")
        assert forward == backward

    def test_metrics_nobody_reported_are_listed_as_gaps(self):
        a = assess_monitoring(_series("spo2", (98, 97, 96, 95, 94))).assessment
        assert [t.metric for t in a.trends] == [VitalMetric.SPO2]
        assert len(a.data_gaps) == len(VITAL_SPECS) - 1
        assert any(g.startswith("Heart rate: no observations") for g in a.data_gaps)
        assert a.data_quality == "partial"

    def test_a_hole_in_the_window_makes_it_sparse_not_partial(self):
        """`insufficient` says the agent had nothing; `sparse` says it had
        something and the window had holes in it."""
        a = assess_monitoring([_sample(0, spo2=97), _sample(4, spo2=95), _sample(8, spo2=93)]).assessment
        assert a.window_days == 9
        assert a.sample_count == 3
        assert a.data_quality == "sparse"
        assert any("cover 33%" in g for g in a.data_gaps)

    def test_one_short_metric_makes_the_window_partial(self):
        full = dict(heart_rate=72, temperature_c=36.8, respiratory_rate=14,
                    sleep_hours=7.2, activity_steps=8200)
        samples = [_sample(0, spo2=98, **full), _sample(1, spo2=97, **full)]
        samples += [_sample(i, **full) for i in (2, 3, 4)]
        a = assess_monitoring(samples).assessment
        assert a.data_quality == "partial"
        trendable = {t.metric: t.slope_per_day is not None for t in a.trends}
        assert trendable[VitalMetric.SPO2] is False
        assert all(v for m, v in trendable.items() if m is not VitalMetric.SPO2)

    def test_the_caveat_ships_with_the_total(self):
        """A partial NEWS2 total rendered as though it were a full one is the
        specific misreading the schema validator prevents."""
        a = assess_monitoring(_demo_samples()).assessment
        assert a.news2_caveat == NEWS2_PARTIAL_SCORE_CAVEAT
        assert "NOT a NEWS2 total" in a.news2_caveat

    def test_window_days_counts_elapsed_days_plus_one(self):
        a = assess_monitoring(_demo_samples()).assessment
        assert a.window_days == 5
        assert a.first_recorded_at == _day(0)
        assert a.last_recorded_at == _day(4)

    def test_the_baseline_method_is_stated(self):
        a = assess_monitoring(_demo_samples()).assessment
        assert "mean of the earliest 2 samples" in a.baseline_method
        assert "excluded" in a.baseline_method, "the current value must not be part of its own baseline"


# --- The demo scenario ----------------------------------------------------------


class TestDemoScenario:
    """The premises the project spec states for the demo patient.

    SpO2 98 -> 91 and heart rate 72 -> 103 over five days must yield rapid
    deterioration with SpO2 as the worst metric. If a fixture edit or a threshold
    change silently breaks one of these, the longitudinal story the dashboard
    tells stops being the story that was specified.
    """

    def test_the_hand_built_series_matches_the_shipped_fixture(self, demo_monitoring):
        assert [s.recorded_at for s in demo_monitoring] == [_day(i) for i in range(5)]
        for i, sample in enumerate(demo_monitoring):
            for field, values in DEMO_SERIES.items():
                assert getattr(sample, field) == values[i], f"{field} on day {i}"

    def test_the_window_is_the_five_specified_days(self, demo_monitoring):
        a = assess_monitoring(demo_monitoring).assessment
        assert a.sample_count == 5
        assert a.window_days == 5
        assert len(a.trends) == 6
        assert a.data_quality == "complete"
        assert a.data_gaps == []

    def test_every_vital_falls_over_five_days(self, demo_monitoring):
        a = assess_monitoring(demo_monitoring).assessment
        slopes = {t.metric: t.slope_per_day for t in a.trends}
        assert slopes[VitalMetric.SPO2] == pytest.approx(-1.8)
        assert slopes[VitalMetric.HEART_RATE] == pytest.approx(8.1)
        assert slopes[VitalMetric.TEMPERATURE_C] == pytest.approx(0.49)
        assert slopes[VitalMetric.RESPIRATORY_RATE] == pytest.approx(2.5)
        assert slopes[VitalMetric.SLEEP_HOURS] == pytest.approx(-0.77)
        assert slopes[VitalMetric.ACTIVITY_STEPS] == pytest.approx(-1980.0)
        assert all(t.direction is TrendDirection.WORSENING for t in a.trends)
        assert all(t.consecutive_worsening_days == 4 for t in a.trends)

    def test_goodness_of_fit_is_hand_computed(self, demo_monitoring):
        """r² = Sxy² / (Sxx · Syy) over each five-point series, with Sxx = 10 for
        daily samples spanning four days."""
        fits = {t.metric: t.r_squared for t in assess_monitoring(demo_monitoring).assessment.trends}
        assert fits[VitalMetric.SPO2] == pytest.approx(324 / 328)
        assert fits[VitalMetric.HEART_RATE] == pytest.approx(6561 / 6828)
        assert fits[VitalMetric.TEMPERATURE_C] == pytest.approx(2401 / 2500)
        assert fits[VitalMetric.RESPIRATORY_RATE] == pytest.approx(625 / 632)
        assert fits[VitalMetric.SLEEP_HOURS] == pytest.approx(5929 / 6012)
        assert fits[VitalMetric.ACTIVITY_STEPS] == pytest.approx(1089 / 1117)

    def test_the_personal_baseline_excludes_the_current_value(self, demo_monitoring):
        trends = {t.metric: t for t in assess_monitoring(demo_monitoring).assessment.trends}
        assert trends[VitalMetric.SPO2].baseline == 97.5, "mean of 98 and 97, not of all five"
        assert trends[VitalMetric.HEART_RATE].baseline == 73.5
        assert trends[VitalMetric.SPO2].delta == -6.5
        assert trends[VitalMetric.SPO2].delta_pct == pytest.approx(-6.5 / 97.5 * 100)
        assert all(t.baseline_deviation_breach for t in trends.values())

    def test_the_partial_news2_total_and_its_trigger(self, demo_monitoring):
        """3 (SpO2 91) + 1 (HR 103) + 1 (temp 38.7) + 2 (RR 24) = 7. Sleep and
        activity are not NEWS2 parameters and contribute nothing, not zero."""
        a = assess_monitoring(demo_monitoring).assessment
        assert a.news2_partial_total == 7
        assert a.news2_any_parameter_trigger is True

    def test_rapid_deterioration_is_set_for_all_six_metrics(self, demo_monitoring):
        a = assess_monitoring(demo_monitoring).assessment
        assert a.rapid_deterioration is True
        assert a.deteriorating_metrics == list(VitalMetric)

    def test_the_anomaly_scores_rank_oxygenation_first(self, demo_monitoring):
        trends = {t.metric: t for t in assess_monitoring(demo_monitoring).assessment.trends}
        assert trends[VitalMetric.SPO2].anomaly_score == pytest.approx(0.9676, abs=1e-4)
        assert trends[VitalMetric.RESPIRATORY_RATE].anomaly_score == pytest.approx(0.8345, abs=1e-4)
        assert trends[VitalMetric.HEART_RATE].anomaly_score == pytest.approx(0.6955, abs=1e-4)
        assert trends[VitalMetric.TEMPERATURE_C].anomaly_score == pytest.approx(0.6954, abs=1e-4)
        assert trends[VitalMetric.SLEEP_HOURS].anomaly_score == pytest.approx(0.5672, abs=1e-4)
        assert trends[VitalMetric.ACTIVITY_STEPS].anomaly_score == pytest.approx(0.565, abs=1e-4)
        worst = assess_monitoring(demo_monitoring).assessment.worst_metric
        assert worst is VitalMetric.SPO2

    def test_the_wearable_signals_never_outrank_a_clinical_measurement(self, demo_monitoring):
        """Sleep fell 40% and activity 92%, both dramatic; `SUPPORTING_SIGNAL_WEIGHT`
        keeps them below every clinical measurement in this window."""
        trends = {t.metric: t.anomaly_score for t in assess_monitoring(demo_monitoring).assessment.trends}
        clinical = [trends[m] for m in (VitalMetric.SPO2, VitalMetric.HEART_RATE,
                                        VitalMetric.TEMPERATURE_C, VitalMetric.RESPIRATORY_RATE)]
        wearable = [trends[VitalMetric.SLEEP_HOURS], trends[VitalMetric.ACTIVITY_STEPS]]
        assert max(wearable) < min(clinical)

    def test_every_metric_is_abnormal_right_now(self, demo_monitoring):
        a = assess_monitoring(demo_monitoring).assessment
        assert len(a.current_abnormalities) == 6
        assert a.current_abnormalities[0] == (
            "SpO2 91% — NEWS2 band <= 91%, parameter score 3; "
            "6.5 percentage points beyond a personal baseline of 97.5%"
        )
        assert a.current_abnormalities[-1] == (
            "Activity 600 steps — 91.6% below a personal baseline of 7150 steps"
        )

    def test_abnormalities_describe_the_present_not_the_trajectory(self):
        """The trend belongs to `direction` and `rapid_deterioration`; folding it
        in here would make this list a claim about the future."""
        for text in assess_monitoring(_demo_samples()).assessment.current_abnormalities:
            lowered = text.lower()
            assert "worsening" not in lowered
            assert "/day" not in lowered
            assert "slope" not in lowered

    def test_the_summary_names_the_worst_metric_and_the_trajectory(self, demo_monitoring):
        summary = assess_monitoring(demo_monitoring).assessment.summary
        assert "5 samples over 5 days" in summary
        assert "from 2026-09-04 to 2026-09-08" in summary
        assert "Most abnormal: SpO2 91%" in summary
        assert "Rapid deterioration in 6 metrics" in summary
        assert "Partial NEWS2 total 7" in summary
        assert "Data quality: complete." in summary

    def test_the_summary_counts_abnormal_metrics_instead_of_restating_them(self):
        """Regression: an earlier version concatenated all six abnormality
        strings, making the summary a second copy of the list it summarises."""
        summary = assess_monitoring(_demo_samples()).assessment.summary
        assert "6 of 6 monitored metrics are abnormal right now" in summary
        assert summary.count("NEWS2 parameter score") == 1, "only the worst metric is described"

    def test_a_percent_is_never_separated_from_its_number(self):
        """Regression: '91 %' came out of an earlier formatting pass, and SpO2 is
        the metric the whole demo turns on, so it would have landed in nearly
        every string the dashboard renders."""
        a = assess_monitoring(_demo_samples()).assessment
        assert " %" not in a.summary
        assert all(" %" not in text for text in a.current_abnormalities)
        assert "SpO2 91%" in a.current_abnormalities[0]
        assert "-1.8%/day" in a.summary

    def test_a_redundant_decimal_is_never_rendered(self, demo_monitoring):
        """Regression: 'Activity is worsening at 1980.0 steps/day' and '0.5 C/day'
        both came from a fixed-precision format applied to magnitudes that did not
        need one."""
        rules = {r.rule_id: r for r in assess_monitoring(demo_monitoring).fired_rules}
        assert rules["R-MON-STEPS-TREND-01"].description == (
            "Activity is worsening at 1980 steps/day over 5 days"
        )
        assert rules["R-MON-TEMP-TREND-01"].description == (
            "Temperature is worsening at 0.49 C/day over 5 days"
        )

    def test_a_trend_description_gives_magnitude_and_its_evidence_gives_the_sign(self, demo_monitoring):
        """'Worsening at 1.8%/day' is a magnitude with the direction in words, so
        the signed number has to be one click away or a falling saturation reads
        as a rising one."""
        rule = next(
            r for r in assess_monitoring(demo_monitoring).fired_rules
            if r.rule_id == "R-MON-SPO2-TREND-01"
        )
        assert rule.description == "SpO2 is worsening at 1.8%/day over 5 days"
        assert "slope -1.8%/day" in rule.evidence
        assert rule.value == pytest.approx(-1.8)


# --- Fired rules ----------------------------------------------------------------


class TestFiredRuleConventions:
    def test_the_demo_fires_eighteen_rules(self, demo_monitoring):
        rules = assess_monitoring(demo_monitoring).fired_rules
        assert len(rules) == 18
        assert len({r.rule_id for r in rules}) == 18, "rule ids must be unique"

    def test_rule_ids_follow_the_documented_scheme(self, demo_monitoring):
        assert [r.rule_id for r in assess_monitoring(demo_monitoring).fired_rules] == [
            "R-MON-SPO2-ABS-01", "R-MON-SPO2-DELTA-01", "R-MON-SPO2-TREND-01",
            "R-MON-HR-ABS-01", "R-MON-HR-DELTA-01", "R-MON-HR-TREND-01",
            "R-MON-TEMP-ABS-01", "R-MON-TEMP-DELTA-01", "R-MON-TEMP-TREND-01",
            "R-MON-RR-ABS-01", "R-MON-RR-DELTA-01", "R-MON-RR-TREND-01",
            "R-MON-SLEEP-DELTA-01", "R-MON-SLEEP-TREND-01",
            "R-MON-STEPS-DELTA-01", "R-MON-STEPS-TREND-01",
            RAPID_RULE_ID, NEWS2_TRIGGER_RULE_ID,
        ]

    def test_no_quality_rule_when_the_window_is_complete(self, demo_monitoring):
        assert QUALITY_RULE_ID not in [r.rule_id for r in assess_monitoring(demo_monitoring).fired_rules]

    def test_no_rule_assigns_risk_points(self, demo_monitoring):
        """The Coordinator's risk scorer owns the mapping onto the monitoring
        band; a second set of numbers here would be a competing answer to 'how
        much did this add?'."""
        for rule in assess_monitoring(demo_monitoring).fired_rules:
            assert rule.contribution == 0.0
            assert rule.category == MONITORING_RULE_CATEGORY

    def test_every_rule_carries_evidence_and_a_value(self, demo_monitoring):
        """A rule with no evidence is an assertion. These strings are rendered
        verbatim in the dashboard's 'why this judgement' panel."""
        for rule in assess_monitoring(demo_monitoring).fired_rules:
            assert rule.description, rule.rule_id
            assert rule.evidence, rule.rule_id
            assert rule.value is not None, rule.rule_id
            assert rule.threshold is not None, rule.rule_id
            assert any(c.isdigit() for c in rule.evidence), rule.rule_id

    def test_absolute_rules_quote_the_published_cutoff_and_its_source(self, demo_monitoring):
        by_id = {r.rule_id: r for r in assess_monitoring(demo_monitoring).fired_rules}
        cutoffs = {
            "R-MON-SPO2-ABS-01": 92.0,
            "R-MON-HR-ABS-01": 91.0,
            "R-MON-TEMP-ABS-01": 38.1,
            "R-MON-RR-ABS-01": 21.0,
        }
        for rule_id, cutoff in cutoffs.items():
            assert by_id[rule_id].threshold == cutoff
            assert NEWS2_SOURCE in by_id[rule_id].evidence

    def test_the_cutoff_is_the_boundary_toward_normal_not_the_band_edge(self, demo_monitoring):
        """SpO2 91 sits in '<= 91%', whose upper bound is 92 — the number a
        reader compares the observation against. For a rising-adverse metric the
        same idea picks the lower bound."""
        by_id = {r.rule_id: r for r in assess_monitoring(demo_monitoring).fired_rules}
        assert by_id["R-MON-SPO2-ABS-01"].threshold == 92.0
        assert by_id["R-MON-HR-ABS-01"].threshold == 91.0, "103 bpm crossed 91, not 111"

    def test_trend_rules_carry_a_signed_threshold(self, demo_monitoring):
        """The threshold is the metric's own noise band with the adverse sign
        applied, so a falling saturation and a rising pulse read consistently."""
        by_id = {r.rule_id: r for r in assess_monitoring(demo_monitoring).fired_rules}
        assert by_id["R-MON-SPO2-TREND-01"].threshold == -0.5
        assert by_id["R-MON-SPO2-TREND-01"].value == pytest.approx(-1.8)
        assert by_id["R-MON-HR-TREND-01"].threshold == 3.0
        assert by_id["R-MON-STEPS-TREND-01"].threshold == -500.0

    def test_baseline_rules_quote_the_first_threshold_crossed(self, demo_monitoring):
        by_id = {r.rule_id: r for r in assess_monitoring(demo_monitoring).fired_rules}
        assert by_id["R-MON-SPO2-DELTA-01"].threshold == 3.0
        assert by_id["R-MON-SPO2-DELTA-01"].value == 91.0
        assert by_id["R-MON-STEPS-DELTA-01"].threshold == 50.0
        for rule_id in ("R-MON-SPO2-DELTA-01", "R-MON-HR-DELTA-01", "R-MON-STEPS-DELTA-01"):
            assert "personal baseline" in by_id[rule_id].evidence

    def test_the_rapid_rule_lists_every_metric_it_covers(self, demo_monitoring):
        rule = next(r for r in assess_monitoring(demo_monitoring).fired_rules if r.rule_id == RAPID_RULE_ID)
        assert rule.value == 6.0
        assert rule.threshold == RAPID_DETERIORATION_MIN_RUN.value
        assert "Rapid deterioration" in rule.description
        for label in ("SpO2", "Heart rate", "Temperature", "Respiratory rate", "Sleep", "Activity"):
            assert label in rule.evidence

    def test_the_trigger_rule_explains_why_it_is_used_instead_of_the_total(self, demo_monitoring):
        rule = next(
            r for r in assess_monitoring(demo_monitoring).fired_rules if r.rule_id == NEWS2_TRIGGER_RULE_ID
        )
        assert rule.value == 1.0, "one parameter is at the escalation score"
        assert rule.threshold == 3.0
        assert "SpO2 91%" in rule.description
        assert NEWS2_PARTIAL_SCORE_CAVEAT in rule.evidence

    def test_the_quality_rule_appears_when_the_window_is_not_complete(self):
        rule = next(r for r in assess_monitoring([]).fired_rules if r.rule_id == QUALITY_RULE_ID)
        assert rule.description == "Monitoring data quality: insufficient"
        assert rule.value == 0.0
        assert rule.threshold == 6.0

    def test_every_rule_passes_the_safety_guard(self, demo_monitoring):
        """Rule text is rendered in the dashboard's evidence panel, so it is
        generated prose and is audited like any other."""
        for rule in assess_monitoring(demo_monitoring).fired_rules:
            fields = {"description": rule.description, "evidence": rule.evidence}
            assert audit_prose(fields) == [], rule.rule_id


# --- The agent -------------------------------------------------------------------


class TestRunMonitoringAgent:
    def test_the_agent_adds_nothing_to_the_engine_result(self, session, demo_monitoring):
        """The engine owns every judgement; the agent owns I/O. If these two ever
        differ, something in the agent is second-guessing the rules."""
        assessment, trace = run_monitoring(session, DEMO_ID)
        assert trace.status.value == "ok"
        assert assessment is not None
        expected = assess_monitoring(demo_monitoring).assessment
        assert assessment.model_dump(mode="json") == expected.model_dump(mode="json")

    def test_the_trace_output_round_trips_through_the_contract(self, session):
        assessment, trace = run_monitoring(session, DEMO_ID)
        assert assessment is not None
        restored = MonitoringAssessment.model_validate(trace.output)
        assert restored == assessment

    def test_the_trace_output_is_json_serialisable(self, session):
        """It goes over the wire to the dashboard, so enums and datetimes must
        already be primitives by the time it lands in the trace."""
        _, trace = run_monitoring(session, DEMO_ID)
        assert json.loads(json.dumps(trace.output))["worst_metric"] == "spo2"

    def test_trace_notes_are_parseable_key_value_pairs(self, session):
        _, trace = run_monitoring(session, DEMO_ID)
        parsed = [n for n in trace.notes if "=" in n and not n.startswith("data_gap")]
        notes = dict(n.split("=", 1) for n in parsed)

        assert notes["reference_date"] == "2026-09-08"
        assert notes["max_window_days"] == "all"
        assert notes["sample_count"] == "5"
        assert notes["window_days"] == "5"
        assert notes["metrics_reported"] == "6"
        assert notes["data_quality"] == "complete"
        assert notes["news2_partial_total"] == "7"
        assert notes["news2_any_parameter_trigger"] == "true"
        assert notes["rapid_deterioration"] == "true"
        assert notes["deteriorating_metrics"] == (
            "spo2,heart_rate,temperature_c,respiratory_rate,sleep_hours,activity_steps"
        )
        assert notes["worst_metric"] == "spo2"

        assert all("(" not in v and " from " not in v for v in notes.values())
        assert any("not the wall clock" in n for n in trace.notes), (
            "the rationale for the anchor survives as its own prose note"
        )

    def test_gaps_are_echoed_into_the_notes(self, session):
        _, trace = run_monitoring(session, DEMO_ID, reference_date=date(2026, 9, 4))
        gap_notes = [n for n in trace.notes if n.startswith("data_gap:")]
        assert len(gap_notes) == 6
        assert all(":" in n.removeprefix("data_gap: ") for n in gap_notes)

    def test_the_agent_reports_it_used_no_llm(self, session):
        _, trace = run_monitoring(session, DEMO_ID)
        assert trace.llm_used is False
        assert trace.agent_name.value == "monitoring"
        assert trace.duration_ms >= 0
        assert trace.started_at <= trace.finished_at

    def test_repeated_runs_are_identical(self, session):
        first, _ = run_monitoring(session, DEMO_ID)
        second, _ = run_monitoring(session, DEMO_ID)
        assert first is not None and second is not None
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_the_agent_writes_nothing(self, session):
        """Nothing to flush or commit: the assessment rides out in the trace and
        the workflow persists it with the rest of the run's results."""
        before = session.scalar(
            select(func.count()).select_from(MonitoringRecordRow).where(
                MonitoringRecordRow.patient_id == DEMO_ID
            )
        )
        run_monitoring(session, DEMO_ID)
        after = session.scalar(
            select(func.count()).select_from(MonitoringRecordRow).where(
                MonitoringRecordRow.patient_id == DEMO_ID
            )
        )
        assert before == after == 5

    def test_an_unknown_patient_is_captured_in_the_trace_not_raised(self, session):
        """The workflow fans out to six agents; one failing must not take the
        others down."""
        assessment, trace = run_monitoring(session, "PT-DOES-NOT-EXIST")
        assert assessment is None
        assert trace.status.value == "error"
        assert "UnknownPatientError" in (trace.error or "")
        assert trace.agent_name.value == "monitoring"

    def test_the_session_is_usable_after_a_failure(self, session):
        _, failed = run_monitoring(session, "PT-DOES-NOT-EXIST")
        assert failed.status.value == "error"
        assessment, ok = run_monitoring(session, DEMO_ID)
        assert ok.status.value == "ok"
        assert assessment is not None

    def test_a_patient_with_no_observations_is_ok_not_an_error(self, session):
        """An empty window is a data gap, not a failure. Failing here would hide
        the absence behind an error badge."""
        session.add(
            PatientRow(id="PT-EMPTY", full_name="No Data", birth_year=1980,
                       sex="female", created_at=utcnow())
        )
        session.flush()

        assessment, trace = run_monitoring(session, "PT-EMPTY")
        assert trace.status.value == "ok"
        assert assessment is not None
        assert assessment.sample_count == 0
        assert assessment.data_quality == "insufficient"
        assert len(assessment.data_gaps) == 7
        assert assessment.worst_metric is None
        assert trace.error is None

    def test_generated_prose_passes_the_safety_guard(self, session):
        assessment, _ = run_monitoring(session, DEMO_ID)
        assert assessment is not None
        fields = {
            "monitoring_summary": assessment.summary,
            "current_abnormalities": assessment.current_abnormalities,
            "data_gaps": assessment.data_gaps,
        }
        assert audit_prose(fields) == []

    def test_the_summary_states_deterioration_without_asserting_a_diagnosis(self, session):
        assessment, _ = run_monitoring(session, DEMO_ID)
        assert assessment is not None
        lowered = assessment.summary.lower()
        for forbidden in ("diagnosis is", "diagnosed with", "confirmed diagnosis", "pneumonia"):
            assert forbidden not in lowered


class TestReplayAndWindowing:
    def test_the_anchor_comes_from_the_patients_own_data(self, session):
        _, anchor = load_samples(session, DEMO_ID)
        assert anchor == ANCHOR, "the latest observation, not the wall clock"

    def test_an_explicit_anchor_wins(self, session):
        _, anchor = load_samples(session, DEMO_ID, reference_date=date(2026, 9, 6))
        assert anchor == date(2026, 9, 6)

    def test_observations_after_the_anchor_are_excluded(self, session):
        """Information leakage guard. Phase 10 replays the patient at an earlier
        date to ask what the system would have said then; a later reading must
        not feed that answer, and least of all the baseline."""
        samples, _ = load_samples(session, DEMO_ID, reference_date=date(2026, 9, 6))
        assert [s.recorded_at.date() for s in samples] == [
            date(2026, 9, 4), date(2026, 9, 5), date(2026, 9, 6)
        ]

    def test_replaying_two_days_earlier_changes_the_story(self, session):
        """On 09-06 the saturation has fallen to 95% and no metric yet combines
        an adverse slope with a sustained run: three samples cannot produce a run
        of three transitions. The flag the demo turns on is genuinely a product
        of the full five days, not of the fixture's last row."""
        assessment, trace = run_monitoring(session, DEMO_ID, reference_date=date(2026, 9, 6))
        assert trace.status.value == "ok"
        assert assessment is not None
        assert assessment.sample_count == 3
        assert assessment.rapid_deterioration is False
        assert assessment.deteriorating_metrics == []
        assert assessment.news2_any_parameter_trigger is False
        assert assessment.data_quality == "complete"

        spo2 = next(t for t in assessment.trends if t.metric is VitalMetric.SPO2)
        assert spo2.current == 95
        assert spo2.baseline == 97.5
        assert spo2.delta == -2.5
        assert spo2.baseline_deviation_breach is False, "2.5 points is below the 3-point threshold"
        assert spo2.absolute_threshold_breach is True
        assert spo2.news2_score == 1
        assert spo2.consecutive_worsening_days == 2
        assert spo2.slope_per_day == pytest.approx(-1.5)

    def test_an_earlier_anchor_can_change_which_metric_is_worst(self, session):
        """At three days the anomaly score renormalises over fewer components and
        respiratory rate — a slope of 2.0/min/day against a 0.5 noise band with a
        perfect fit — outranks a saturation that has only fallen 2.5 points.
        Pinned because 'SpO2 is always the worst metric' would be a convenient
        thing to assume and is not true of the trajectory."""
        assessment, _ = run_monitoring(session, DEMO_ID, reference_date=date(2026, 9, 6))
        assert assessment is not None
        assert assessment.worst_metric is VitalMetric.RESPIRATORY_RATE

    def test_two_days_of_data_is_not_enough_for_a_trend(self, session):
        assessment, _ = run_monitoring(session, DEMO_ID, reference_date=date(2026, 9, 5))
        assert assessment is not None
        assert assessment.sample_count == 2
        assert assessment.data_quality == "insufficient"
        assert all(t.slope_per_day is None for t in assessment.trends)
        assert all(t.direction is TrendDirection.UNKNOWN for t in assessment.trends)
        assert len(assessment.data_gaps) == 6
        assert all("below the 3 required" in g for g in assessment.data_gaps)

    def test_a_single_day_of_data_still_scores_the_published_instrument(self, session):
        """The point-in-time component needs no history at all, so day one is not
        a blank page — but the total is 0, a real total, not None."""
        assessment, _ = run_monitoring(session, DEMO_ID, reference_date=date(2026, 9, 4))
        assert assessment is not None
        assert assessment.sample_count == 1
        assert assessment.news2_partial_total == 0
        assert assessment.news2_partial_total is not None
        assert assessment.worst_metric is None
        assert all(t.baseline is None for t in assessment.trends)
        assert assessment.data_quality == "insufficient"

    def test_the_cap_counts_the_anchor_day_as_the_last_one(self, session):
        samples, _ = load_samples(session, DEMO_ID, max_window_days=3)
        assert [s.recorded_at.date() for s in samples] == [
            date(2026, 9, 6), date(2026, 9, 7), date(2026, 9, 8)
        ]

    def test_the_cap_anchors_to_the_reference_date_not_to_the_latest_observation(self, session):
        samples, anchor = load_samples(
            session, DEMO_ID, reference_date=date(2026, 9, 7), max_window_days=2
        )
        assert anchor == date(2026, 9, 7)
        assert [s.recorded_at.date() for s in samples] == [date(2026, 9, 6), date(2026, 9, 7)]

    def test_a_three_day_cap_still_shows_a_baseline_breach_at_exactly_the_threshold(self, session):
        assessment, _ = run_monitoring(session, DEMO_ID, max_window_days=3)
        assert assessment is not None
        spo2 = next(t for t in assessment.trends if t.metric is VitalMetric.SPO2)
        assert spo2.baseline == 94, "mean of 95 and 93, the two earliest in the capped window"
        assert spo2.delta == -3
        assert spo2.baseline_deviation_breach is True, "the threshold is >= 3, so exactly 3 crosses"
        assert spo2.consecutive_worsening_days == 2
        assert assessment.rapid_deterioration is False

    def test_the_cap_filters_rather_than_pads_the_window(self, session):
        """A three-day series inside a ten-day cap reports three days, not ten."""
        assessment, _ = run_monitoring(session, DEMO_ID, max_window_days=10)
        assert assessment is not None
        assert assessment.sample_count == 5
        assert assessment.window_days == 5

    def test_a_one_day_cap_reduces_to_point_in_time_scoring(self, session):
        assessment, trace = run_monitoring(session, DEMO_ID, max_window_days=1)
        assert assessment is not None
        assert assessment.sample_count == 1
        assert all(t.sample_count == 1 for t in assessment.trends)
        assert assessment.news2_partial_total == 7, "the last day's readings, unchanged"
        assert assessment.news2_any_parameter_trigger is True
        assert assessment.rapid_deterioration is False
        assert assessment.data_quality == "insufficient"
        assert QUALITY_RULE_ID in [r.rule_id for r in trace.fired_rules]

    def test_a_cap_below_one_day_is_an_error_not_an_empty_window(self, session):
        """`max_window_days=0` is a caller mistake, not a data condition, and
        silently returning an empty window would report it as one."""
        with pytest.raises(ValueError, match="max_window_days must be >= 1"):
            load_samples(session, DEMO_ID, max_window_days=0)

        assessment, trace = run_monitoring(session, DEMO_ID, max_window_days=0)
        assert assessment is None
        assert trace.status.value == "error"
        assert "ValueError" in (trace.error or "")


class TestSafetyGuardEnforcement:
    @pytest.mark.parametrize("field", ["summary", "current_abnormalities", "data_gaps"])
    def test_unsafe_prose_in_any_audited_field_fails_the_agent(self, session, monkeypatch, field):
        """The guard is structural, not advisory: if generated prose ever asserted
        a diagnosis, the agent must fail rather than publish it. Parametrised
        over all three fields the agent hands to `audit_prose`, so dropping one
        from that call is a test failure."""
        import app.agents.monitoring as monitoring_module

        real = assess_monitoring(load_samples(session, DEMO_ID)[0])
        unsafe = {"summary": "The diagnosis is pneumonia.",
                  "current_abnormalities": ["The patient has pneumonia."],
                  "data_gaps": ["You have been diagnosed with tuberculosis."]}
        patched = real.assessment.model_copy(update={field: unsafe[field]})
        monkeypatch.setattr(
            monitoring_module, "assess_monitoring",
            lambda samples: MonitoringDerivation(assessment=patched, fired_rules=()),
        )

        assessment, trace = run_monitoring(session, DEMO_ID)
        assert assessment is None
        assert trace.status.value == "error"
        assert "asserts_diagnosis" in (trace.error or "")

    def test_every_string_the_dashboard_can_render_passes_the_guard(self, session):
        """Wider than the agent's own call, which audits only the summary, the
        abnormalities and the gaps. Rule text is rendered in the evidence panel
        too, and it quotes `Threshold.source` strings written by hand in
        `thresholds.py`, so this is the only thing standing between one of those
        and diagnostic wording reaching the UI."""
        assessment, trace = run_monitoring(session, DEMO_ID)
        assert trace.status.value == "ok"
        assert assessment is not None
        fields = {
            "monitoring_summary": assessment.summary,
            "current_abnormalities": assessment.current_abnormalities,
            "data_gaps": assessment.data_gaps,
            "rule_descriptions": [r.description for r in trace.fired_rules],
            "rule_evidence": [r.evidence for r in trace.fired_rules],
        }
        assert audit_prose(fields) == []
