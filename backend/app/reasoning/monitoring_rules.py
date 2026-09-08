"""Monitoring Agent rule engine.

Combines the trend maths in `trends.py` with the cutoffs in `thresholds.py` to
produce `TrendFeature`s, one `MonitoringAssessment`, and the `FiredRule` audit
trail behind both. Pure functions over `VitalSample`s — no database, no LLM, no
wall clock — so the same rules apply to the seeded demo patient, to a CSV a user
uploads, and to the replayed windows Phase 10 evaluates against.

The governing constraint is the one the project brief sets for this agent:
**never conclude from a single time point.** Three independent components are
computed for every vital — the current value against a published instrument, an
adverse temporal trend, and deviation from the patient's own baseline — and the
flag that escalates (`rapid_deterioration`) needs a fitted slope *and* a
sustained monotonic run *and* an adequate goodness of fit. A lone abnormal
reading still yields a `TrendFeature` and still fires the absolute-threshold
rule, because 91% is 91%; what it does not yield is a claim about where the
patient is heading.

Two limitations are stated in the output rather than hidden:

* The NEWS2 total is PARTIAL — four of its seven parameters are recorded. The
  per-parameter scores and the "3 in any one parameter" trigger remain valid and
  are what these rules escalate on. The total ships with its caveat attached by
  a validator on `MonitoringAssessment`, so it cannot be rendered bare.
* No rule here assigns risk points. `contribution` is 0.0 throughout because the
  Coordinator's risk scorer owns the mapping from these findings onto the
  monitoring band, and a second set of numbers here would be a competing answer
  to "how much did this add?". What these rules carry is evidence: the observed
  value, its unit, the cutoff it crossed, and where the cutoff came from.

Prose is generated here rather than in the agent, unlike the History Agent's
summary. That one quotes the raw record — names, dates, medication doses — which
only the agent's `PatientRecord` holds; this one quotes only fields the rules
already derived, and threading fifteen of them back into a template in the agent
would be a second copy of the same derivation. The agent still enforces the
safety guard over the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from app.reasoning.thresholds import (
    ACTIVITY_ADVERSE_SLOPE,
    ACTIVITY_BASELINE_DROP_PCT,
    ANOMALY_RUN_FULL_SCORE,
    ANOMALY_WEIGHT_BASELINE,
    ANOMALY_WEIGHT_RUN,
    ANOMALY_WEIGHT_SLOPE,
    ANOMALY_WEIGHT_THRESHOLD,
    BASELINE_WINDOW_SAMPLES,
    EXPECTED_SAMPLES_PER_DAY,
    HEART_RATE_ADVERSE_SLOPE,
    HEART_RATE_BASELINE_RISE,
    HEART_RATE_BASELINE_RISE_PCT,
    MIN_R_SQUARED_FOR_TREND,
    MIN_SAMPLES_FOR_TREND,
    NEWS2_ANY_PARAMETER_TRIGGER,
    NEWS2_PARTIAL_SCORE_CAVEAT,
    NEWS2_SOURCE,
    RAPID_DETERIORATION_MIN_RUN,
    RESPIRATORY_RATE_ADVERSE_SLOPE,
    RESPIRATORY_RATE_BASELINE_RISE,
    SLEEP_ADVERSE_SLOPE,
    SLEEP_BASELINE_DROP,
    SPARSE_COVERAGE_FRACTION,
    SPO2_ADVERSE_SLOPE,
    SPO2_BASELINE_DROP,
    SPO2_BASELINE_DROP_PCT,
    SUPPORTING_SIGNAL_WEIGHT,
    TEMPERATURE_ADVERSE_SLOPE,
    TEMPERATURE_BASELINE_RISE,
    ScoreBand,
    Threshold,
    news2_band,
    news2_max_parameter_score,
)
from app.reasoning.trends import SECONDS_PER_DAY, SeriesStats, consecutive_adverse_run, describe_series
from app.schemas import (
    ClinicalDimension,
    FiredRule,
    MonitoringAssessment,
    TrendDirection,
    TrendFeature,
    VitalMetric,
    VitalSample,
)

MONITORING_RULE_CATEGORY = "monitoring"

RAPID_RULE_ID = "R-MON-RAPID-01"
NEWS2_TRIGGER_RULE_ID = "R-MON-NEWS2-TRIGGER-01"
QUALITY_RULE_ID = "R-MON-QUALITY-01"

MIN_TREND_SAMPLES = int(MIN_SAMPLES_FOR_TREND.value)
BASELINE_WINDOW = int(BASELINE_WINDOW_SAMPLES.value)


def _plural(count: float, singular: str) -> str:
    return singular if count == 1 else f"{singular}s"


def _qty(value: float, unit: str, fmt: str = "g") -> str:
    """A value with its unit attached, e.g. '91%', '103 bpm', '-1.8%/day'.

    The separator is dropped before a percent sign. SpO₂ is the metric this
    whole demo turns on, so "91 %" would appear in nearly every string the
    dashboard renders; every other unit here takes a space.
    """
    separator = "" if unit.startswith("%") else " "
    return f"{value:{fmt}}{separator}{unit}"


def _deviation_phrase(spec: VitalSpec, threshold: Threshold, observed: float) -> str:
    """How far past the personal baseline, in the form that was actually crossed.

    One helper because the same statement appears in the fired rule's
    description and in `current_abnormalities`, and two copies would eventually
    phrase the same deviation differently.

    The relative form names the direction ("below"/"above") instead of repeating
    "of baseline", which would otherwise land next to the words "personal
    baseline" in the same sentence. Which way is which comes from
    `adverse_direction`: a breached baseline threshold means the movement was
    adverse, so falling-adverse metrics are below and rising-adverse ones above.
    """
    if threshold is spec.baseline_delta_pct:
        return f"{observed:.1f}% {'below' if spec.adverse_direction < 0 else 'above'}"
    return f"{observed:.1f} {threshold.unit} beyond"


@dataclass(frozen=True)
class VitalSpec:
    """Everything the rules need to know about one vital, in one place.

    `adverse_direction` is the only encoding of which way is bad: -1 when
    falling is adverse (SpO₂, sleep, activity), +1 when rising is (heart rate,
    temperature, respiratory rate). `trends.py` takes it as an argument and
    every threshold in `thresholds.py` is stored as an unsigned magnitude, so a
    falling saturation and a rising pulse share one comparison shape instead of
    each call site re-deriving the sign.
    """

    metric: VitalMetric
    label: str
    unit: str
    adverse_direction: int
    dimension: ClinicalDimension
    rule_code: str
    adverse_slope: Threshold
    baseline_delta: Threshold | None = None
    baseline_delta_pct: Threshold | None = None
    clinical_measurement: bool = True

    @property
    def is_supporting_signal(self) -> bool:
        """Not `bool(news2_bands(...))`, even though the two coincide today.

        NEWS2 coverage is a property of a published instrument; whether a signal
        is a clinical measurement is a property of where the number comes from.
        A future device-derived NEWS2 parameter would be scoreable and still not
        a clinical measurement, so the two facts are kept separate rather than
        deriving one from the other.
        """
        return not self.clinical_measurement


VITAL_SPECS: tuple[VitalSpec, ...] = (
    VitalSpec(
        metric=VitalMetric.SPO2,
        label="SpO2",
        unit="%",
        adverse_direction=-1,
        dimension=ClinicalDimension.OXYGENATION,
        rule_code="SPO2",
        adverse_slope=SPO2_ADVERSE_SLOPE,
        baseline_delta=SPO2_BASELINE_DROP,
        baseline_delta_pct=SPO2_BASELINE_DROP_PCT,
    ),
    VitalSpec(
        metric=VitalMetric.HEART_RATE,
        label="Heart rate",
        unit="bpm",
        adverse_direction=1,
        dimension=ClinicalDimension.CARDIAC_RATE,
        rule_code="HR",
        adverse_slope=HEART_RATE_ADVERSE_SLOPE,
        baseline_delta=HEART_RATE_BASELINE_RISE,
        baseline_delta_pct=HEART_RATE_BASELINE_RISE_PCT,
    ),
    VitalSpec(
        metric=VitalMetric.TEMPERATURE_C,
        label="Temperature",
        unit="C",
        adverse_direction=1,
        dimension=ClinicalDimension.TEMPERATURE,
        rule_code="TEMP",
        adverse_slope=TEMPERATURE_ADVERSE_SLOPE,
        # No relative form: Celsius has an arbitrary zero, so "2.7% above
        # baseline" means something different depending on whether the baseline
        # was 35.0 or 37.0. The absolute rise is the meaningful statement.
        baseline_delta=TEMPERATURE_BASELINE_RISE,
    ),
    VitalSpec(
        metric=VitalMetric.RESPIRATORY_RATE,
        label="Respiratory rate",
        unit="breaths/min",
        adverse_direction=1,
        dimension=ClinicalDimension.RESPIRATORY_RATE,
        rule_code="RR",
        adverse_slope=RESPIRATORY_RATE_ADVERSE_SLOPE,
        baseline_delta=RESPIRATORY_RATE_BASELINE_RISE,
    ),
    VitalSpec(
        metric=VitalMetric.SLEEP_HOURS,
        label="Sleep",
        unit="hours",
        adverse_direction=-1,
        dimension=ClinicalDimension.FUNCTIONAL_STATUS,
        rule_code="SLEEP",
        adverse_slope=SLEEP_ADVERSE_SLOPE,
        baseline_delta=SLEEP_BASELINE_DROP,
        clinical_measurement=False,
    ),
    VitalSpec(
        metric=VitalMetric.ACTIVITY_STEPS,
        label="Activity",
        unit="steps",
        adverse_direction=-1,
        dimension=ClinicalDimension.FUNCTIONAL_STATUS,
        rule_code="STEPS",
        adverse_slope=ACTIVITY_ADVERSE_SLOPE,
        # Relative form only: a fixed drop of 3000 steps is a rounding error for
        # a very active patient and most of the day for a sedentary one.
        baseline_delta_pct=ACTIVITY_BASELINE_DROP_PCT,
        clinical_measurement=False,
    ),
)
"""Iterated in `VitalMetric` declaration order, so `trends`, `data_gaps` and
`deteriorating_metrics` come out in a stable order across runs. A test asserts
this table covers every member of the enum, because a metric added to
`VitalMetric` and forgotten here would simply never be analysed."""

SPEC_BY_METRIC: dict[VitalMetric, VitalSpec] = {s.metric: s for s in VITAL_SPECS}


# --- Series extraction --------------------------------------------------------


def series_for(samples: Sequence[VitalSample], spec: VitalSpec) -> tuple[list[datetime], list[float]]:
    """The `(times, values)` pairs one metric actually reported, in time order.

    `getattr` is called without a default on purpose. `VitalMetric`'s values are
    the `VitalSample` field names, and a rename on either side should raise here
    rather than silently produce an empty series for a metric the device did
    report.

    Sparse rows are the normal case, not an error: a pulse oximeter reports SpO₂
    and heart rate, a thermometer reports only temperature, so each metric is
    filtered independently.
    """
    times: list[datetime] = []
    values: list[float] = []
    for sample in sorted(samples, key=lambda s: s.recorded_at):
        observed = getattr(sample, spec.metric.value)
        if observed is None:
            continue
        times.append(sample.recorded_at)
        values.append(float(observed))
    return times, values


# --- Anomaly score ------------------------------------------------------------


def composite_anomaly(
    components: Sequence[tuple[float, float]], *, supporting_signal: bool
) -> float:
    """Weighted mean over the components that apply, in [0, 1].

    `components` is a sequence of `(weight, value)` pairs. Renormalising over
    only the applicable weights is what makes metrics comparable across scales:
    without it, a metric NEWS2 does not score could never exceed the sum of the
    three remaining weights (0.60) and would be structurally unable to rank
    first no matter how dramatic its change. Renormalising alone overshoots — a
    40% sleep decline then beats an SpO₂ of 91% — so wearable-derived signals
    are scaled back down by `SUPPORTING_SIGNAL_WEIGHT`.
    """
    total_weight = sum(weight for weight, _ in components)
    if total_weight <= 0:
        return 0.0
    score = sum(weight * value for weight, value in components) / total_weight
    if supporting_signal:
        score *= SUPPORTING_SIGNAL_WEIGHT.value
    return round(min(max(score, 0.0), 1.0), 4)


# --- Per-metric analysis ------------------------------------------------------


@dataclass(frozen=True)
class MetricAnalysis:
    """One vital's series, its derived feature, and the rules it fired."""

    spec: VitalSpec
    stats: SeriesStats
    band: ScoreBand | None
    feature: TrendFeature
    fired_rules: tuple[FiredRule, ...] = ()
    gaps: tuple[str, ...] = ()
    abnormality: str | None = None

    @property
    def is_trendable(self) -> bool:
        return self.feature.slope_per_day is not None

    @property
    def is_rapid(self) -> bool:
        """Adverse slope + sustained run + enough samples + adequate fit.

        All four, because each one alone has a failure mode: a slope from two
        points is an artefact, a run of one is a single jump, a steep slope
        through scattered points is noise, and a well-fitted slope below the
        noise band is a patient who is simply not changing.
        """
        f = self.feature
        if f.direction is not TrendDirection.WORSENING or f.slope_per_day is None:
            return False
        if f.sample_count < MIN_TREND_SAMPLES:
            return False
        if f.consecutive_worsening_days < RAPID_DETERIORATION_MIN_RUN.value:
            return False
        return f.r_squared is not None and f.r_squared >= MIN_R_SQUARED_FOR_TREND.value


def classify_direction(spec: VitalSpec, slope: float | None) -> TrendDirection:
    """Map a slope onto a direction, with the metric's own noise band in between.

    `spec.adverse_slope` does double duty: it is the size of change per day that
    counts as adverse, and therefore also the band inside which movement is
    reported as `stable` rather than as an improving or worsening trend. Without
    that band, a slope of -0.02 %/day from sensor noise would be reported as a
    worsening oxygenation trend.
    """
    if slope is None:
        return TrendDirection.UNKNOWN
    if abs(slope) < spec.adverse_slope.value:
        return TrendDirection.STABLE
    return TrendDirection.WORSENING if slope * spec.adverse_direction > 0 else TrendDirection.IMPROVING


def _cutoff_crossed(spec: VitalSpec, band: ScoreBand) -> float | None:
    """The NEWS2 boundary between the observed band and the next band toward normal.

    This is the number a reader compares the observation against, so it is taken
    from the table rather than restated: for a metric where falling is adverse it
    is the band's upper bound (SpO₂ 91 → 92), and where rising is adverse the
    lower bound (HR 103 → 91). None when the normal side is unbounded, which
    only happens for a band that is not a breach.
    """
    return band.high if spec.adverse_direction < 0 else band.low


def _anomaly_components(
    spec: VitalSpec,
    stats: SeriesStats,
    band: ScoreBand | None,
    adverse_delta: float | None,
    adverse_delta_pct: float | None,
    slope: float | None,
    run: int,
) -> list[tuple[float, float]]:
    """The `(weight, value)` pairs that apply to this metric. See `composite_anomaly`."""
    components: list[tuple[float, float]] = []

    if band is not None:
        maximum = news2_max_parameter_score(spec.metric.value)
        if maximum:
            components.append((ANOMALY_WEIGHT_THRESHOLD.value, band.score / maximum))

    if stats.has_baseline:
        fractions = []
        if spec.baseline_delta is not None and adverse_delta is not None:
            fractions.append(max(0.0, adverse_delta) / spec.baseline_delta.value)
        if spec.baseline_delta_pct is not None and adverse_delta_pct is not None:
            fractions.append(max(0.0, adverse_delta_pct) / spec.baseline_delta_pct.value)
        # Guarded rather than assumed non-empty: a metric whose only baseline
        # threshold is the relative form, observed against a baseline of exactly
        # zero, has no usable fraction at all.
        if fractions:
            components.append((ANOMALY_WEIGHT_BASELINE.value, min(1.0, max(fractions))))

    if slope is not None:
        adverse = max(0.0, slope * spec.adverse_direction) / spec.adverse_slope.value
        # r² is None only for a zero-variance series, where the slope is 0 and
        # the component is 0 regardless. Discounting by the fit is what stops a
        # steep slope through scattered points scoring as deterioration.
        fit = stats.r_squared if stats.r_squared is not None else 0.0
        components.append((ANOMALY_WEIGHT_SLOPE.value, min(1.0, adverse) * fit))

    components.append(
        (ANOMALY_WEIGHT_RUN.value, min(1.0, run / ANOMALY_RUN_FULL_SCORE.value))
    )
    return components


def _absolute_rule(spec: VitalSpec, stats: SeriesStats, band: ScoreBand) -> FiredRule:
    cutoff = _cutoff_crossed(spec, band)
    observed = _qty(stats.current, spec.unit)
    return FiredRule(
        rule_id=f"R-MON-{spec.rule_code}-ABS-01",
        category=MONITORING_RULE_CATEGORY,
        description=(
            f"{spec.label} {observed} falls in the NEWS2 band {band.label} "
            f"(parameter score {band.score})"
        ),
        contribution=0.0,
        evidence=(
            f"observed {observed} at {stats.last_at:%Y-%m-%d %H:%M}, the most "
            f"recent of {stats.count} {_plural(stats.count, 'sample')}; band "
            f"{band.label} scores {band.score} — {NEWS2_SOURCE}. A per-parameter "
            f"score stays valid when not every parameter is recorded, which the "
            f"total does not."
        ),
        value=stats.current,
        threshold=cutoff,
    )


def _baseline_rule(
    spec: VitalSpec,
    stats: SeriesStats,
    adverse_delta: float | None,
    adverse_delta_pct: float | None,
    met: tuple[Threshold, ...],
) -> FiredRule:
    first = met[0]
    observed = adverse_delta_pct if first is spec.baseline_delta_pct else adverse_delta
    pct_text = (
        f" ({stats.delta_pct:+.1f}% of baseline)" if stats.delta_pct is not None else ""
    )
    return FiredRule(
        rule_id=f"R-MON-{spec.rule_code}-DELTA-01",
        category=MONITORING_RULE_CATEGORY,
        description=(
            f"{spec.label} has moved {_deviation_phrase(spec, first, observed)} "
            f"this patient's own baseline"
        ),
        contribution=0.0,
        evidence=(
            f"current {_qty(stats.current, spec.unit)} vs personal baseline "
            f"{_qty(stats.baseline, spec.unit)} ({stats.baseline_method}); delta "
            f"{_qty(stats.delta, spec.unit, '+g')}{pct_text}; crossed "
            + " and ".join(t.describe() for t in met)
            + ". Independent of any absolute cutoff: this is what makes a "
            "still-normal reading alarming in a patient whose own baseline was "
            "higher."
        ),
        value=stats.current,
        threshold=first.value,
    )


def _trend_rule(spec: VitalSpec, stats: SeriesStats, run: int) -> FiredRule:
    slope = stats.slope_per_day
    per_day = f"{spec.unit}/day"
    return FiredRule(
        rule_id=f"R-MON-{spec.rule_code}-TREND-01",
        category=MONITORING_RULE_CATEGORY,
        description=(
            f"{spec.label} is worsening at "
            f"{_qty(slope * spec.adverse_direction, per_day)} over "
            f"{stats.window_days} {_plural(stats.window_days, 'day')}"
        ),
        contribution=0.0,
        evidence=(
            f"ordinary least-squares slope {_qty(slope, per_day, '+g')} fitted to "
            f"{stats.count} {_plural(stats.count, 'sample')} spanning "
            f"{stats.span_days:.1f} days inside a {stats.window_days}-day window "
            f"(the slope is per elapsed day); r^2 {stats.r_squared:.3f}; {run} "
            f"consecutive adverse {_plural(run, 'transition')}; movement exceeds "
            f"{spec.adverse_slope.describe()}"
        ),
        value=slope,
        threshold=spec.adverse_slope.value * spec.adverse_direction,
    )


def _metric_gaps(spec: VitalSpec, stats: SeriesStats, direction: TrendDirection) -> list[str]:
    """What this metric's series could not support, stated as a limitation.

    Surfaced rather than silently omitted: a trend that was not computed and a
    trend that was computed and found flat look identical on a dashboard unless
    the difference is written down.
    """
    gaps: list[str] = []

    if not stats.has_baseline:
        gaps.append(f"{spec.label}: {stats.baseline_method}.")
    elif stats.count < MIN_TREND_SAMPLES:
        gaps.append(
            f"{spec.label}: {stats.count} "
            f"{_plural(stats.count, 'sample')} in the window, below the "
            f"{MIN_TREND_SAMPLES} required to report a trend "
            f"({MIN_SAMPLES_FOR_TREND.source}), so the current value is reported "
            f"and no slope is claimed."
        )

    if SPARSE_COVERAGE_FRACTION.is_met_by(stats.coverage):
        gaps.append(
            f"{spec.label}: observations cover {stats.coverage:.0%} of the "
            f"{stats.window_days}-day window, below the "
            f"{SPARSE_COVERAGE_FRACTION.value:.0%} at which a trend is reported "
            f"as fact rather than as a sparse indication."
        )

    if (
        direction is TrendDirection.WORSENING
        and stats.r_squared is not None
        and stats.r_squared < MIN_R_SQUARED_FOR_TREND.value
    ):
        gaps.append(
            f"{spec.label}: the slope is fitted with r^2 {stats.r_squared:.2f}, "
            f"below the {MIN_R_SQUARED_FOR_TREND.value:g} minimum, so the trend "
            f"is reported but flagged as a poor fit."
        )

    return gaps


def analyse_metric(samples: Sequence[VitalSample], spec: VitalSpec) -> MetricAnalysis | None:
    """Analyse one vital across the whole sample list.

    Returns None when the metric was never reported, which is a normal outcome
    for a sparse device rather than an error; the caller records it as an absent
    metric and feeds that into `data_quality`.
    """
    times, values = series_for(samples, spec)
    stats = describe_series(
        times, values, BASELINE_WINDOW, EXPECTED_SAMPLES_PER_DAY.value
    )
    if stats.count == 0 or stats.current is None:
        return None

    band = news2_band(spec.metric.value, stats.current)
    breach_band = band if band is not None and band.score >= 1 else None

    # Below MIN_TREND_SAMPLES the slope is suppressed rather than reported with
    # a caveat. `linear_trend` will happily fit two points and return r^2 = 1.0,
    # which on a dashboard reads as a confident trend line; the honest output
    # for two numbers is `direction=unknown` plus a data gap naming the count.
    trendable = stats.count >= MIN_TREND_SAMPLES
    slope = stats.slope_per_day if trendable else None
    r_squared = stats.r_squared if trendable else None
    direction = classify_direction(spec, slope)

    run = consecutive_adverse_run(values, spec.adverse_direction)

    adverse_delta = (
        stats.delta * spec.adverse_direction if stats.delta is not None else None
    )
    adverse_delta_pct = (
        stats.delta_pct * spec.adverse_direction if stats.delta_pct is not None else None
    )
    met = tuple(
        threshold
        for threshold, observed in (
            (spec.baseline_delta, adverse_delta),
            (spec.baseline_delta_pct, adverse_delta_pct),
        )
        if threshold is not None and threshold.is_met_by(observed)
    )

    anomaly = composite_anomaly(
        _anomaly_components(
            spec, stats, breach_band, adverse_delta, adverse_delta_pct, slope, run
        ),
        supporting_signal=spec.is_supporting_signal,
    )

    feature = TrendFeature(
        metric=spec.metric,
        unit=spec.unit,
        dimension=spec.dimension,
        sample_count=stats.count,
        current=stats.current,
        baseline=stats.baseline,
        delta=stats.delta,
        delta_pct=stats.delta_pct,
        min_value=stats.minimum,
        max_value=stats.maximum,
        slope_per_day=slope,
        r_squared=r_squared,
        direction=direction,
        consecutive_worsening_days=run,
        news2_score=band.score if band is not None else None,
        news2_band=band.label if band is not None else None,
        absolute_threshold_breach=breach_band is not None,
        threshold_value=_cutoff_crossed(spec, breach_band) if breach_band else None,
        threshold_source=NEWS2_SOURCE if breach_band else None,
        baseline_deviation_breach=bool(met),
        anomaly_score=anomaly,
    )

    rules: list[FiredRule] = []
    if breach_band is not None:
        rules.append(_absolute_rule(spec, stats, breach_band))
    if met:
        rules.append(_baseline_rule(spec, stats, adverse_delta, adverse_delta_pct, met))
    if direction is TrendDirection.WORSENING:
        rules.append(_trend_rule(spec, stats, run))

    return MetricAnalysis(
        spec=spec,
        stats=stats,
        band=band,
        feature=feature,
        fired_rules=tuple(rules),
        gaps=tuple(_metric_gaps(spec, stats, direction)),
        abnormality=_abnormality_text(spec, feature, met, adverse_delta, adverse_delta_pct),
    )


def _abnormality_text(
    spec: VitalSpec,
    feature: TrendFeature,
    met: tuple[Threshold, ...],
    adverse_delta: float | None,
    adverse_delta_pct: float | None,
) -> str | None:
    """One line per metric that is abnormal *now*, or None when it is not.

    Deliberately limited to the two point-in-time components — the published
    band and the deviation from baseline. The trend belongs to `direction` and
    `rapid_deterioration`; folding it in here would make this list a claim about
    the future rather than a description of the present.
    """
    clauses: list[str] = []
    if feature.absolute_threshold_breach and feature.news2_score is not None:
        clauses.append(
            f"NEWS2 band {feature.news2_band}, parameter score {feature.news2_score}"
        )
    if met:
        observed = (
            adverse_delta_pct if met[0] is spec.baseline_delta_pct else adverse_delta
        )
        clauses.append(
            f"{_deviation_phrase(spec, met[0], observed)} a personal baseline of "
            f"{_qty(feature.baseline, spec.unit)}"
        )
    if not clauses:
        return None
    return f"{spec.label} {_qty(feature.current, spec.unit)} — " + "; ".join(clauses)


# --- Assessment ---------------------------------------------------------------


@dataclass(frozen=True)
class MonitoringDerivation:
    """The assessment plus the rules that produced it, returned together so the
    trace can quote the same evaluation the payload was built from."""

    assessment: MonitoringAssessment
    fired_rules: tuple[FiredRule, ...]


def _rapid_rule(analyses: Sequence[MetricAnalysis]) -> FiredRule:
    details = "; ".join(
        f"{a.spec.label} "
        f"{_qty(a.feature.slope_per_day, f'{a.spec.unit}/day', '+g')} "
        f"(r^2 {a.feature.r_squared:.2f}, {a.feature.consecutive_worsening_days} "
        f"consecutive adverse {_plural(a.feature.consecutive_worsening_days, 'transition')})"
        for a in analyses
    )
    return FiredRule(
        rule_id=RAPID_RULE_ID,
        category=MONITORING_RULE_CATEGORY,
        description=(
            f"Rapid deterioration: {len(analyses)} monitored "
            f"{_plural(len(analyses), 'metric')} combine an adverse slope with a "
            f"sustained worsening run"
        ),
        contribution=0.0,
        evidence=(
            f"{details}. Criteria: {RAPID_DETERIORATION_MIN_RUN.describe()}; "
            f"{MIN_R_SQUARED_FOR_TREND.describe()}; "
            f"{MIN_SAMPLES_FOR_TREND.describe()}"
        ),
        value=float(len(analyses)),
        threshold=RAPID_DETERIORATION_MIN_RUN.value,
    )


def _news2_trigger_rule(
    analyses: Sequence[MetricAnalysis], partial_total: int | None
) -> FiredRule:
    scoring = [a for a in analyses if a.band is not None and a.band.score >= NEWS2_ANY_PARAMETER_TRIGGER]
    names = ", ".join(
        f"{a.spec.label} {_qty(a.stats.current, a.spec.unit)} ({a.band.label})"
        for a in scoring
    )
    return FiredRule(
        rule_id=NEWS2_TRIGGER_RULE_ID,
        category=MONITORING_RULE_CATEGORY,
        description=(
            f"NEWS2 escalation trigger: {names} at the maximum single-parameter "
            f"score of {NEWS2_ANY_PARAMETER_TRIGGER}"
        ),
        contribution=0.0,
        evidence=(
            f"{NEWS2_SOURCE}. A score of {NEWS2_ANY_PARAMETER_TRIGGER} in any one "
            f"parameter escalates regardless of the total, and unlike the total it "
            f"stays valid when only some parameters are recorded — which is why "
            f"this trigger, not the total, is what the rules use. Partial total "
            f"across the recorded parameters: {partial_total}. "
            f"{NEWS2_PARTIAL_SCORE_CAVEAT}"
        ),
        value=float(len(scoring)),
        threshold=float(NEWS2_ANY_PARAMETER_TRIGGER),
    )


def _quality_rule(quality: str, gaps: Sequence[str], present: int, total: int) -> FiredRule:
    return FiredRule(
        rule_id=QUALITY_RULE_ID,
        category=MONITORING_RULE_CATEGORY,
        description=f"Monitoring data quality: {quality}",
        contribution=0.0,
        evidence=(
            f"{present} of {total} monitored metrics reported at least one "
            f"observation. " + ("; ".join(gaps) if gaps else "No gaps recorded.")
        ),
        value=float(present),
        threshold=float(total),
    )


def classify_data_quality(
    analyses: Sequence[MetricAnalysis], absent: Sequence[VitalSpec]
) -> str:
    """complete | partial | sparse | insufficient.

    `insufficient` means no trend could be computed for any metric, which is a
    different statement from `sparse`: the first says the agent had nothing to
    work with, the second says it had something and the window had holes in it.
    """
    if not analyses or not any(a.is_trendable for a in analyses):
        return "insufficient"
    if SPARSE_COVERAGE_FRACTION.is_met_by(max(a.stats.coverage for a in analyses)):
        return "sparse"
    if absent or any(not a.is_trendable for a in analyses):
        return "partial"
    if any(SPARSE_COVERAGE_FRACTION.is_met_by(a.stats.coverage) for a in analyses):
        return "partial"
    return "complete"


def _baseline_method_text() -> str:
    return (
        f"mean of the earliest {BASELINE_WINDOW} "
        f"{_plural(BASELINE_WINDOW, 'sample')} in the observation window "
        f"({BASELINE_WINDOW_SAMPLES.source}); the latest observation is always "
        f"excluded, because a value that is part of its own baseline cannot show "
        f"deterioration"
    )


def _describe_feature(feature: TrendFeature) -> str:
    """The three components of one vital, as prose. Total over every None case,
    because this is rendered text and a missing baseline is a finding, not a
    formatting error."""
    spec = SPEC_BY_METRIC[feature.metric]
    parts = [f"{spec.label} {_qty(feature.current, feature.unit)}"]

    if feature.baseline is None:
        parts.append("no personal baseline derivable")
    else:
        delta = f"delta {_qty(feature.delta, feature.unit, '+g')}"
        if feature.delta_pct is not None:
            delta += f", {feature.delta_pct:+.1f}% of baseline"
        parts.append(
            f"personal baseline {_qty(feature.baseline, feature.unit)} ({delta})"
        )

    if feature.slope_per_day is None:
        # `direction` is UNKNOWN exactly when the slope is None, so there is no
        # direction to name here.
        parts.append("no trend reported")
    else:
        fit = (
            f", r^2 {feature.r_squared:.2f}"
            if feature.r_squared is not None
            else ", fit undefined for a constant series"
        )
        parts.append(
            f"{feature.direction.value} at "
            f"{_qty(feature.slope_per_day, f'{feature.unit}/day', '+g')}{fit}"
        )

    if feature.consecutive_worsening_days:
        parts.append(
            f"{feature.consecutive_worsening_days} consecutive adverse "
            f"{_plural(feature.consecutive_worsening_days, 'transition')}"
        )
    if feature.news2_score is not None:
        parts.append(f"NEWS2 parameter score {feature.news2_score}")
    parts.append(f"anomaly score {feature.anomaly_score:.2f}")
    return ", ".join(parts)


def _build_summary(assessment: MonitoringAssessment) -> str:
    if assessment.sample_count == 0:
        return (
            "No monitoring samples in the observation window, so no current "
            "value, temporal trend or personal baseline could be computed. This "
            "is a data gap rather than a normal result."
        )

    span = ""
    if assessment.first_recorded_at and assessment.last_recorded_at:
        span = (
            f" from {assessment.first_recorded_at:%Y-%m-%d} to "
            f"{assessment.last_recorded_at:%Y-%m-%d}"
        )
    parts = [
        f"{assessment.sample_count} {_plural(assessment.sample_count, 'sample')} "
        f"over {assessment.window_days} {_plural(assessment.window_days, 'day')}"
        f"{span}, covering {len(assessment.trends)} of {len(VITAL_SPECS)} "
        f"monitored metrics."
    ]

    worst = next(
        (t for t in assessment.trends if t.metric == assessment.worst_metric), None
    )
    if worst:
        parts.append(f"Most abnormal: {_describe_feature(worst)}.")

    if assessment.rapid_deterioration:
        labels = ", ".join(
            SPEC_BY_METRIC[m].label for m in assessment.deteriorating_metrics
        )
        count = len(assessment.deteriorating_metrics)
        parts.append(
            f"Rapid deterioration in {count} {_plural(count, 'metric')} "
            f"({labels}): each combines an adverse slope with a sustained "
            f"worsening run, which is a statement about the trajectory rather "
            f"than about any single reading."
        )
    else:
        parts.append(
            "No metric combined an adverse slope with a sustained worsening run, "
            "so no rapid deterioration is reported."
        )

    if assessment.news2_partial_total is not None:
        trigger = (
            f", and at least one parameter is at the escalation score of "
            f"{NEWS2_ANY_PARAMETER_TRIGGER}"
            if assessment.news2_any_parameter_trigger
            else ""
        )
        parts.append(
            f"Partial NEWS2 total {assessment.news2_partial_total} across the "
            f"parameters this system records{trigger}; the total is partial and "
            f"the caveat ships with it."
        )

    if assessment.current_abnormalities:
        count = len(assessment.current_abnormalities)
        parts.append(
            f"{count} of {len(assessment.trends)} monitored "
            f"{_plural(count, 'metric')} "
            f"{'is' if count == 1 else 'are'} abnormal right now against either "
            f"the published band or this patient's own baseline."
        )

    gaps = (
        f", with {len(assessment.data_gaps)} recorded "
        f"{_plural(len(assessment.data_gaps), 'gap')}"
        if assessment.data_gaps
        else ""
    )
    parts.append(f"Data quality: {assessment.data_quality}{gaps}.")
    return " ".join(parts)


def assess_monitoring(samples: Sequence[VitalSample]) -> MonitoringDerivation:
    """Single entry point used by the Monitoring Agent."""
    ordered = sorted(samples, key=lambda s: s.recorded_at)

    analyses = [
        a
        for a in (analyse_metric(ordered, spec) for spec in VITAL_SPECS)
        if a is not None
    ]
    present = {a.spec.metric for a in analyses}
    absent = tuple(s for s in VITAL_SPECS if s.metric not in present)

    if ordered:
        span_days = (
            ordered[-1].recorded_at - ordered[0].recorded_at
        ).total_seconds() / SECONDS_PER_DAY
        window_days = int(span_days) + 1
    else:
        window_days = 0

    trends = [a.feature for a in analyses]
    deteriorating = [a for a in analyses if a.is_rapid]
    scores = [a.band.score for a in analyses if a.band is not None]
    partial_total = sum(scores) if scores else None
    trigger = any(s >= NEWS2_ANY_PARAMETER_TRIGGER for s in scores)

    # `max` keeps the first of equals, and `trends` is in VITAL_SPECS order, so
    # a tie resolves to the metric declared first rather than to whichever
    # happened to sort last.
    top = max(trends, key=lambda t: t.anomaly_score) if trends else None
    # A score of exactly 0.0 means nothing was abnormal, and naming a "worst
    # metric" out of an entirely normal window invites a reading the data does
    # not support.
    worst_metric = top.metric if top and top.anomaly_score > 0.0 else None

    gaps: list[str] = [g for a in analyses for g in a.gaps]
    gaps.extend(
        f"{spec.label}: no observations in the window, so nothing could be "
        f"computed for this metric."
        for spec in absent
    )
    if not ordered:
        gaps.append(
            "No monitoring samples at all, so current value, temporal trend and "
            "deviation from personal baseline are all unavailable."
        )

    quality = classify_data_quality(analyses, absent)

    assessment = MonitoringAssessment(
        window_days=window_days,
        sample_count=len(ordered),
        first_recorded_at=ordered[0].recorded_at if ordered else None,
        last_recorded_at=ordered[-1].recorded_at if ordered else None,
        trends=trends,
        news2_partial_total=partial_total,
        news2_any_parameter_trigger=trigger,
        rapid_deterioration=bool(deteriorating),
        deteriorating_metrics=[a.spec.metric for a in deteriorating],
        current_abnormalities=[a.abnormality for a in analyses if a.abnormality],
        worst_metric=worst_metric,
        baseline_method=_baseline_method_text() if analyses else "",
        data_quality=quality,
        data_gaps=gaps,
    )
    # model_copy skips validation, which is safe for a str field and avoids
    # restating fifteen constructor arguments to attach prose to them.
    assessment = assessment.model_copy(update={"summary": _build_summary(assessment)})

    rules: list[FiredRule] = [r for a in analyses for r in a.fired_rules]
    if deteriorating:
        rules.append(_rapid_rule(deteriorating))
    if trigger:
        rules.append(_news2_trigger_rule(analyses, partial_total))
    if quality != "complete":
        rules.append(_quality_rule(quality, gaps, len(analyses), len(VITAL_SPECS)))

    return MonitoringDerivation(assessment=assessment, fired_rules=tuple(rules))
