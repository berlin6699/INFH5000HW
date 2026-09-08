"""Monitoring Agent contracts: continuous physiological observation.

The governing concept is `current value + temporal trend + personal baseline`.
A single SpO₂ reading of 93% is ambiguous; 93% in a patient whose own baseline
is 98% and who has fallen one point per day for four days is not. Every field
here exists to carry one of those three components.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from app.reasoning.thresholds import NEWS2_PARTIAL_SCORE_CAVEAT
from app.schemas.common import ClinicalDimension, Schema, TrendDirection, VitalMetric


class VitalSample(Schema):
    """One observation timestamp. Fields are optional because different sources
    report different subsets — a pulse oximeter gives SpO₂ and HR, a thermometer
    gives only temperature. Sparse rows are normal, not an error."""

    recorded_at: datetime
    spo2: float | None = Field(default=None, ge=0, le=100, description="Percent.")
    heart_rate: float | None = Field(default=None, ge=0, le=300, description="BPM.")
    temperature_c: float | None = Field(default=None, ge=30, le=45, description="Celsius.")
    respiratory_rate: float | None = Field(default=None, ge=0, le=80, description="Breaths/min.")
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    activity_steps: int | None = Field(default=None, ge=0)
    source: Literal["synthetic", "csv", "device", "manual"] = "synthetic"


class TrendFeature(Schema):
    """Time-series analysis for a single vital over the observation window.

    Optional numeric fields are `None` when the series cannot support them, and
    that is **not** the same as `0.0`. A `slope_per_day` of `None` means "too
    few observations to fit a line"; `0.0` means "fitted, and flat". Collapsing
    the two would report an unmeasured trend as a stable one, which is the exact
    failure mode this agent exists to avoid.
    """

    metric: VitalMetric
    unit: str = Field(
        description="Unit for `current`, `delta` and `slope_per_day`, e.g. '%' or "
        "'bpm'. Carried here so the frontend renders units from the contract "
        "instead of maintaining its own metric-to-unit map that can drift."
    )
    dimension: ClinicalDimension = Field(
        description="The longitudinal axis this vital informs, so the Coordinator "
        "can place a trend against a historical finding without re-deriving it."
    )
    sample_count: int = Field(ge=0)

    current: float = Field(description="Most recent observed value.")
    baseline: float | None = Field(
        default=None,
        description="This patient's own baseline, not a population normal range. "
        "Computed by `baseline_method` below. None when the series has fewer than "
        "two samples, because one observation cannot be its own baseline.",
    )
    delta: float | None = Field(
        default=None, description="current - baseline, in `unit`. None when there is no baseline."
    )
    delta_pct: float | None = Field(
        default=None,
        description="`delta` as a percentage of baseline. None when there is no "
        "baseline or the baseline is zero, where a percentage is meaningless "
        "rather than infinite.",
    )

    min_value: float
    max_value: float

    slope_per_day: float | None = Field(
        default=None,
        description="Ordinary least-squares slope over the window, `unit` per day. "
        "This is the quantity that distinguishes a stable low value from a falling "
        "one. None when fewer than two samples exist or all samples share one "
        "timestamp, where the slope is undefined rather than zero.",
    )
    r_squared: float | None = Field(
        default=None,
        description="Goodness of fit for the slope. A steep slope with low r² is "
        "noise, not deterioration, and is reported as such. None for a series with "
        "no variance: a constant series is perfectly flat, but there was nothing "
        "for the fit to explain, so the quality is undefined rather than 1.0.",
    )
    direction: TrendDirection
    consecutive_worsening_days: int = Field(
        default=0,
        description="Length of the current monotonic adverse run ending at the "
        "latest sample, counted as day-over-day transitions. Sustained runs matter "
        "more than single jumps, and a flat step breaks the run because a value "
        "that has stopped moving has stopped worsening.",
    )

    news2_score: int | None = Field(
        default=None,
        description="NEWS2 parameter score for `current` (Royal College of "
        "Physicians, 2017). None when NEWS2 does not score this metric — sleep and "
        "activity are not parameters — or when no value was observed. Never 0 for "
        "'not applicable', because 0 is a real score meaning normal.",
    )
    news2_band: str | None = Field(
        default=None,
        description="The published band `current` fell into, e.g. '<= 91%'. Rendered "
        "next to the score so the number is auditable against the source."
    )

    absolute_threshold_breach: bool = Field(
        default=False,
        description="True if `current` scores abnormally on the published "
        "point-in-time instrument. This is the one component with external "
        "validation behind it.",
    )
    threshold_value: float | None = None
    threshold_source: str | None = Field(
        default=None,
        description="Citation for the threshold, e.g. NEWS2. A threshold without a "
        "source is an assertion, not evidence.",
    )

    baseline_deviation_breach: bool = Field(
        default=False,
        description="True if the deviation from this patient's own baseline exceeds "
        "the per-metric threshold, independent of any absolute cutoff. This is what "
        "makes a still-normal 95% alarming in a patient whose baseline is 98%.",
    )

    anomaly_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted composite of threshold breach, baseline deviation, "
        "slope and run length. Normalised so metrics on different scales — percent, "
        "beats per minute, degrees — are comparable and can be ranked against each "
        "other to pick `worst_metric`.",
    )


class MonitoringAssessment(Schema):
    """Monitoring Agent output."""

    window_days: int
    sample_count: int
    first_recorded_at: datetime | None = None
    last_recorded_at: datetime | None = None

    trends: list[TrendFeature] = Field(default_factory=list)

    news2_partial_total: int | None = Field(
        default=None,
        description="Sum of the NEWS2 parameter scores over the metrics this system "
        "records. PARTIAL: NEWS2 also scores systolic blood pressure and level of "
        "consciousness, which `VitalSample` does not carry, so this must not be read "
        "against the published total-score response bands. None when no parameter "
        "was scoreable.",
    )
    news2_any_parameter_trigger: bool = Field(
        default=False,
        description="True when any single parameter scores 3. NEWS2 escalates on this "
        "regardless of the total, and unlike the total it stays valid when only some "
        "parameters are recorded — so this is the trigger the rules actually use.",
    )
    news2_caveat: str = Field(
        default="",
        description="The partial-score limitation, carried in the payload rather than "
        "left to the UI to remember. Auto-filled by a validator so a caller cannot "
        "omit it.",
    )

    rapid_deterioration: bool = Field(
        default=False,
        description="True when at least one vital combines an adverse slope with "
        "a sustained worsening run. This flag, not any single value, is what "
        "escalates the assessment.",
    )
    deteriorating_metrics: list[VitalMetric] = Field(default_factory=list)
    current_abnormalities: list[str] = Field(default_factory=list)
    worst_metric: VitalMetric | None = None

    baseline_method: str = Field(
        default="",
        description="How the personal baseline was derived, stated explicitly so "
        "the delta can be audited rather than taken on trust.",
    )
    data_quality: Literal["complete", "partial", "sparse", "insufficient"] = "insufficient"
    data_gaps: list[str] = Field(default_factory=list)
    summary: str = ""

    @model_validator(mode="after")
    def _always_disclose_the_partial_score(self) -> MonitoringAssessment:
        """Fill the caveat when the caller omitted it.

        Same mechanism as `ImagingFinding.badge`: a disclosure that depends on
        every caller remembering to pass it will eventually be rendered without
        it. A partial NEWS2 total shown as though it were a full one is the
        specific misreading this prevents.
        """
        if not self.news2_caveat.strip():
            self.news2_caveat = NEWS2_PARTIAL_SCORE_CAVEAT
        return self
