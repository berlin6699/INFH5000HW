"""Clinical cutoffs used by the deterministic rule engine.

**Read this before trusting any number in this file.**

Every constant here is transcribed from a publicly documented source, and every
one carries that source in a `Threshold.source` string so it can be rendered
next to the rule that fired. That is an auditability measure, not a claim of
clinical validity:

* This is an educational research prototype. Nothing here has been validated
  against patient outcomes, and none of it has been reviewed by a clinician as
  part of this project's implementation.
* Several weights are **project-defined modelling choices** rather than
  published figures. Those say so explicitly in their source string. Inventing
  a plausible-looking citation would be worse than admitting the number is ours.
* Real decision support derives risk from validated instruments fitted to
  cohorts. A product of hand-chosen multipliers is a teaching device that
  demonstrates *where* a risk adjustment comes from, not how much it should be.

Any deployment beyond a classroom demo requires clinician review of every
value below.

`Threshold` exists so a cutoff and its provenance cannot be separated: the rule
engine passes `describe()` straight into `FiredRule.evidence`, which is how the
UI answers "why is this value a problem?" with a citation instead of an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Comparator = Literal[">=", ">", "<=", "<"]

PROJECT_DEFINED = "project-defined modelling choice; not a published clinical cutoff"

# The clamp on the history multiplier. These are the Field bounds on both
# PatientProfile.history_multiplier and FinalAssessment.history_multiplier, so
# the rule engine and the wire contract share one definition instead of two
# literals that a test has to keep honest. This stays acyclic because this
# module imports nothing from app.schemas.
HISTORY_MULTIPLIER_FLOOR = 1.0
HISTORY_MULTIPLIER_CEILING = 1.4


@dataclass(frozen=True)
class Threshold:
    """A cutoff, its unit, and where it came from."""

    value: float
    unit: str = ""
    comparator: Comparator = ">="
    source: str = ""

    def is_met_by(self, value: float | None) -> bool:
        """False for a missing value: absent data is a gap, never a trigger."""
        if value is None:
            return False
        if self.comparator == ">=":
            return value >= self.value
        if self.comparator == ">":
            return value > self.value
        if self.comparator == "<=":
            return value <= self.value
        return value < self.value

    def describe(self) -> str:
        """Render as evidence text, e.g. 'age >= 65 years — CURB-65 ...'.

        An em-dash rather than parentheses: callers wrap this in parentheses of
        their own, and several source strings already contain a parenthetical
        citation, so nesting brackets quickly becomes unreadable.
        """
        unit = f" {self.unit}" if self.unit else ""
        text = f"{self.comparator} {self.value:g}{unit}"
        return f"{text} — {self.source}" if self.source else text


NEWS2_SOURCE = (
    "NEWS2 (Royal College of Physicians, 2017) — validated, published "
    "early-warning score"
)

NEWS2_PARTIAL_SCORE_CAVEAT = (
    "PARTIAL NEWS2: this system records only respiratory rate, SpO2, heart rate "
    "and temperature. NEWS2 also scores systolic blood pressure and level of "
    "consciousness, so the sum below is NOT a NEWS2 total and must not be read "
    "against the published total-score response bands. The per-parameter scores "
    "and the 'score of 3 in any one parameter' trigger are valid on their own, "
    "and are what the rules actually use."
)


@dataclass(frozen=True)
class ScoreBand:
    """One band of an ordinal early-warning score: `low <= value < high` earns `score`.

    Bounds are **low-inclusive, high-exclusive** and `None` means unbounded, so
    a metric's bands tile the real line with no gaps and no overlaps.

    NEWS2 is published against integer readings ("SpO2 92-93 scores 2"), which
    leaves fractional values undefined. Shifting each published boundary to the
    top of its band reproduces the integer behaviour exactly and resolves
    fractions toward the more abnormal score — 91.5% is treated like 91%, not
    like 92%. Under-reading a falling saturation is the worse error here.
    """

    score: int
    low: float | None = None
    high: float | None = None
    label: str = ""

    def contains(self, value: float) -> bool:
        if self.low is not None and value < self.low:
            return False
        return self.high is None or value < self.high


NEWS2_BANDS: dict[str, tuple[ScoreBand, ...]] = {
    "spo2": (
        ScoreBand(3, None, 92, "<= 91%"),
        ScoreBand(2, 92, 94, "92-93%"),
        ScoreBand(1, 94, 96, "94-95%"),
        ScoreBand(0, 96, None, ">= 96%"),
    ),
    "heart_rate": (
        ScoreBand(3, None, 41, "<= 40 bpm"),
        ScoreBand(1, 41, 51, "41-50 bpm"),
        ScoreBand(0, 51, 91, "51-90 bpm"),
        ScoreBand(1, 91, 111, "91-110 bpm"),
        ScoreBand(2, 111, 131, "111-130 bpm"),
        ScoreBand(3, 131, None, ">= 131 bpm"),
    ),
    "temperature_c": (
        ScoreBand(3, None, 35.1, "<= 35.0 C"),
        ScoreBand(1, 35.1, 36.1, "35.1-36.0 C"),
        ScoreBand(0, 36.1, 38.1, "36.1-38.0 C"),
        ScoreBand(1, 38.1, 39.1, "38.1-39.0 C"),
        ScoreBand(2, 39.1, None, ">= 39.1 C"),
    ),
    "respiratory_rate": (
        ScoreBand(3, None, 9, "<= 8 /min"),
        ScoreBand(1, 9, 12, "9-11 /min"),
        ScoreBand(0, 12, 21, "12-20 /min"),
        ScoreBand(2, 21, 25, "21-24 /min"),
        ScoreBand(3, 25, None, ">= 25 /min"),
    ),
}
"""Four of NEWS2's seven parameters. Blood pressure and consciousness are not
in `VitalSample`, so they are absent rather than assumed normal — see
`NEWS2_PARTIAL_SCORE_CAVEAT`."""

NEWS2_ANY_PARAMETER_TRIGGER = 3
"""NEWS2 escalates on a score of 3 in any single parameter regardless of the
total. This stays valid under partial scoring, which the total does not."""


def news2_bands(metric: str) -> tuple[ScoreBand, ...]:
    """Bands for a metric, or an empty tuple for one NEWS2 does not score.

    Sleep and activity are not NEWS2 parameters. Returning nothing rather than
    raising keeps the caller's loop over all six vitals simple, and the rules
    report the absence as "not covered by NEWS2" instead of as a score of 0 —
    zero would read as 'normal'.
    """
    return NEWS2_BANDS.get(metric, ())


def news2_band(metric: str, value: float | None) -> ScoreBand | None:
    """The band a reading falls into, or None when it is not scoreable.

    The single place the band table is scanned. Callers need the band itself
    rather than just the score, because its `label` is what makes the number
    auditable against the published table and its bounds are the cutoff the
    reading crossed.
    """
    if value is None:
        return None
    return next((b for b in news2_bands(metric) if b.contains(value)), None)


def news2_score(metric: str, value: float | None) -> int | None:
    """The NEWS2 parameter score, or None when it cannot be computed.

    None means "not scoreable" — either the sample is missing or the metric is
    outside NEWS2. It never means zero, because zero is a real score meaning
    normal.
    """
    band = news2_band(metric, value)
    return band.score if band else None


def news2_max_parameter_score(metric: str) -> int | None:
    """The highest score NEWS2 assigns to this parameter, or None if unscored.

    Derived from the table rather than written down as a literal 3. The anomaly
    score normalises a parameter score against this, so a hardcoded maximum
    would silently stop matching the moment a band table is edited.
    """
    return max((b.score for b in news2_bands(metric)), default=None)


# --- Age ----------------------------------------------------------------------

AGE_ELDERLY = Threshold(
    value=65,
    unit="years",
    source="CURB-65 age criterion (Lim et al., BMJ 2003; BTS community-acquired "
    "pneumonia guideline)",
)
"""Age >= 65 scores a point in CURB-65, the best-known public severity rule for
community-acquired pneumonia. Used here as a risk modifier, not as a severity
score."""

AGE_MIDDLE = Threshold(
    value=50,
    unit="years",
    source="age band aligned with USPSTF 2021 lung-cancer screening eligibility "
    "(ages 50-80)",
)
"""Not a diagnostic cutoff. The demo patient is 55, and a middle-age band keeps
the modifier proportional rather than jumping straight from 1.0 to the elderly
weight."""

# --- Smoking ------------------------------------------------------------------

PACK_YEARS_SIGNIFICANT = Threshold(
    value=20,
    unit="pack-years",
    source="USPSTF 2021 lung-cancer screening eligibility (>= 20 pack-years)",
)
PACK_YEARS_ANY = Threshold(
    value=10,
    unit="pack-years",
    source=PROJECT_DEFINED + "; below the USPSTF 20 pack-year screening threshold",
)
"""A lighter exposure band. The demo patient sits here at ~8 pack-years, below
both cutoffs, and therefore contributes no multiplier — recording the factor at
weight 1.0 keeps it visible in the UI without inflating the score."""

# --- Body habitus -------------------------------------------------------------

BMI_OVERWEIGHT = Threshold(
    value=25,
    unit="kg/m^2",
    source="WHO BMI classification (overweight)",
)
BMI_OBESE = Threshold(
    value=30,
    unit="kg/m^2",
    source="WHO BMI classification (obese)",
)

# --- Record completeness ------------------------------------------------------

STALE_HISTORY_MONTHS = Threshold(
    value=18,
    unit="months",
    comparator=">",
    source=PROJECT_DEFINED + "; chosen so the demo's 2026-02 hypertension review "
    "counts as current while the 2025-06 chest X-ray counts as dated",
)
"""How old the most recent record may be before the profile declares a gap.
Arbitrary by nature — there is no published rule for 'record too old' — but
surfacing the judgement is better than silently trusting a two-year-old note."""

NO_PRIOR_IMAGING_DAYS = Threshold(
    value=365,
    unit="days",
    comparator=">",
    source=PROJECT_DEFINED + "; a respiratory presentation with no chest imaging "
    "in the last year has no comparison anchor",
)
"""Longitudinal imaging comparison needs a reasonably recent prior study. When
the only prior is older than this, the absence of a comparison baseline is
reported as a limitation rather than quietly skipped."""

# --- Risk-factor weights ------------------------------------------------------
#
# Multiplicative modifiers, each in [1.0, 1.5] per the RiskFactor contract.
# Their product is clamped to HISTORY_MULTIPLIER_CEILING. These are the least
# defensible numbers in the file and the most important to review: they are
# modelling choices made for a demonstrable, explainable prototype.

WEIGHT_CHRONIC_RESPIRATORY = Threshold(
    value=1.15,
    source=PROJECT_DEFINED + "; chronic lung disease reduces respiratory reserve",
)
WEIGHT_PRIOR_PULMONARY_FINDING = Threshold(
    value=1.12,
    source=PROJECT_DEFINED + "; prior structural abnormality on the same "
    "dimension the current imaging is assessed on",
)
WEIGHT_CHRONIC_CARDIAC = Threshold(
    value=1.12,
    source=PROJECT_DEFINED + "; cardiac comorbidity complicates a hypoxic "
    "presentation",
)
WEIGHT_AGE_ELDERLY = Threshold(
    value=1.20,
    source="modifier applied when AGE_ELDERLY is met; the threshold itself is "
    "CURB-65, the magnitude is " + PROJECT_DEFINED,
)
WEIGHT_AGE_MIDDLE = Threshold(
    value=1.05,
    source=PROJECT_DEFINED + "; deliberately small so age alone cannot dominate",
)
WEIGHT_SMOKING_SIGNIFICANT = Threshold(
    value=1.15,
    source="modifier applied when PACK_YEARS_SIGNIFICANT is met; the threshold "
    "is USPSTF, the magnitude is " + PROJECT_DEFINED,
)
WEIGHT_SMOKING_LIGHT = Threshold(
    value=1.05,
    source="modifier applied when PACK_YEARS_ANY is met; magnitude is "
    + PROJECT_DEFINED,
)
WEIGHT_BMI_OBESE = Threshold(
    value=1.10,
    source="modifier applied when BMI_OBESE is met; the threshold is WHO, the "
    "magnitude is " + PROJECT_DEFINED,
)
WEIGHT_BMI_OVERWEIGHT = Threshold(
    value=1.04,
    source="modifier applied when BMI_OVERWEIGHT is met; the threshold is WHO, "
    "the magnitude is " + PROJECT_DEFINED,
)

# --- Monitoring: deviation from the personal baseline --------------------------
#
# Every constant below is a **magnitude of adverse change**, not a signed delta.
# Direction comes from the metric's own semantics in the rule engine, so a
# falling SpO2 and a rising heart rate share one comparison shape instead of
# each caller re-deriving which way is bad.
#
# These are the numbers that make the system's central claim work: a single
# reading of 93% is ambiguous, but 93% in a patient whose own baseline is 98%
# is not. None of the magnitudes below is published — there is no guideline for
# "how far from your own baseline counts" — so all of them say so.

SPO2_BASELINE_DROP = Threshold(
    value=3,
    unit="percentage points",
    source=PROJECT_DEFINED + "; a drop this large from a patient's own baseline "
    "is treated as adverse regardless of whether the absolute value is still "
    "inside the normal range",
)
SPO2_BASELINE_DROP_PCT = Threshold(
    value=3,
    unit="% of baseline",
    source=PROJECT_DEFINED + "; relative form, so a patient whose baseline is "
    "already low is not exempted by the absolute rule",
)
HEART_RATE_BASELINE_RISE = Threshold(
    value=20,
    unit="bpm",
    source=PROJECT_DEFINED + "; sustained tachycardia relative to the patient's "
    "own resting rate",
)
HEART_RATE_BASELINE_RISE_PCT = Threshold(
    value=25,
    unit="% of baseline",
    source=PROJECT_DEFINED,
)
TEMPERATURE_BASELINE_RISE = Threshold(
    value=1.0,
    unit="C",
    source=PROJECT_DEFINED + "; a one-degree rise over a personal baseline that "
    "was itself afebrile",
)
RESPIRATORY_RATE_BASELINE_RISE = Threshold(
    value=4,
    unit="breaths/min",
    source=PROJECT_DEFINED,
)
SLEEP_BASELINE_DROP = Threshold(
    value=2.0,
    unit="hours",
    source=PROJECT_DEFINED + "; wearable-derived, so treated as a supporting "
    "signal rather than a clinical measurement",
)
ACTIVITY_BASELINE_DROP_PCT = Threshold(
    value=50,
    unit="% of baseline",
    source=PROJECT_DEFINED + "; a halving of daily steps is a functional-status "
    "signal, not a vital-sign abnormality",
)

# --- Monitoring: adverse slope -------------------------------------------------
#
# Units per day, again as magnitudes. Doubles as the stability floor: a slope
# smaller than this in either direction is reported as `stable`, so "improving"
# and "worsening" both require movement faster than the noise band.

SPO2_ADVERSE_SLOPE = Threshold(
    value=0.5, unit="percentage points/day", source=PROJECT_DEFINED
)
HEART_RATE_ADVERSE_SLOPE = Threshold(value=3.0, unit="bpm/day", source=PROJECT_DEFINED)
TEMPERATURE_ADVERSE_SLOPE = Threshold(value=0.2, unit="C/day", source=PROJECT_DEFINED)
RESPIRATORY_RATE_ADVERSE_SLOPE = Threshold(
    value=0.5, unit="breaths/min/day", source=PROJECT_DEFINED
)
SLEEP_ADVERSE_SLOPE = Threshold(value=0.3, unit="hours/day", source=PROJECT_DEFINED)
ACTIVITY_ADVERSE_SLOPE = Threshold(value=500, unit="steps/day", source=PROJECT_DEFINED)

# --- Monitoring: trend reliability ---------------------------------------------

MIN_SAMPLES_FOR_TREND = Threshold(
    value=3,
    unit="samples",
    source=PROJECT_DEFINED + "; with two points an ordinary least-squares fit "
    "passes through both and r-squared is 1.0 by construction, which looks like "
    "a perfect trend and is really just two numbers",
)
"""Below this, no slope is reported at all. Reporting a confident trend from two
points is the single easiest way for this agent to be wrong and look right."""

MIN_R_SQUARED_FOR_TREND = Threshold(
    value=0.5,
    source=PROJECT_DEFINED + "; below this the slope is reported but flagged as "
    "a poor fit, because a steep slope through scattered points is noise rather "
    "than deterioration",
)

RAPID_DETERIORATION_MIN_RUN = Threshold(
    value=3,
    unit="consecutive days",
    source=PROJECT_DEFINED + "; a sustained monotonic adverse run, which matters "
    "more than any single jump",
)

# --- Monitoring: personal baseline ---------------------------------------------

BASELINE_WINDOW_SAMPLES = Threshold(
    value=2,
    unit="earliest samples",
    source=PROJECT_DEFINED + "; the mean of the earliest observations in the "
    "window, which for an acute presentation approximates the pre-illness state. "
    "Two rather than one so a single noisy first reading does not become the "
    "patient's baseline",
)
"""The stated method behind every `delta` in a `TrendFeature`. Surfaced in
`MonitoringAssessment.baseline_method` so the delta can be audited rather than
taken on trust."""

# --- Monitoring: anomaly score -------------------------------------------------
#
# Weights for the four components of `TrendFeature.anomaly_score`. They sum to
# 1.0 and a test asserts it, because a score normalised to [0, 1] silently stops
# being normalised when one weight is retuned.

ANOMALY_WEIGHT_THRESHOLD = Threshold(
    value=0.40,
    source=PROJECT_DEFINED + "; the published point-in-time abnormality (NEWS2) "
    "carries the most weight, because it is the only component with external "
    "validation behind it",
)
ANOMALY_WEIGHT_BASELINE = Threshold(
    value=0.25, source=PROJECT_DEFINED + "; deviation from the personal baseline"
)
ANOMALY_WEIGHT_SLOPE = Threshold(value=0.20, source=PROJECT_DEFINED + "; rate of change")
ANOMALY_WEIGHT_RUN = Threshold(
    value=0.15, source=PROJECT_DEFINED + "; length of the sustained adverse run"
)

ANOMALY_RUN_FULL_SCORE = Threshold(
    value=5,
    unit="consecutive days",
    source=PROJECT_DEFINED + "; the run length at which the run component is "
    "fully earned. Shorter runs score proportionally",
)

SUPPORTING_SIGNAL_WEIGHT = Threshold(
    value=0.6,
    source=PROJECT_DEFINED + "; applied to wearable-derived signals (sleep, "
    "activity) after the component weights are renormalised. Without it, "
    "renormalising over only the applicable components lets a 40% sleep decline "
    "outrank an SpO2 of 91%, because sleep has no NEWS2 component to dilute it. "
    "These signals inform functional status; they are not clinical measurements",
)
"""Scales the anomaly score of metrics NEWS2 does not cover.

Necessary because the anomaly score renormalises its component weights over
whichever components apply — without that, a non-NEWS2 metric could never exceed
0.6 and would be structurally unable to rank first no matter how dramatic the
change. Renormalising fixes that but overshoots, so the two corrections are
applied together.
"""

# --- Monitoring: data quality --------------------------------------------------

EXPECTED_SAMPLES_PER_DAY = Threshold(
    value=1, unit="samples/day", source=PROJECT_DEFINED + "; a daily wearable reading"
)

SPARSE_COVERAGE_FRACTION = Threshold(
    value=0.6,
    unit="fraction of days covered",
    comparator="<",
    source=PROJECT_DEFINED + "; below this the window has more gaps than "
    "observations and trends are reported as sparse rather than as fact",
)
