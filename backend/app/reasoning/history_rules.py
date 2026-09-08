"""History Agent rule engine.

Pure functions over plain data. Nothing here imports SQLAlchemy, so every rule
is testable without a database and the same rules apply when Phase 10 swaps in
alternative patients.

Three things come out of a patient's record and all three are derived rather
than asserted:

* `risk_factors` — the modifiers, each naming the rule that produced it.
* `baseline_vulnerabilities` — the `ClinicalDimension` axes where this patient's
  own history makes them more fragile than a blank-slate patient. The
  Coordinator uses these to decide which monitoring deviations matter more.
* `data_gaps` — what the record does not contain.

`data_gaps` is not a courtesy. An assessment that silently ignored an absent
prior study would look identical to one that had a prior study and found it
unchanged, and those two mean very different things clinically.

Vulnerabilities are never inferred across dimensions: a prior pulmonary finding
flags `pulmonary_imaging`, not `oxygenation`. Crossing that line would be the
system reasoning from plausibility rather than from the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.reasoning.thresholds import (
    AGE_ELDERLY,
    AGE_MIDDLE,
    BMI_OBESE,
    BMI_OVERWEIGHT,
    HISTORY_MULTIPLIER_CEILING,
    HISTORY_MULTIPLIER_FLOOR,
    NO_PRIOR_IMAGING_DAYS,
    PACK_YEARS_ANY,
    PACK_YEARS_SIGNIFICANT,
    PROJECT_DEFINED,
    STALE_HISTORY_MONTHS,
    WEIGHT_AGE_ELDERLY,
    WEIGHT_AGE_MIDDLE,
    WEIGHT_BMI_OBESE,
    WEIGHT_BMI_OVERWEIGHT,
    WEIGHT_CHRONIC_CARDIAC,
    WEIGHT_CHRONIC_RESPIRATORY,
    WEIGHT_PRIOR_PULMONARY_FINDING,
    WEIGHT_SMOKING_LIGHT,
    WEIGHT_SMOKING_SIGNIFICANT,
)
from app.schemas import (
    Allergy,
    ClinicalDimension,
    Condition,
    FiredRule,
    Medication,
    PriorFinding,
    RiskFactor,
    SocialHistory,
)

RESPIRATORY_DIMENSIONS = frozenset({
    ClinicalDimension.PULMONARY_IMAGING,
    ClinicalDimension.OXYGENATION,
    ClinicalDimension.RESPIRATORY_RATE,
})
CARDIAC_DIMENSIONS = frozenset({ClinicalDimension.CARDIAC_RATE})

HISTORY_RULE_CATEGORY = "history"
MULTIPLIER_RULE_ID = "R-HIS-MULT-01"

# Mean month length. Used only to compare a record's age against
# STALE_HISTORY_MONTHS, where a day or two of drift cannot change the answer.
DAYS_PER_MONTH = 30.4375


@dataclass(frozen=True)
class HistoryInputs:
    """Everything the history rules need, already lifted out of the ORM.

    Deliberately plain data: `age` rather than `birth_year`, resolved dates
    rather than queries. The agent owns the database, the rules own the logic.
    """

    reference_date: date
    age: int | None
    chronic_conditions: tuple[Condition, ...] = ()
    previous_findings: tuple[PriorFinding, ...] = ()
    social_history: tuple[SocialHistory, ...] = ()
    active_medications: tuple[Medication, ...] = ()
    allergies: tuple[Allergy, ...] = ()
    last_record_date: date | None = None
    last_prior_imaging_date: date | None = None
    monitoring_sample_count: int = 0


@dataclass(frozen=True)
class HistoryDerivation:
    risk_factors: tuple[RiskFactor, ...]
    fired_rules: tuple[FiredRule, ...]
    baseline_vulnerabilities: tuple[ClinicalDimension, ...]
    data_gaps: tuple[str, ...]
    history_multiplier: float


def _months_between(earlier: date, later: date) -> float:
    return max(0.0, (later - earlier).days) / DAYS_PER_MONTH


# --- Risk factors -------------------------------------------------------------


def _age_factor(age: int | None) -> RiskFactor | None:
    if age is None:
        return None
    if AGE_ELDERLY.is_met_by(age):
        return RiskFactor(
            rule_id="R-HIS-AGE-01",
            factor=f"Age {age} ({AGE_ELDERLY.describe()})",
            category="demographic",
            weight=WEIGHT_AGE_ELDERLY.value,
            source=WEIGHT_AGE_ELDERLY.source,
        )
    if AGE_MIDDLE.is_met_by(age):
        return RiskFactor(
            rule_id="R-HIS-AGE-02",
            factor=f"Age {age} ({AGE_MIDDLE.describe()})",
            category="demographic",
            weight=WEIGHT_AGE_MIDDLE.value,
            source=WEIGHT_AGE_MIDDLE.source,
        )
    return None


def _comorbidity_factors(conditions: tuple[Condition, ...]) -> list[RiskFactor]:
    respiratory = sorted({c.name for c in conditions if c.affects_dimension in RESPIRATORY_DIMENSIONS})
    cardiac = sorted({c.name for c in conditions if c.affects_dimension in CARDIAC_DIMENSIONS})

    factors = []
    if respiratory:
        factors.append(RiskFactor(
            rule_id="R-HIS-COMORB-RESP-01",
            factor=f"Chronic respiratory condition: {', '.join(respiratory)}",
            category="comorbidity",
            weight=WEIGHT_CHRONIC_RESPIRATORY.value,
            source=WEIGHT_CHRONIC_RESPIRATORY.source,
        ))
    if cardiac:
        factors.append(RiskFactor(
            rule_id="R-HIS-COMORB-CARD-01",
            factor=f"Chronic cardiac condition: {', '.join(cardiac)}",
            category="comorbidity",
            weight=WEIGHT_CHRONIC_CARDIAC.value,
            source=WEIGHT_CHRONIC_CARDIAC.source,
        ))
    return factors


def _prior_finding_factors(findings: tuple[PriorFinding, ...]) -> list[RiskFactor]:
    pulmonary = [f for f in findings if f.dimension == ClinicalDimension.PULMONARY_IMAGING]
    if not pulmonary:
        return []

    described = ", ".join(
        f"{f.label}{' (resolved)' if f.resolved else ''}"
        for f in sorted(pulmonary, key=lambda f: f.occurred_on)
    )
    return [RiskFactor(
        rule_id="R-HIS-PRIOR-PULM-01",
        factor=f"Prior pulmonary imaging abnormality: {described}",
        category="historical",
        weight=WEIGHT_PRIOR_PULMONARY_FINDING.value,
        source=WEIGHT_PRIOR_PULMONARY_FINDING.source,
    )]


def _exposure_text(entry: SocialHistory) -> str:
    """Describe smoking exposure without inventing a number it does not have.

    A record may say 'current smoker' with no pack-years quantified; formatting
    None as a number would either crash or silently become zero.
    """
    if entry.pack_years is None:
        return "exposure not quantified"
    return f"{entry.pack_years:g} pack-years"


def _smoking_factor(entry: SocialHistory) -> RiskFactor | None:
    pack_years = entry.pack_years
    exposure = _exposure_text(entry)
    status = "current smoker" if entry.current_smoker else "former smoker"

    if PACK_YEARS_SIGNIFICANT.is_met_by(pack_years):
        rule_id, weight, band = "R-HIS-SMOKE-01", WEIGHT_SMOKING_SIGNIFICANT, PACK_YEARS_SIGNIFICANT
    elif PACK_YEARS_ANY.is_met_by(pack_years) or entry.current_smoker:
        rule_id, weight, band = "R-HIS-SMOKE-02", WEIGHT_SMOKING_LIGHT, PACK_YEARS_ANY
    else:
        # Considered and deliberately not weighted. Recording it at 1.0 keeps the
        # factor visible in the UI with the reason it changed nothing, which is
        # more informative than omitting it and leaving the reader to wonder
        # whether smoking was ever checked.
        return RiskFactor(
            rule_id="R-HIS-SMOKE-03",
            factor=(
                f"{entry.factor} ({exposure}) — below the "
                f"{PACK_YEARS_ANY.value:g} pack-year band; no adjustment applied"
            ),
            category="lifestyle",
            weight=HISTORY_MULTIPLIER_FLOOR,
            source=PACK_YEARS_ANY.source,
        )

    return RiskFactor(
        rule_id=rule_id,
        factor=f"Smoking history — {status}, {exposure} ({band.describe()})",
        category="lifestyle",
        weight=weight.value,
        source=weight.source,
    )


def _bmi_factor(entry: SocialHistory) -> RiskFactor | None:
    bmi = entry.bmi
    if bmi is None:
        return None
    if BMI_OBESE.is_met_by(bmi):
        weight, band = WEIGHT_BMI_OBESE, BMI_OBESE
        rule_id = "R-HIS-BMI-01"
    elif BMI_OVERWEIGHT.is_met_by(bmi):
        weight, band = WEIGHT_BMI_OVERWEIGHT, BMI_OVERWEIGHT
        rule_id = "R-HIS-BMI-02"
    else:
        return None
    return RiskFactor(
        rule_id=rule_id,
        factor=f"BMI {bmi:g} kg/m^2 ({band.describe()})",
        category="lifestyle",
        weight=weight.value,
        source=weight.source,
    )


def _social_factors(entries: tuple[SocialHistory, ...]) -> list[RiskFactor]:
    factors = []
    for entry in entries:
        if entry.pack_years is not None or entry.current_smoker is not None:
            smoking = _smoking_factor(entry)
            if smoking:
                factors.append(smoking)
        bmi = _bmi_factor(entry)
        if bmi:
            factors.append(bmi)
    return factors


def derive_risk_factors(inputs: HistoryInputs) -> list[RiskFactor]:
    factors: list[RiskFactor] = []
    age = _age_factor(inputs.age)
    if age:
        factors.append(age)
    factors.extend(_comorbidity_factors(inputs.chronic_conditions))
    factors.extend(_prior_finding_factors(inputs.previous_findings))
    factors.extend(_social_factors(inputs.social_history))
    return factors


def compute_history_multiplier(factors: list[RiskFactor] | tuple[RiskFactor, ...]) -> float:
    """Product of the weights, clamped to the shared bounds.

    The clamp is what stops a long comorbidity list from dominating the score:
    with five 1.15 modifiers the raw product is already 2.0.
    """
    product = 1.0
    for f in factors:
        product *= f.weight
    clamped = min(max(product, HISTORY_MULTIPLIER_FLOOR), HISTORY_MULTIPLIER_CEILING)
    return round(clamped, 4)


# --- Vulnerabilities and gaps -------------------------------------------------


def derive_baseline_vulnerabilities(inputs: HistoryInputs) -> list[ClinicalDimension]:
    """Dimensions with direct evidence in this patient's own record.

    Iterated in enum order so the output is stable across runs — the dashboard
    renders this list and a changing order would look like a changing patient.
    """
    evidenced: set[ClinicalDimension] = set()
    for condition in inputs.chronic_conditions:
        if condition.affects_dimension:
            evidenced.add(condition.affects_dimension)
    for finding in inputs.previous_findings:
        evidenced.add(finding.dimension)
    return [d for d in ClinicalDimension if d in evidenced]


def derive_data_gaps(inputs: HistoryInputs) -> list[str]:
    """What the record does not contain, stated as a limitation."""
    gaps: list[str] = []

    if not inputs.chronic_conditions and not inputs.previous_findings:
        gaps.append(
            "No prior conditions or findings on record, so there is no personal "
            "baseline to compare the current presentation against."
        )

    if inputs.last_prior_imaging_date is None:
        gaps.append(
            "No prior chest imaging on record; the current study has no "
            "comparison anchor, so 'unchanged' and 'new' cannot be distinguished."
        )
    elif NO_PRIOR_IMAGING_DAYS.is_met_by(
        (inputs.reference_date - inputs.last_prior_imaging_date).days
    ):
        days = (inputs.reference_date - inputs.last_prior_imaging_date).days
        gaps.append(
            f"Most recent prior chest imaging is {days} days old "
            f"({inputs.last_prior_imaging_date.isoformat()}); the longitudinal "
            f"comparison spans a long interval."
        )

    if inputs.monitoring_sample_count == 0:
        gaps.append(
            "No continuous monitoring data, so time-series trend and deviation "
            "from personal baseline cannot be computed."
        )

    if inputs.last_record_date is None:
        gaps.append("No dated clinical records on file.")
    elif STALE_HISTORY_MONTHS.is_met_by(
        _months_between(inputs.last_record_date, inputs.reference_date)
    ):
        months = _months_between(inputs.last_record_date, inputs.reference_date)
        gaps.append(
            f"Most recent clinical record is {months:.0f} months old "
            f"({inputs.last_record_date.isoformat()}); intervening history may be missing."
        )

    if not inputs.active_medications:
        gaps.append(
            "No active medications recorded — this may be genuine or may be an "
            "incomplete medication list."
        )

    if not inputs.allergies:
        gaps.append(
            "No allergies recorded. An empty allergy list is not evidence of no "
            "allergy, and is flagged rather than treated as reassuring."
        )

    if not any(s.pack_years is not None or s.current_smoker is not None for s in inputs.social_history):
        gaps.append("No smoking history recorded.")
    if not any(s.bmi is not None for s in inputs.social_history):
        gaps.append("No BMI recorded.")

    return gaps


# --- Trace --------------------------------------------------------------------


def history_fired_rules(
    factors: list[RiskFactor] | tuple[RiskFactor, ...],
    history_multiplier: float,
) -> list[FiredRule]:
    """One rule per factor, plus a summary rule carrying the combined multiplier.

    `contribution` is 0.0 throughout: history modifies the score
    multiplicatively rather than adding points to it, and recording a fake
    additive contribution here would break the property that summing
    `score_breakdown` reproduces `risk_score`.
    """
    rules = [
        FiredRule(
            rule_id=f.rule_id,
            category=HISTORY_RULE_CATEGORY,
            description=f.factor,
            contribution=0.0,
            evidence=(
                f"weight x{f.weight:g}"
                + (f" — {f.source}" if f.source else "")
            ),
            value=f.weight,
        )
        for f in factors
    ]

    active = [f for f in factors if f.weight != HISTORY_MULTIPLIER_FLOOR]
    product = " x ".join(f"{f.weight:g}" for f in active) or "1"
    clamped = history_multiplier >= HISTORY_MULTIPLIER_CEILING - 1e-9
    rules.append(FiredRule(
        rule_id=MULTIPLIER_RULE_ID,
        category=HISTORY_RULE_CATEGORY,
        description=(
            f"Combined history multiplier x{history_multiplier:g}"
            + (" (clamped at ceiling)" if clamped else "")
        ),
        contribution=0.0,
        evidence=(
            f"{len(active)} of {len(factors)} factors adjusted the multiplier: "
            f"{product} = {history_multiplier:g}, clamped to "
            f"[{HISTORY_MULTIPLIER_FLOOR:g}, {HISTORY_MULTIPLIER_CEILING:g}]"
            f" ({PROJECT_DEFINED})"
        ),
        value=history_multiplier,
        threshold=HISTORY_MULTIPLIER_CEILING,
    ))
    return rules


def derive_history(inputs: HistoryInputs) -> HistoryDerivation:
    """Single entry point used by the History Agent."""
    factors = derive_risk_factors(inputs)
    multiplier = compute_history_multiplier(factors)
    return HistoryDerivation(
        risk_factors=tuple(factors),
        fired_rules=tuple(history_fired_rules(factors, multiplier)),
        baseline_vulnerabilities=tuple(derive_baseline_vulnerabilities(inputs)),
        data_gaps=tuple(derive_data_gaps(inputs)),
        history_multiplier=multiplier,
    )
