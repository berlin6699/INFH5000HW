"""History Agent contracts: the patient's longitudinal record.

The History Agent's job is not to summarise notes into prose. It is to
normalise an heterogeneous record (diagnoses, medications, prior imaging
reports, encounters) onto `ClinicalDimension` axes so the Coordinator can later
compare a past state against a present one arithmetically.
`PriorFinding.severity_score` exists for exactly that purpose.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from app.reasoning.thresholds import HISTORY_MULTIPLIER_CEILING, HISTORY_MULTIPLIER_FLOOR
from app.schemas.common import Acuity, ClinicalDimension, Schema


class TimelineEventType(StrEnum):
    CONDITION = "condition"
    IMAGING = "imaging"
    LAB = "lab"
    MEDICATION = "medication"
    ENCOUNTER = "encounter"
    CURRENT = "current"


class Condition(Schema):
    name: str
    icd10: str | None = None
    onset_date: date | None = None
    is_chronic: bool = False
    is_active: bool = True
    affects_dimension: ClinicalDimension | None = Field(
        default=None,
        description="Maps this condition onto a comparison axis, which is what "
        "lets a 2024 diagnosis inform a 2026 imaging comparison.",
    )
    notes: str | None = None


class Medication(Schema):
    name: str
    dose: str | None = None
    frequency: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    is_active: bool = True


class Allergy(Schema):
    allergen: str
    reaction: str | None = None
    severity: Literal["mild", "moderate", "severe", "unknown"] = "unknown"


class SocialHistory(Schema):
    """Structured social history: smoking exposure, body habitus.

    Typed rather than a free-form dict so the risk-factor rules read numbers.
    The alternative is parsing "~8 pack-years, quit 2018" as prose, which is
    exactly the kind of implicit inference this project is trying to avoid.
    """

    key: str = Field(description="Stable identifier, e.g. 'former_smoker'.")
    factor: str = Field(description="Display label, e.g. 'Former smoker'.")
    detail: str | None = Field(default=None, description="Original human-readable wording.")

    current_smoker: bool | None = None
    pack_years: float | None = Field(default=None, ge=0)
    quit_year: int | None = None
    bmi: float | None = Field(default=None, ge=5, le=100)


class PriorFinding(Schema):
    """A historical observation, normalised onto a comparison dimension."""

    dimension: ClinicalDimension
    label: str = Field(description="Normalised finding name, e.g. 'focal_opacity'.")
    description: str | None = Field(
        default=None, description="Original wording from the source record."
    )
    occurred_on: date
    location: str | None = Field(default=None, description="Anatomical site, e.g. 'RLL'.")
    acuity: Acuity = Acuity.INDETERMINATE
    severity_score: int = Field(
        default=0,
        ge=0,
        le=5,
        description="0=absent … 5=life-threatening. This ordinal scale is what "
        "makes two observations of the same dimension taken years apart "
        "directly comparable; without it, change detection falls back to "
        "string similarity.",
    )
    resolved: bool = Field(
        default=False,
        description="True if a later record shows this finding cleared. A "
        "resolved finding that reappears is classified RECURRENT, not NEW.",
    )
    source_record_id: str | None = None


class RiskFactor(Schema):
    rule_id: str = Field(
        description="Identifier of the rule in reasoning/history_rules.py that "
        "produced this factor, e.g. R-HIS-AGE-01. Ties the profile entry to its "
        "FiredRule so the UI anchors on an id rather than on label text.",
    )
    factor: str
    category: Literal["demographic", "comorbidity", "lifestyle", "historical", "medication"]
    weight: float = Field(
        default=1.0,
        ge=1.0,
        le=1.5,
        description="Multiplicative modifier applied to the risk score. Capped "
        "so no single factor can dominate the assessment. A weight of exactly "
        "1.0 means the factor was considered and did not adjust the score, "
        "which is reported rather than omitted.",
    )
    source: str | None = None


class TimelineEvent(Schema):
    """One row in the longitudinal timeline rendered by the dashboard."""

    occurred_on: date
    event_type: TimelineEventType
    label: str
    dimension: ClinicalDimension | None = None
    detail: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    sort_order: int = 0
    is_current: bool = Field(
        default=False, description="True for the 'Today' cluster at the timeline head."
    )


class Patient(Schema):
    """Demographics only — the list/selection view of a patient."""

    id: str
    full_name: str
    birth_year: int
    sex: Literal["male", "female", "other", "unknown"] = "unknown"


class PatientProfile(Schema):
    """History Agent output."""

    patient_id: str
    full_name: str
    age: int
    sex: Literal["male", "female", "other", "unknown"] = "unknown"

    chronic_conditions: list[Condition] = Field(default_factory=list)
    active_medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    social_history: list[SocialHistory] = Field(
        default_factory=list,
        description="Structured smoking/BMI data. Kept alongside the derived "
        "risk_factors so the UI can show the number behind the label.",
    )
    previous_findings: list[PriorFinding] = Field(default_factory=list)
    risk_factors: list[RiskFactor] = Field(default_factory=list)

    baseline_vulnerabilities: list[ClinicalDimension] = Field(
        default_factory=list,
        description="Dimensions where prior history makes this patient more "
        "fragile than baseline. Consumed by the Coordinator as risk multipliers.",
    )
    history_multiplier: float = Field(
        default=HISTORY_MULTIPLIER_FLOOR,
        ge=HISTORY_MULTIPLIER_FLOOR,
        le=HISTORY_MULTIPLIER_CEILING,
        description="Product of applicable risk-factor weights, clamped to "
        "HISTORY_MULTIPLIER_CEILING in reasoning/thresholds.py.",
    )

    timeline: list[TimelineEvent] = Field(default_factory=list)
    data_gaps: list[str] = Field(
        default_factory=list,
        description="Relevant history that is absent. Surfacing gaps is part of "
        "the output: a confident assessment built on an incomplete record is "
        "worse than one that declares what it could not see.",
    )
    summary: str = ""
