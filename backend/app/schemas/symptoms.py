"""Triage Agent contracts: current symptom presentation.

The Triage Agent produces a *preliminary* read only. `preliminary_urgency` is
never the final answer — the Coordinator owns `risk_level` and `urgency`,
because a symptom set that looks moderate in isolation can be HIGH once a
falling SpO₂ trend and a prior pulmonary finding are added.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from app.schemas.common import (
    ClinicalDomain,
    Schema,
    Severity,
    Urgency,
    WarningSignSeverity,
)


class SymptomName(StrEnum):
    """Closed vocabulary for the MVP respiratory scope.

    Exact enum matching (rather than fuzzy string matching over user text) is
    what makes the warning-sign rules deterministic and testable. Anything
    outside the vocabulary arrives as OTHER with the original wording in
    `Symptom.label`, so the rule engine never silently ignores input it failed
    to recognise.
    """

    FEVER = "fever"
    COUGH = "cough"
    DYSPNEA = "dyspnea"
    CHEST_PAIN = "chest_pain"
    CHEST_TIGHTNESS = "chest_tightness"
    SPUTUM = "sputum"
    HEMOPTYSIS = "hemoptysis"
    CHILLS = "chills"
    WHEEZE = "wheeze"
    SORE_THROAT = "sore_throat"
    FATIGUE = "fatigue"
    PALPITATIONS = "palpitations"
    DIZZINESS = "dizziness"
    CONFUSION = "confusion"
    CYANOSIS = "cyanosis"
    ORTHOPNEA = "orthopnea"
    NAUSEA = "nausea"
    OTHER = "other"


class Symptom(Schema):
    name: SymptomName
    label: str | None = Field(
        default=None,
        description="Display text, and the only carrier of the original wording "
        "when name is OTHER.",
    )
    present: bool = True
    value: float | None = Field(
        default=None, description="Measured value where the symptom is quantitative, e.g. 38.7."
    )
    unit: str | None = Field(default=None, description="C or mmHg or similar; required if value is set.")
    severity: Severity = Severity.MILD
    onset_days: int | None = Field(
        default=None, ge=0, description="Days since onset. Null when the patient cannot recall."
    )
    is_progressive: bool = Field(
        default=False, description="True if the patient reports it getting worse, not just persisting."
    )
    notes: str | None = None


class SymptomInput(Schema):
    """Payload accepted by POST /api/patients/{id}/symptoms."""

    symptoms: list[Symptom] = Field(default_factory=list)
    free_text: str | None = Field(
        default=None,
        description="Optional chief complaint in the patient's own words. Parsed "
        "into `symptoms` only when an LLM provider is configured; in offline "
        "mode it is recorded verbatim and flagged as unparsed rather than "
        "guessed at.",
    )
    reported_at: datetime | None = None


class WarningSign(Schema):
    sign: str
    severity: WarningSignSeverity
    rule_id: str
    description: str
    source: str = Field(
        default="symptom_report",
        description="Where the sign was observed: symptom_report | monitoring | imaging.",
    )
    observed_value: float | None = None
    threshold: float | None = None


class SymptomAssessment(Schema):
    """Triage Agent output."""

    symptoms: list[Symptom] = Field(default_factory=list)
    warning_signs: list[WarningSign] = Field(default_factory=list)

    dominant_domain: ClinicalDomain = ClinicalDomain.OTHER
    affected_domains: list[ClinicalDomain] = Field(default_factory=list)

    respiratory_symptom_count: int = 0
    has_progressive_symptoms: bool = False
    longest_duration_days: int | None = None

    missing_information: list[str] = Field(
        default_factory=list,
        description="Clinically relevant fields that were not supplied.",
    )
    followup_questions: list[str] = Field(
        default_factory=list,
        description="Questions worth asking before a clinician consultation.",
    )
    unparsed_free_text: str | None = Field(
        default=None,
        description="Echoed back when free_text could not be structurally parsed, "
        "so nothing is quietly dropped.",
    )

    preliminary_urgency: Urgency = Urgency.ROUTINE
    summary: str = ""
