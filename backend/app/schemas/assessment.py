"""Coordinator output and the per-run result container.

There is deliberately **no `diagnosis` field anywhere in this module.** The
vocabulary is confined to risk level, urgency and a recommended department.
That constraint is structural rather than stylistic: a schema with no place to
put a diagnosis cannot leak one, which is a stronger guarantee than a prose
instruction to an LLM. `app/safety.py` adds a lexical check on top.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import Field

from app.reasoning.thresholds import HISTORY_MULTIPLIER_CEILING, HISTORY_MULTIPLIER_FLOOR
from app.schemas.common import (
    AgentStatus,
    ClinicalDimension,
    ImagingSourceMode,
    RetrieverKind,
    RiskLevel,
    RunStatus,
    Schema,
    Significance,
    ChangeType,
    Urgency,
)
from app.schemas.imaging import ImagingFinding
from app.schemas.knowledge import KnowledgeResult, RetrievedEvidence
from app.schemas.monitoring import MonitoringAssessment
from app.schemas.patient import PatientProfile
from app.schemas.symptoms import Symptom, SymptomAssessment
from app.schemas.trace import AgentTrace, FiredRule


class Department(StrEnum):
    EMERGENCY = "Emergency Department"
    RESPIRATORY = "Respiratory Medicine"
    CARDIOLOGY = "Cardiology"
    INTERNAL = "Internal Medicine"
    INFECTIOUS_DISEASE = "Infectious Disease"
    GENERAL_PRACTICE = "General Practice"
    SELF_MONITORING = "Self-monitoring / no immediate visit indicated"


class HistoricalChange(Schema):
    """One axis on which the present differs from this patient's own past.

    This is the model behind the project's central claim. `basis` records how
    the comparison was decided, so "worsening since 2025" is an auditable
    computation over two severity scores rather than a generated sentence.
    """

    dimension: ClinicalDimension
    change_type: ChangeType
    significance: Significance

    prior_state: str | None = Field(
        default=None, description="Normalised description of the historical state. Null when no prior exists."
    )
    current_state: str
    prior_date: date | None = None
    current_date: date | None = None

    prior_severity: int | None = Field(default=None, ge=0, le=5)
    current_severity: int | None = Field(default=None, ge=0, le=5)
    severity_delta: int | None = Field(
        default=None, description="current_severity - prior_severity. The arithmetic behind change_type."
    )

    interval_days: int | None = Field(
        default=None, ge=0, description="Elapsed days between the two observations."
    )
    basis: str = Field(default="", description="How this classification was reached.")


class RunConfig(Schema):
    """What this run was configured to do.

    Persisted with the result so Phase 10 ablations can compare runs that
    differ in exactly one switch, and so the UI can label a result with the
    mode that produced it.
    """

    llm_provider: str = "null"
    llm_enabled: bool = False
    rag_enabled: bool = True
    retriever: RetrieverKind = RetrieverKind.BM25
    imaging_mode: ImagingSourceMode = ImagingSourceMode.MOCK_PRESET
    longitudinal_enabled: bool = True


class FinalAssessment(Schema):
    """Coordinator Agent output."""

    run_id: str
    patient_id: str
    generated_at: datetime

    risk_level: RiskLevel
    risk_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Pre-clamp weighted total; see score_breakdown."
    )
    score_breakdown: list[FiredRule] = Field(
        default_factory=list,
        description="Every scoring rule that fired, with its contribution. Summing "
        "these reproduces risk_score, which is the point: the number is "
        "derivable, not asserted.",
    )
    overrides_applied: list[FiredRule] = Field(
        default_factory=list,
        description="Hard rules that set the level directly, bypassing the score. "
        "Kept separate because an override is a different kind of justification "
        "from an accumulated one.",
    )
    history_multiplier: float = Field(
        default=HISTORY_MULTIPLIER_FLOOR,
        ge=HISTORY_MULTIPLIER_FLOOR,
        le=HISTORY_MULTIPLIER_CEILING,
    )

    recommended_department: Department
    urgency: Urgency
    care_advice: str = Field(
        default="",
        description="Non-prescriptive next-step wording. Never names a drug or a dose."
    )

    key_findings: list[str] = Field(default_factory=list)
    historical_changes: list[HistoricalChange] = Field(default_factory=list)
    longitudinal_summary: str = Field(
        default="", description="The 'past vs present' narrative — the core output of this system."
    )
    reasoning_summary: str = ""

    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    contributing_agents: list[str] = Field(default_factory=list)

    limitations: list[str] = Field(
        default_factory=list,
        description="What this assessment could not determine. Always non-empty in "
        "practice: synthetic data, mock imaging, or absent history each belong here.",
    )
    disclaimer: str = ""


class AnalysisResult(Schema):
    """Everything produced by one run.

    Holds both the typed agent outputs (for the dashboard panels) and the
    `AgentTrace` list (for execution order, timing and debugging). Typed access
    lives here rather than inside the traces so the frontend gets real types
    instead of `object`.
    """

    run_id: str
    patient_id: str
    status: RunStatus
    config: RunConfig

    history: PatientProfile | None = None
    triage: SymptomAssessment | None = None
    imaging: ImagingFinding | None = None
    monitoring: MonitoringAssessment | None = None
    knowledge: KnowledgeResult | None = None
    assessment: FinalAssessment | None = None

    traces: list[AgentTrace] = Field(default_factory=list)
    agent_statuses: dict[str, AgentStatus] = Field(default_factory=dict)

    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int = 0
    error: str | None = None


class AnalysisRequest(Schema):
    """Payload for POST /api/analysis/run.

    All switches default to the safe offline configuration, so a bare
    `{"patient_id": ...}` body runs the complete pipeline.
    """

    patient_id: str
    symptoms: list[Symptom] = Field(default_factory=list)
    free_text: str | None = None
    use_llm: bool | None = Field(
        default=None, description="Null means 'use the server default'. True cannot override a missing API key."
    )
    rag_enabled: bool = True
    longitudinal_enabled: bool = True
    imaging_mode: ImagingSourceMode | None = None
    radiology_report_text: str | None = Field(
        default=None, description="Required when imaging_mode is uploaded_report."
    )
