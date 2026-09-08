"""Public surface of the schema package.

Agents, the workflow and the API layer import from `app.schemas` rather than
from individual modules, so the contract surface is visible in one place.
"""

from __future__ import annotations

from app.schemas.assessment import (
    AnalysisRequest,
    AnalysisResult,
    Department,
    FinalAssessment,
    HistoricalChange,
    RunConfig,
)
from app.schemas.common import (
    Acuity,
    AgentName,
    AgentStatus,
    ChangeType,
    ClinicalDimension,
    ClinicalDomain,
    ImagingModality,
    ImagingSourceMode,
    RetrieverKind,
    RiskLevel,
    RunStatus,
    Schema,
    Severity,
    Significance,
    TrendDirection,
    Urgency,
    VitalMetric,
    WarningSignSeverity,
)
from app.schemas.imaging import (
    MOCK_BADGE_TEXT,
    REPORT_BADGE_TEXT,
    Abnormality,
    ImagingFinding,
    ImagingLabel,
)
from app.schemas.knowledge import KnowledgeResult, RetrievedEvidence
from app.schemas.monitoring import MonitoringAssessment, TrendFeature, VitalSample
from app.schemas.patient import (
    Allergy,
    Condition,
    Medication,
    Patient,
    PatientProfile,
    PriorFinding,
    RiskFactor,
    SocialHistory,
    TimelineEvent,
    TimelineEventType,
)
from app.schemas.symptoms import (
    Symptom,
    SymptomAssessment,
    SymptomInput,
    SymptomName,
    WarningSign,
)
from app.schemas.trace import AgentTrace, FiredRule, utcnow

__all__ = [
    # assessment
    "AnalysisRequest",
    "AnalysisResult",
    "Department",
    "FinalAssessment",
    "HistoricalChange",
    "RunConfig",
    # common
    "Acuity",
    "AgentName",
    "AgentStatus",
    "ChangeType",
    "ClinicalDimension",
    "ClinicalDomain",
    "ImagingModality",
    "ImagingSourceMode",
    "RetrieverKind",
    "RiskLevel",
    "RunStatus",
    "Schema",
    "Severity",
    "Significance",
    "TrendDirection",
    "Urgency",
    "VitalMetric",
    "WarningSignSeverity",
    # imaging
    "MOCK_BADGE_TEXT",
    "REPORT_BADGE_TEXT",
    "Abnormality",
    "ImagingFinding",
    "ImagingLabel",
    # knowledge
    "KnowledgeResult",
    "RetrievedEvidence",
    # monitoring
    "MonitoringAssessment",
    "TrendFeature",
    "VitalSample",
    # patient
    "Allergy",
    "Condition",
    "Medication",
    "Patient",
    "PatientProfile",
    "PriorFinding",
    "RiskFactor",
    "SocialHistory",
    "TimelineEvent",
    "TimelineEventType",
    # symptoms
    "Symptom",
    "SymptomAssessment",
    "SymptomInput",
    "SymptomName",
    "WarningSign",
    # trace
    "AgentTrace",
    "FiredRule",
    "utcnow",
]
