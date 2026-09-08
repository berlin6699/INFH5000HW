"""Shared vocabulary and the base model for every agent contract.

Enums are declared once here rather than as inline `Literal` unions scattered
across modules. Two reasons: the JSON Schema export that drives
`frontend/src/types/generated.ts` produces a single named type per concept
instead of duplicated anonymous unions, and the rule registry in
`reasoning/rules.py` can key off the same symbols the schemas use.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    """Base for all wire contracts.

    `extra="forbid"` is deliberate: agents exchange structured JSON rather than
    free text, so an unexpected field almost always means two modules have
    drifted apart. Failing at construction surfaces that immediately instead of
    letting a typo'd field silently vanish.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentName(StrEnum):
    HISTORY = "history"
    TRIAGE = "triage"
    IMAGING = "imaging"
    MONITORING = "monitoring"
    KNOWLEDGE = "knowledge"
    COORDINATOR = "coordinator"


class AgentStatus(StrEnum):
    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Urgency(StrEnum):
    IMMEDIATE = "IMMEDIATE"
    URGENT = "URGENT"
    SEMI_URGENT = "SEMI_URGENT"
    ROUTINE = "ROUTINE"


class Severity(StrEnum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class Acuity(StrEnum):
    """Whether an imaging finding is new/active or long-standing.

    This distinction drives both risk scoring (acute findings weigh far more)
    and longitudinal comparison (a chronic finding that is unchanged is
    reassuring, an acute one is not).
    """

    ACUTE = "acute"
    CHRONIC = "chronic"
    INDETERMINATE = "indeterminate"


class ChangeType(StrEnum):
    """Result of comparing one clinical dimension against the patient's history.

    RECURRENT is modelled separately from NEW because a finding that previously
    resolved and has come back carries different clinical meaning from one
    never seen before, even though both are absent from the prior record.
    """

    NEW = "new"
    WORSENING = "worsening"
    IMPROVING = "improving"
    STABLE = "stable"
    RECURRENT = "recurrent"


class Significance(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class TrendDirection(StrEnum):
    """Direction of a vital over the observation window.

    `UNKNOWN` is not a synonym for `STABLE`. A series too short to fit has no
    measured direction, and reporting it as stable would assert a flat trend
    that nothing ever observed — the failure mode this agent exists to avoid.
    """

    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    UNKNOWN = "unknown"


class ClinicalDimension(StrEnum):
    """The fixed axes on which past and present are made comparable.

    This enum is the mechanism behind longitudinal reasoning. A 2025 radiology
    report and a 2026 imaging result are not compared as text; both are mapped
    onto the same dimension, and the comparison happens between normalised
    states on that axis. Without a closed vocabulary here, "change since last
    time" degenerates into string similarity.
    """

    PULMONARY_IMAGING = "pulmonary_imaging"
    OXYGENATION = "oxygenation"
    CARDIAC_RATE = "cardiac_rate"
    TEMPERATURE = "temperature"
    RESPIRATORY_RATE = "respiratory_rate"
    SYMPTOM_BURDEN = "symptom_burden"
    FUNCTIONAL_STATUS = "functional_status"


class VitalMetric(StrEnum):
    SPO2 = "spo2"
    HEART_RATE = "heart_rate"
    TEMPERATURE_C = "temperature_c"
    RESPIRATORY_RATE = "respiratory_rate"
    SLEEP_HOURS = "sleep_hours"
    ACTIVITY_STEPS = "activity_steps"


class ClinicalDomain(StrEnum):
    RESPIRATORY = "respiratory"
    CARDIAC = "cardiac"
    INFECTIOUS = "infectious"
    NEUROLOGIC = "neurologic"
    GASTROINTESTINAL = "gastrointestinal"
    OTHER = "other"


class WarningSignSeverity(StrEnum):
    """MAJOR signs can force a HIGH risk override on their own; MINOR cannot."""

    MAJOR = "major"
    MINOR = "minor"


class ImagingModality(StrEnum):
    """Chest X-ray only. CT/MRI are explicitly out of MVP scope."""

    CHEST_XRAY = "chest_xray"


class ImagingSourceMode(StrEnum):
    """Provenance of an imaging result. Required, never inferred.

    MOCK_PRESET output must always be badged as demo data in the UI. The system
    never presents a predefined finding as though a model produced it.
    """

    REAL_MODEL = "real_model"
    MOCK_PRESET = "mock_preset"
    UPLOADED_REPORT = "uploaded_report"


class RetrieverKind(StrEnum):
    BM25 = "bm25"
    EMBEDDING = "embedding"
