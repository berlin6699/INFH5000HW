"""ORM models. SQLite-backed, string primary keys, JSON columns for payloads.

Design notes:

* **String IDs, not integers.** Human-readable identifiers (`PT-DEMO-001`,
  `RUN-2026-09-08-a1b2`) make agent traces and logs legible during a demo,
  which matters more here than insert throughput.
* **Enums stored as String.** SQLite has no enum type, and SQLAlchemy's `Enum`
  emits DDL that is awkward to alter. Validation happens in the pydantic layer,
  which is where the contracts live.
* **JSON columns for agent payloads.** `agent_results` stores the full input and
  output of every agent for every run. That is what makes "why did the AI decide
  this?" answerable after the fact rather than only while a request is live.
* **Every JSON column is `JSONNoneAsNull`.** Several of these payloads are
  legitimately absent for some rows, and "absent" has to be queryable. See
  `app/db/types.py` — the SQLAlchemy default stores `None` as `'null'`, which
  makes `IS NULL` match nothing.
* **`UTCDateTime` for every timestamp.** `DateTime(timezone=True)` loses the UTC
  offset on SQLite and returns naive datetimes, which cannot be compared against
  the aware values this system computes. See `app/db/types.py`.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.db.types import JSONNoneAsNull, UTCDateTime


class PatientRow(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200))
    birth_year: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(20), default="unknown")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)

    medical_records: Mapped[list[MedicalRecordRow]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", order_by="MedicalRecordRow.occurred_on"
    )
    medications: Mapped[list[MedicationRow]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    allergies: Mapped[list[AllergyRow]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    timeline_events: Mapped[list[TimelineEventRow]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", order_by="TimelineEventRow.sort_order"
    )
    images: Mapped[list[MedicalImageRow]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", order_by="MedicalImageRow.study_date"
    )
    monitoring_records: Mapped[list[MonitoringRecordRow]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", order_by="MonitoringRecordRow.recorded_at"
    )
    runs: Mapped[list[AnalysisRunRow]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class MedicalRecordRow(Base):
    """A historical clinical event: diagnosis, surgery, encounter, lab or report.

    `occurred_on` is nullable because social history (smoking status, BMI) is
    clinically relevant context with no event date. Inventing one would
    misrepresent the record on the timeline.
    """

    __tablename__ = "medical_records"
    __table_args__ = (Index("ix_medrec_patient_date", "patient_id", "occurred_on"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    occurred_on: Mapped[date | None] = mapped_column(nullable=True)
    category: Mapped[str] = mapped_column(String(32))
    code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    title: Mapped[str] = mapped_column(String(300))
    detail: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)
    is_chronic: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    clinical_dimension: Mapped[str | None] = mapped_column(String(40), nullable=True)

    # The coded form of this record's clinical finding: label, acuity,
    # severity_score, resolved. Kept separate from `detail` because `detail` is
    # source-specific extras while `normalised` is the shape the History Agent
    # consumes. Storing it rather than re-deriving it means the agent never has
    # to parse radiology prose — that job belongs to the Phase 8 Imaging Agent,
    # and its output lands in this same column.
    normalised: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)

    patient: Mapped[PatientRow] = relationship(back_populates="medical_records")


class MedicationRow(Base):
    __tablename__ = "medications"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    dose: Mapped[str | None] = mapped_column(String(80), nullable=True)
    frequency: Mapped[str | None] = mapped_column(String(80), nullable=True)
    start_date: Mapped[date | None] = mapped_column(nullable=True)
    end_date: Mapped[date | None] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    patient: Mapped[PatientRow] = relationship(back_populates="medications")


class AllergyRow(Base):
    __tablename__ = "allergies"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    allergen: Mapped[str] = mapped_column(String(200))
    reaction: Mapped[str | None] = mapped_column(String(300), nullable=True)
    severity: Mapped[str] = mapped_column(String(20), default="unknown")

    patient: Mapped[PatientRow] = relationship(back_populates="allergies")


class TimelineEventRow(Base):
    """Materialised timeline, written by the History Agent and read straight by the UI."""

    __tablename__ = "timeline_events"
    __table_args__ = (Index("ix_timeline_patient_order", "patient_id", "sort_order"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    occurred_on: Mapped[date] = mapped_column()
    event_type: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(400))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    clinical_dimension: Mapped[str | None] = mapped_column(String(40), nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)

    patient: Mapped[PatientRow] = relationship(back_populates="timeline_events")


class MedicalImageRow(Base):
    __tablename__ = "medical_images"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    study_date: Mapped[date | None] = mapped_column(nullable=True)
    modality: Mapped[str] = mapped_column(String(32), default="chest_xray")
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    report_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Provenance is persisted, not recomputed, so a result saved as mock stays
    # labelled mock even if the server's configured mode later changes.
    source_mode: Mapped[str] = mapped_column(String(32))
    provenance: Mapped[str | None] = mapped_column(Text, nullable=True)
    badge: Mapped[str | None] = mapped_column(Text, nullable=True)
    findings: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime)

    patient: Mapped[PatientRow] = relationship(back_populates="images")


class MonitoringRecordRow(Base):
    """One wearable/CSV observation. All vitals nullable: sources report different subsets."""

    __tablename__ = "monitoring_records"
    __table_args__ = (
        # The Monitoring Agent's core query is a time-ordered window per patient.
        Index("ix_monitoring_patient_time", "patient_id", "recorded_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime)
    spo2: Mapped[float | None] = mapped_column(Float, nullable=True)
    heart_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    respiratory_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    activity_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="synthetic")

    patient: Mapped[PatientRow] = relationship(back_populates="monitoring_records")


class AnalysisRunRow(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (Index("ix_runs_patient_started", "patient_id", "started_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(20), default="pending")

    # RunConfig persisted as JSON so Phase 10 ablations can group runs by the
    # exact switches that were on.
    config: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)

    symptom_input: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)
    final_assessment: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)
    full_result: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    patient: Mapped[PatientRow] = relationship(back_populates="runs")
    agent_results: Mapped[list[AgentResultRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="AgentResultRow.seq"
    )


class AgentResultRow(Base):
    """Per-agent trace for one run — the backing store for the Reasoning panel."""

    __tablename__ = "agent_results"
    __table_args__ = (Index("ix_agent_results_run", "run_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id", ondelete="CASCADE"))
    seq: Mapped[int] = mapped_column(Integer, default=0)
    agent_name: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(20))

    input_payload: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)
    output_payload: Mapped[dict | None] = mapped_column(JSONNoneAsNull, nullable=True)
    fired_rules: Mapped[list | None] = mapped_column(JSONNoneAsNull, nullable=True)
    notes: Mapped[list | None] = mapped_column(JSONNoneAsNull, nullable=True)

    llm_used: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    run: Mapped[AnalysisRunRow] = relationship(back_populates="agent_results")
