"""History Agent.

Turns a heterogeneous record — coded diagnoses, medications, allergies, radiology
reports, social history, wearable samples — into one `PatientProfile` that every
other agent can reason over. It also materialises the timeline the dashboard
renders, because that view is derived from the record and having two generators
of the same rows is how they come to disagree.

Two properties are load-bearing:

**No natural-language inference.** Prior findings come from the `normalised`
column, a coded block carried by the record that produced them. This agent never
parses radiology prose; when Phase 8's Imaging Agent parses an uploaded report,
its output lands in that same column.

**No wall clock.** `reference_date` is resolved from the patient's own latest
observation. Using `datetime.now()` would age the demo patient by a year every
year the project sits unused, and would start firing the stale-record rules for
a reason that has nothing to do with the patient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import (
    AllergyRow,
    MedicalImageRow,
    MedicalRecordRow,
    MedicationRow,
    MonitoringRecordRow,
    PatientRow,
    TimelineEventRow,
)
from app.reasoning.history_rules import HistoryDerivation, HistoryInputs, derive_history
from app.safety import assert_prose_is_safe
from app.schemas import (
    AgentName,
    AgentStatus,
    AgentTrace,
    Allergy,
    ClinicalDimension,
    Condition,
    ImagingFinding,
    ImagingLabel,
    Medication,
    PatientProfile,
    PriorFinding,
    SocialHistory,
    TimelineEvent,
    TimelineEventType,
    VitalSample,
    utcnow,
)

log = logging.getLogger("medai.history")

LIFESTYLE_CATEGORY = "lifestyle"
IMAGING_CATEGORY = "imaging_report"

_EVENT_TYPE_BY_CATEGORY = {
    "diagnosis": TimelineEventType.CONDITION,
    "imaging_report": TimelineEventType.IMAGING,
    "lab": TimelineEventType.LAB,
    "encounter": TimelineEventType.ENCOUNTER,
    "medication": TimelineEventType.MEDICATION,
    "procedure": TimelineEventType.ENCOUNTER,
}


def _human(value: str | StrEnum) -> str:
    """Render a coded value as readable prose: `multifocal_opacities` -> `multifocal opacities`.

    Accepts either a plain string or an enum member so callers do not have to
    reach for `.value`, and so a later change in how labels are typed cannot
    reintroduce underscores into user-facing text.
    """
    return str(getattr(value, "value", value)).replace("_", " ")


class UnknownPatientError(RuntimeError):
    """Raised when a run names a patient that is not in the database."""


# --- Plain-data view of the record --------------------------------------------
#
# The ORM is confined to `load_record`. Everything downstream works on these
# dataclasses, so `build_profile` and `build_timeline` are testable from a
# fixture dict without a database, and Phase 10 can feed them alternative
# patients without touching a session.


@dataclass(frozen=True)
class RecordNote:
    """One `medical_records` row."""

    id: str
    occurred_on: date | None
    category: str
    title: str
    code: str | None = None
    detail: dict[str, Any] | None = None
    is_chronic: bool = False
    is_active: bool = True
    dimension: ClinicalDimension | None = None
    normalised: PriorFinding | None = None


@dataclass(frozen=True)
class ImagingStudy:
    """One `medical_images` row, with its provenance intact."""

    id: str
    study_date: date | None
    source_mode: str
    badge: str | None = None
    provenance: str | None = None
    report_text: str | None = None
    finding: ImagingFinding | None = None


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    full_name: str
    birth_year: int
    sex: str
    reference_date: date
    notes: tuple[RecordNote, ...] = ()
    social_history: tuple[SocialHistory, ...] = ()
    medications: tuple[Medication, ...] = ()
    allergies: tuple[Allergy, ...] = ()
    studies: tuple[ImagingStudy, ...] = ()
    monitoring: tuple[VitalSample, ...] = ()

    @property
    def age(self) -> int:
        """Year difference only.

        The record holds a birth year, not a birth date, so there is no way to
        know whether the birthday has passed. Claiming a precise age would
        invent a fact the data does not contain.
        """
        return self.reference_date.year - self.birth_year

    @property
    def clinical_notes(self) -> tuple[RecordNote, ...]:
        return tuple(n for n in self.notes if n.category != LIFESTYLE_CATEGORY)

    @property
    def notes_as_of(self) -> tuple[RecordNote, ...]:
        """Clinical notes dated on or before the reference date.

        Anything dated later is a future observation relative to that anchor.
        Feeding it into the multiplier or the baseline would be information
        leakage — and it is reachable, because Phase 10 replays a patient at an
        earlier date to ask what the system would have said then.

        In the normal case the reference date *is* the latest dated observation,
        so this filter changes nothing; it only matters for a replay.

        Undated notes are kept: social history has no event date and is always
        applicable context.
        """
        return tuple(
            n for n in self.clinical_notes
            if n.occurred_on is None or n.occurred_on <= self.reference_date
        )

    @property
    def previous_findings(self) -> tuple[PriorFinding, ...]:
        return tuple(
            n.normalised
            for n in sorted(self.notes_as_of, key=lambda n: n.occurred_on or date.min)
            if n.normalised is not None
        )

    @property
    def chronic_conditions(self) -> tuple[Condition, ...]:
        return tuple(
            Condition(
                name=n.title,
                icd10=n.code,
                onset_date=n.occurred_on,
                is_chronic=True,
                is_active=n.is_active,
                affects_dimension=n.dimension,
                notes=(n.detail or {}).get("managed_by"),
            )
            for n in sorted(self.notes_as_of, key=lambda n: n.occurred_on or date.min)
            if n.category == "diagnosis" and n.is_chronic
        )

    @property
    def last_record_date(self) -> date | None:
        dated = [n.occurred_on for n in self.notes_as_of if n.occurred_on]
        return max(dated) if dated else None

    @property
    def prior_studies(self) -> tuple[ImagingStudy, ...]:
        """Studies strictly before the reference date — the comparison anchors.

        Drawn from both tables: historical reports arrive as `medical_records`,
        while anything uploaded as an image lands in `medical_images`. Reading
        only one would silently lose the comparison baseline in the other.
        """
        return tuple(
            s for s in self.studies
            if s.study_date is not None and s.study_date < self.reference_date
        )

    @property
    def current_studies(self) -> tuple[ImagingStudy, ...]:
        return tuple(
            s for s in self.studies
            if s.study_date is not None and s.study_date >= self.reference_date
        )

    @property
    def last_prior_imaging_date(self) -> date | None:
        from_reports = [
            n.occurred_on
            for n in self.clinical_notes
            if n.category == IMAGING_CATEGORY
            and n.occurred_on is not None
            and n.occurred_on < self.reference_date
        ]
        from_studies = [s.study_date for s in self.prior_studies if s.study_date]
        dated = from_reports + from_studies
        return max(dated) if dated else None

    def history_inputs(self) -> HistoryInputs:
        return HistoryInputs(
            reference_date=self.reference_date,
            age=self.age,
            chronic_conditions=self.chronic_conditions,
            previous_findings=self.previous_findings,
            social_history=self.social_history,
            active_medications=self.medications,
            allergies=self.allergies,
            last_record_date=self.last_record_date,
            last_prior_imaging_date=self.last_prior_imaging_date,
            monitoring_sample_count=len(self.monitoring),
        )


# --- Loading ------------------------------------------------------------------


def resolve_reference_date(session: Session, patient_id: str, explicit: date | None = None) -> date:
    """The date 'today' means for this patient.

    Falls back to the wall clock only for a patient with no dated data at all,
    where there is nothing better available.
    """
    if explicit is not None:
        return explicit

    candidates: list[date] = []
    latest_sample = session.scalar(
        select(func.max(MonitoringRecordRow.recorded_at)).where(
            MonitoringRecordRow.patient_id == patient_id
        )
    )
    if latest_sample is not None:
        candidates.append(latest_sample.date())
    candidates.extend(
        d for d in session.scalars(
            select(MedicalImageRow.study_date).where(MedicalImageRow.patient_id == patient_id)
        ).all() if d is not None
    )
    candidates.extend(
        d for d in session.scalars(
            select(MedicalRecordRow.occurred_on).where(MedicalRecordRow.patient_id == patient_id)
        ).all() if d is not None
    )
    if candidates:
        return max(candidates)

    log.warning(
        "Patient %s has no dated observations; falling back to the wall clock. "
        "Longitudinal output for this patient is not reproducible.", patient_id,
    )
    return utcnow().date()


def _to_record_note(row: MedicalRecordRow) -> RecordNote:
    normalised: PriorFinding | None = None
    if row.normalised is not None:
        dimension = ClinicalDimension(row.clinical_dimension) if row.clinical_dimension else None
        if dimension is None:
            raise ValueError(
                f"Record {row.id} carries a normalised finding but no clinical_dimension; "
                "without a dimension it cannot be compared against the present."
            )
        normalised = PriorFinding.model_validate(
            {
                "dimension": dimension,
                "occurred_on": row.occurred_on,
                "source_record_id": row.id,
                **row.normalised,
            }
        )
        if dimension == ClinicalDimension.PULMONARY_IMAGING:
            # Enforced here rather than assumed: Phase 6 matches a prior study
            # against a current abnormality by label, and a label outside the
            # closed vocabulary can never match, silently reporting a known
            # finding as new.
            try:
                ImagingLabel(normalised.label)
            except ValueError as exc:
                raise ValueError(
                    f"Record {row.id} has pulmonary_imaging label {normalised.label!r}, "
                    f"which is not an ImagingLabel value: {exc}"
                ) from exc

    return RecordNote(
        id=row.id,
        occurred_on=row.occurred_on,
        category=row.category,
        title=row.title,
        code=row.code,
        detail=row.detail,
        is_chronic=row.is_chronic,
        is_active=row.is_active,
        dimension=ClinicalDimension(row.clinical_dimension) if row.clinical_dimension else None,
        normalised=normalised,
    )


def load_record(
    session: Session, patient_id: str, reference_date: date | None = None
) -> PatientRecord:
    """Read a patient's whole record into plain data."""
    patient = session.get(PatientRow, patient_id)
    if patient is None:
        raise UnknownPatientError(f"No patient with id {patient_id!r}")

    resolved = resolve_reference_date(session, patient_id, reference_date)

    notes = tuple(
        _to_record_note(row)
        for row in session.scalars(
            select(MedicalRecordRow)
            .where(MedicalRecordRow.patient_id == patient_id)
            .order_by(MedicalRecordRow.occurred_on, MedicalRecordRow.id)
        ).all()
    )

    social = tuple(
        SocialHistory.model_validate(row.detail)
        for row in session.scalars(
            select(MedicalRecordRow).where(
                MedicalRecordRow.patient_id == patient_id,
                MedicalRecordRow.category == LIFESTYLE_CATEGORY,
            )
        ).all()
    )

    medications = tuple(
        Medication(
            name=row.name,
            dose=row.dose,
            frequency=row.frequency,
            start_date=row.start_date,
            end_date=row.end_date,
            is_active=row.is_active,
        )
        for row in session.scalars(
            select(MedicationRow)
            .where(MedicationRow.patient_id == patient_id, MedicationRow.is_active.is_(True))
            .order_by(MedicationRow.start_date)
        ).all()
    )

    allergies = tuple(
        Allergy(allergen=row.allergen, reaction=row.reaction, severity=row.severity)
        for row in session.scalars(
            select(AllergyRow).where(AllergyRow.patient_id == patient_id)
        ).all()
    )

    studies = tuple(
        ImagingStudy(
            id=row.id,
            study_date=row.study_date,
            source_mode=row.source_mode,
            badge=row.badge,
            provenance=row.provenance,
            report_text=row.report_text,
            finding=ImagingFinding.model_validate(row.findings) if row.findings else None,
        )
        for row in session.scalars(
            select(MedicalImageRow)
            .where(MedicalImageRow.patient_id == patient_id)
            .order_by(MedicalImageRow.study_date)
        ).all()
    )

    monitoring = tuple(
        VitalSample(
            recorded_at=row.recorded_at,
            spo2=row.spo2,
            heart_rate=row.heart_rate,
            temperature_c=row.temperature_c,
            respiratory_rate=row.respiratory_rate,
            sleep_hours=row.sleep_hours,
            activity_steps=row.activity_steps,
            source=row.source,
        )
        for row in session.scalars(
            select(MonitoringRecordRow)
            .where(MonitoringRecordRow.patient_id == patient_id)
            .order_by(MonitoringRecordRow.recorded_at)
        ).all()
    )

    return PatientRecord(
        patient_id=patient.id,
        full_name=patient.full_name,
        birth_year=patient.birth_year,
        sex=patient.sex,
        reference_date=resolved,
        notes=notes,
        social_history=social,
        medications=medications,
        allergies=allergies,
        studies=studies,
        monitoring=monitoring,
    )


# --- Profile ------------------------------------------------------------------


def _assemble(record: PatientRecord) -> tuple[PatientProfile, HistoryDerivation]:
    """Build the profile and return the derivation that produced it.

    Returned together so the trace can quote the same rule evaluation the
    profile was built from. Deriving twice would let the two disagree.
    """
    derivation = derive_history(record.history_inputs())

    profile = PatientProfile(
        patient_id=record.patient_id,
        full_name=record.full_name,
        age=record.age,
        sex=record.sex,
        chronic_conditions=list(record.chronic_conditions),
        active_medications=list(record.medications),
        allergies=list(record.allergies),
        social_history=list(record.social_history),
        previous_findings=list(record.previous_findings),
        risk_factors=list(derivation.risk_factors),
        baseline_vulnerabilities=list(derivation.baseline_vulnerabilities),
        history_multiplier=derivation.history_multiplier,
        timeline=build_timeline(record),
        data_gaps=list(derivation.data_gaps),
        summary=build_summary(record, derivation.history_multiplier, derivation.baseline_vulnerabilities),
    )

    if get_settings().enforce_safety_guard:
        assert_prose_is_safe({"history_summary": profile.summary})
    return profile, derivation


def build_profile(record: PatientRecord) -> PatientProfile:
    return _assemble(record)[0]


def build_summary(
    record: PatientRecord,
    history_multiplier: float,
    vulnerabilities: tuple[ClinicalDimension, ...] | list[ClinicalDimension],
) -> str:
    """Template-generated prose. Deterministic, and audited by the safety guard.

    Written in the standard clinical "N-year-old with X" register, which states
    what is on the record without asserting a current diagnosis.
    """
    head = f"{record.age}-year-old {record.sex}"
    conditions = [c.name for c in record.chronic_conditions]
    if conditions:
        head += " with " + ", ".join(conditions)

    clauses: list[str] = []

    findings = sorted(record.previous_findings, key=lambda f: f.occurred_on)
    if findings:
        # ", " inside the list, because "; " separates the outer clauses.
        clauses.append("prior findings: " + ", ".join(
            f"{_human(f.label)} ({f.occurred_on.isoformat()}"
            f"{', resolved' if f.resolved else ''}, severity {f.severity_score}/5)"
            for f in findings
        ))

    if record.medications:
        meds = ", ".join(
            f"{m.name}{f' {m.dose}' if m.dose else ''}" for m in record.medications
        )
        clauses.append(f"current medication: {meds}")
    if record.allergies:
        clauses.append(f"allergies: {', '.join(a.allergen for a in record.allergies)}")

    span = ""
    if record.last_record_date and record.clinical_notes:
        earliest = min(
            (n.occurred_on for n in record.clinical_notes if n.occurred_on), default=None
        )
        if earliest:
            span = f" Record spans {earliest.isoformat()} to {record.last_record_date.isoformat()}."

    monitoring = (
        f" {len(record.monitoring)} monitoring samples available."
        if record.monitoring
        else " No monitoring data on file."
    )

    tail = (
        f" History adjusts the risk multiplier to x{history_multiplier:g}"
        + (
            f" across {len(vulnerabilities)} vulnerable dimensions "
            f"({', '.join(_human(v) for v in vulnerabilities)})."
            if vulnerabilities
            else "."
        )
    )

    text = head + (f"; {'; '.join(clauses)}" if clauses else "")
    return text + "." + span + monitoring + tail


# --- Timeline -----------------------------------------------------------------


def _monitoring_event(record: PatientRecord) -> TimelineEvent | None:
    """One event summarising the observation window, not one per sample.

    Five daily rows would bury the clinical history; the per-sample detail
    belongs to the Monitoring panel. The raw series rides along in `payload` so
    the event is self-contained.
    """
    if not record.monitoring:
        return None

    samples = sorted(record.monitoring, key=lambda s: s.recorded_at)
    first, last = samples[0].recorded_at, samples[-1].recorded_at
    window_days = (last.date() - first.date()).days + 1

    return TimelineEvent(
        occurred_on=last.date(),
        event_type=TimelineEventType.CURRENT,
        label=f"Continuous monitoring window — {len(samples)} samples over {window_days} days",
        detail=f"{first.date().isoformat()} to {last.date().isoformat()}",
        dimension=ClinicalDimension.OXYGENATION,
        payload={
            "sample_count": len(samples),
            "window_days": window_days,
            "first_recorded_at": first.isoformat(),
            "last_recorded_at": last.isoformat(),
            "series": [s.model_dump(mode="json") for s in samples],
        },
        is_current=True,
    )


def _study_events(record: PatientRecord) -> list[TimelineEvent]:
    events = []
    for study in record.studies:
        if study.study_date is None:
            continue
        finding = study.finding
        is_current = study.study_date >= record.reference_date
        labels = (
            ", ".join(_human(a.label) for a in finding.abnormalities) if finding else None
        )
        events.append(TimelineEvent(
            occurred_on=study.study_date,
            event_type=TimelineEventType.IMAGING,
            label=(
                f"Chest X-ray — {labels}" if labels
                else "Chest X-ray" + (" (current study)" if is_current else "")
            ),
            detail=(finding.summary if finding else None) or study.provenance,
            dimension=ClinicalDimension.PULMONARY_IMAGING,
            payload={
                "study_id": study.id,
                "source_mode": study.source_mode,
                # Provenance travels with the event: a mock finding must stay
                # labelled mock wherever it is rendered, including the timeline.
                "badge": study.badge,
                "abnormalities": (
                    [a.model_dump(mode="json") for a in finding.abnormalities] if finding else []
                ),
            },
            is_current=is_current,
        ))
    return events


def _medication_events(record: PatientRecord) -> list[TimelineEvent]:
    return [
        TimelineEvent(
            occurred_on=m.start_date,
            event_type=TimelineEventType.MEDICATION,
            label=f"Started {m.name}" + (f" {m.dose}" if m.dose else ""),
            detail=m.frequency,
            payload={"active": m.is_active},
        )
        for m in record.medications
        if m.start_date is not None
    ]


def build_timeline(record: PatientRecord) -> list[TimelineEvent]:
    """Materialise the longitudinal timeline, oldest first, present last."""
    events: list[TimelineEvent] = []

    for note in record.clinical_notes:
        if note.occurred_on is None:
            continue
        detail = note.detail or {}
        events.append(TimelineEvent(
            occurred_on=note.occurred_on,
            event_type=_EVENT_TYPE_BY_CATEGORY.get(note.category, TimelineEventType.ENCOUNTER),
            label=note.title,
            # Authored prose only. Measurements are not flattened into a string
            # here: that restates values already carried structurally in
            # `source_detail` below, and the flattened form loses the units.
            detail=detail.get("report_text") or detail.get("indication"),
            dimension=note.dimension,
            payload={
                "record_id": note.id,
                "category": note.category,
                "code": note.code,
                "chronic": note.is_chronic,
                "active": note.is_active,
                # The structured source dict, so the UI can render lab values
                # and encounter measurements as data rather than parsing text.
                "source_detail": note.detail,
                **({"normalised": note.normalised.model_dump(mode="json")} if note.normalised else {}),
            },
        ))

    events.extend(_medication_events(record))
    events.extend(_study_events(record))

    monitoring_event = _monitoring_event(record)
    if monitoring_event:
        events.append(monitoring_event)

    # Sort on (is_current, date) so the historical record reads chronologically
    # and the "Today" cluster sits at the end regardless of its timestamp.
    events.sort(key=lambda e: (e.is_current, e.occurred_on))
    return [e.model_copy(update={"sort_order": i}) for i, e in enumerate(events)]


def persist_timeline(session: Session, patient_id: str, events: list[TimelineEvent]) -> int:
    """Replace this patient's timeline rows. Returns the number written.

    Delete-then-insert rather than diff-and-patch: the timeline is a derived
    view rebuilt on every run, and ids are deterministic, so a re-run converges
    on identical rows instead of accumulating duplicates.
    """
    for row in session.scalars(
        select(TimelineEventRow).where(TimelineEventRow.patient_id == patient_id)
    ).all():
        session.delete(row)
    session.flush()

    for event in events:
        session.add(TimelineEventRow(
            id=f"TL-{patient_id}-{event.sort_order:03d}",
            patient_id=patient_id,
            occurred_on=event.occurred_on,
            event_type=event.event_type.value,
            label=event.label,
            detail=event.detail,
            clinical_dimension=event.dimension.value if event.dimension else None,
            payload=event.payload,
            sort_order=event.sort_order,
            is_current=event.is_current,
        ))
    session.flush()
    return len(events)


# --- Agent entry point --------------------------------------------------------


def run_history(
    session: Session,
    patient_id: str,
    reference_date: date | None = None,
    persist: bool = True,
) -> tuple[PatientProfile | None, AgentTrace]:
    """Run the History Agent. Returns the profile and its trace.

    Flushes but never commits: the caller owns the transaction. Committing here
    would silently commit whatever else the surrounding pipeline had staged.

    Errors are captured in the trace rather than raised, because the workflow
    fans out to six agents and one failing must not take the others down. The
    caller reads `trace.status` and decides.
    """
    started = utcnow()
    try:
        record = load_record(session, patient_id, reference_date)
        profile, derivation = _assemble(record)
        written = persist_timeline(session, patient_id, profile.timeline) if persist else 0

        finished = utcnow()
        trace = AgentTrace(
            agent_name=AgentName.HISTORY,
            status=AgentStatus.OK,
            output=profile.model_dump(mode="json"),
            fired_rules=list(derivation.fired_rules),
            notes=[
                f"reference_date={record.reference_date.isoformat()}",
                # A rationale rather than a key=value pair, so the parseable
                # notes around it stay parseable.
                "reference_date is derived from the patient's latest observation, not the wall clock",
                f"age={profile.age}",
                f"birth_year={record.birth_year}",
                f"prior_findings={len(profile.previous_findings)}",
                f"chronic_conditions={len(profile.chronic_conditions)}",
                f"timeline_events={written}",
                f"history_multiplier=x{profile.history_multiplier:g}",
                *(f"data_gap: {gap}" for gap in profile.data_gaps),
            ],
            llm_used=False,
            duration_ms=int((finished - started).total_seconds() * 1000),
            started_at=started,
            finished_at=finished,
        )
        return profile, trace

    except Exception as exc:  # noqa: BLE001 — the trace is the error channel
        session.rollback()
        finished = utcnow()
        log.exception("History Agent failed for patient %s", patient_id)
        return None, AgentTrace(
            agent_name=AgentName.HISTORY,
            status=AgentStatus.ERROR,
            notes=[f"{type(exc).__name__}: {exc}"],
            llm_used=False,
            duration_ms=int((finished - started).total_seconds() * 1000),
            started_at=started,
            finished_at=finished,
            error=f"{type(exc).__name__}: {exc}",
        )
