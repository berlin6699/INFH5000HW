"""Monitoring Agent.

Loads a patient's continuous observations, runs the deterministic rule engine in
`reasoning/monitoring_rules.py`, and returns a `MonitoringAssessment` plus its
audit trail. The engine owns every judgement — current value against the
published instrument, temporal trend, deviation from the personal baseline —
and this module owns I/O, the trace, and two properties that are load-bearing
for the whole system:

**No wall clock.** The window is anchored to `reference_date`, resolved by the
same helper the History Agent uses, so the two agents cannot disagree about when
"now" is and a Phase 10 replay at an earlier anchor is reproducible.

**No future samples.** Observations dated after the anchor are excluded, for the
same reason the History Agent filters `notes_as_of`: feeding a later observation
into the baseline is information leakage, and it becomes reachable the moment a
replay moves the anchor backwards.

An empty window is not an error. The rules turn it into an explicit
`insufficient` assessment with data gaps, which states the absence; reporting
normal vitals for a patient nobody measured would be the dangerous version.

`llm_used` is False and the summary is template prose audited by the safety
guard before it leaves this module, so no structured field and no rendered
sentence here depends on a language model.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.history import UnknownPatientError, resolve_reference_date
from app.config import get_settings
from app.db.models import MonitoringRecordRow, PatientRow
from app.reasoning.monitoring_rules import assess_monitoring
from app.safety import assert_prose_is_safe
from app.schemas import (
    AgentName,
    AgentStatus,
    AgentTrace,
    MonitoringAssessment,
    VitalSample,
    utcnow,
)

log = logging.getLogger("medai.monitoring")


def _to_vital_sample(row: MonitoringRecordRow) -> VitalSample:
    return VitalSample(
        recorded_at=row.recorded_at,
        spo2=row.spo2,
        heart_rate=row.heart_rate,
        temperature_c=row.temperature_c,
        respiratory_rate=row.respiratory_rate,
        sleep_hours=row.sleep_hours,
        activity_steps=row.activity_steps,
        source=row.source,
    )


def load_samples(
    session: Session,
    patient_id: str,
    reference_date: date | None = None,
    max_window_days: int | None = None,
) -> tuple[list[VitalSample], date]:
    """Read the patient's observation window as plain `VitalSample`s.

    Returns the samples together with the resolved anchor date, because the
    trace reports the anchor that was actually used and re-resolving it could
    pick a different one.

    `max_window_days` caps how far back the window reaches, counting the anchor
    day as the last one: 5 keeps the anchor day and the four before it. None
    keeps everything on or before the anchor. The cap filters; it does not
    pad — the assessment's `window_days` still reports what the data spans, so
    a three-day series inside a five-day cap says three, not five.
    """
    patient = session.get(PatientRow, patient_id)
    if patient is None:
        raise UnknownPatientError(f"No patient with id {patient_id!r}")

    anchor = resolve_reference_date(session, patient_id, reference_date)

    floor: datetime | None = None
    if max_window_days is not None:
        if max_window_days < 1:
            raise ValueError(f"max_window_days must be >= 1, got {max_window_days}")
        first_day = anchor - timedelta(days=max_window_days - 1)
        floor = datetime.combine(first_day, time.min, tzinfo=timezone.utc)

    rows = session.scalars(
        select(MonitoringRecordRow)
        .where(MonitoringRecordRow.patient_id == patient_id)
        .order_by(MonitoringRecordRow.recorded_at, MonitoringRecordRow.id)
    ).all()

    samples = []
    for row in rows:
        if row.recorded_at.date() > anchor:
            continue
        if floor is not None and row.recorded_at < floor:
            continue
        samples.append(_to_vital_sample(row))
    return samples, anchor


def run_monitoring(
    session: Session,
    patient_id: str,
    reference_date: date | None = None,
    max_window_days: int | None = None,
) -> tuple[MonitoringAssessment | None, AgentTrace]:
    """Run the Monitoring Agent. Returns the assessment and its trace.

    Same transaction discipline as the History Agent: nothing is written, so
    there is nothing to flush or commit — the assessment rides out in the trace
    and the workflow persists it with the rest of the run's agent results.

    Errors are captured in the trace rather than raised, because the workflow
    fans out to six agents and one failing must not take the others down.
    """
    started = utcnow()
    try:
        samples, anchor = load_samples(
            session, patient_id, reference_date, max_window_days
        )
        derivation = assess_monitoring(samples)
        assessment = derivation.assessment

        if get_settings().enforce_safety_guard:
            assert_prose_is_safe({
                "monitoring_summary": assessment.summary,
                "current_abnormalities": assessment.current_abnormalities,
                "data_gaps": assessment.data_gaps,
            })

        finished = utcnow()
        trace = AgentTrace(
            agent_name=AgentName.MONITORING,
            status=AgentStatus.OK,
            output=assessment.model_dump(mode="json"),
            fired_rules=list(derivation.fired_rules),
            notes=[
                f"reference_date={anchor.isoformat()}",
                # A rationale rather than a key=value pair, so the parseable
                # notes around it stay parseable.
                "the window is anchored to the patient's latest observation, not the wall clock",
                f"max_window_days={max_window_days if max_window_days is not None else 'all'}",
                f"sample_count={assessment.sample_count}",
                f"window_days={assessment.window_days}",
                f"metrics_reported={len(assessment.trends)}",
                f"data_quality={assessment.data_quality}",
                f"news2_partial_total={assessment.news2_partial_total if assessment.news2_partial_total is not None else 'none'}",
                f"news2_any_parameter_trigger={str(assessment.news2_any_parameter_trigger).lower()}",
                f"rapid_deterioration={str(assessment.rapid_deterioration).lower()}",
                f"deteriorating_metrics={','.join(m.value for m in assessment.deteriorating_metrics) or 'none'}",
                f"worst_metric={assessment.worst_metric.value if assessment.worst_metric else 'none'}",
                *(f"data_gap: {gap}" for gap in assessment.data_gaps),
            ],
            llm_used=False,
            duration_ms=int((finished - started).total_seconds() * 1000),
            started_at=started,
            finished_at=finished,
        )
        return assessment, trace

    except Exception as exc:  # noqa: BLE001 — the trace is the error channel
        session.rollback()
        finished = utcnow()
        log.exception("Monitoring Agent failed for patient %s", patient_id)
        return None, AgentTrace(
            agent_name=AgentName.MONITORING,
            status=AgentStatus.ERROR,
            notes=[f"{type(exc).__name__}: {exc}"],
            llm_used=False,
            duration_ms=int((finished - started).total_seconds() * 1000),
            started_at=started,
            finished_at=finished,
            error=f"{type(exc).__name__}: {exc}",
        )
