"""Load the synthetic demo patient into a fresh database.

Run with `make seed`. Destructive by design: it drops and recreates every table
so a demo always starts from a known state.

Seed data is validated through the same pydantic contracts the agents use, so a
malformed JSON fixture fails here rather than surfacing later as an agent error.

`timeline_events` is deliberately left empty. The timeline is a materialised
view owned by the History Agent (Phase 2), not source data — populating it here
would mean two places generating the same rows.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import BACKEND_ROOT
from app.db.models import (
    AllergyRow,
    MedicalImageRow,
    MedicalRecordRow,
    MedicationRow,
    MonitoringRecordRow,
    PatientRow,
    TimelineEventRow,
)
from app.db.session import drop_all, get_session_factory, init_db
from app.schemas import Abnormality, ImagingFinding, PriorFinding, SocialHistory, VitalSample

DATA_DIR = BACKEND_ROOT / "data"
PATIENT_FIXTURE = DATA_DIR / "synthetic" / "patient_demo.json"
MONITORING_CSV = DATA_DIR / "synthetic" / "monitoring_5d.csv"


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _parse_dt(value: str) -> datetime:
    # Fixtures use a trailing Z; fromisoformat only accepts +00:00 before 3.11
    # semantics differ, so normalise explicitly.
    text = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_fixture() -> dict:
    if not PATIENT_FIXTURE.exists():
        raise FileNotFoundError(f"Missing demo fixture: {PATIENT_FIXTURE}")
    return json.loads(PATIENT_FIXTURE.read_text(encoding="utf-8"))


def load_monitoring_csv() -> list[VitalSample]:
    if not MONITORING_CSV.exists():
        raise FileNotFoundError(f"Missing monitoring fixture: {MONITORING_CSV}")

    samples: list[VitalSample] = []
    with MONITORING_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            def num(key: str) -> float | None:
                raw = (row.get(key) or "").strip()
                return float(raw) if raw else None

            samples.append(
                VitalSample(
                    recorded_at=_parse_dt(row["recorded_at"]),
                    spo2=num("spo2"),
                    heart_rate=num("heart_rate"),
                    temperature_c=num("temperature_c"),
                    respiratory_rate=num("respiratory_rate"),
                    sleep_hours=num("sleep_hours"),
                    activity_steps=(
                        int(row["activity_steps"]) if (row.get("activity_steps") or "").strip() else None
                    ),
                    source=(row.get("source") or "csv").strip(),
                )
            )
    return samples


def seed(session: Session, fixture: dict | None = None) -> str:
    """Insert the demo patient. Returns the patient id."""
    fx = fixture or load_fixture()

    p = fx["patient"]
    patient = PatientRow(
        id=p["id"],
        full_name=p["full_name"],
        birth_year=int(p["birth_year"]),
        sex=p.get("sex", "unknown"),
        is_synthetic=bool(p.get("is_synthetic", True)),
        notes=p.get("notes"),
        created_at=datetime.now(timezone.utc),
    )
    session.add(patient)

    for r in fx.get("medical_records", []):
        normalised = r.get("normalised")
        if normalised is not None:
            # Validate the coded finding against the contract the History Agent
            # consumes, so a malformed fixture fails at seed time rather than
            # surfacing later as an agent error.
            PriorFinding.model_validate(
                {
                    "dimension": r["clinical_dimension"],
                    "occurred_on": r["occurred_on"],
                    "source_record_id": r["id"],
                    **normalised,
                }
            )
        session.add(
            MedicalRecordRow(
                id=r["id"],
                patient_id=patient.id,
                occurred_on=_parse_date(r["occurred_on"]),
                category=r["category"],
                code=r.get("code"),
                title=r["title"],
                detail=r.get("detail"),
                is_chronic=bool(r.get("is_chronic", False)),
                is_active=bool(r.get("is_active", True)),
                clinical_dimension=r.get("clinical_dimension"),
                normalised=normalised,
            )
        )

    # Social history has no event date, so occurred_on stays null rather than
    # being fabricated. The structured payload (pack_years, bmi, ...) goes into
    # `detail` so the risk-factor rules read numbers instead of parsing prose.
    for lf in fx.get("lifestyle", []):
        social = SocialHistory.model_validate(lf)
        session.add(
            MedicalRecordRow(
                id=f"{patient.id}-LS-{social.key}",
                patient_id=patient.id,
                occurred_on=None,
                category="lifestyle",
                title=social.factor,
                detail=social.model_dump(mode="json"),
                is_chronic=True,
                is_active=True,
                clinical_dimension=None,
                normalised=None,
            )
        )

    for m in fx.get("medications", []):
        session.add(
            MedicationRow(
                id=m["id"],
                patient_id=patient.id,
                name=m["name"],
                dose=m.get("dose"),
                frequency=m.get("frequency"),
                start_date=_parse_date(m.get("start_date")),
                end_date=_parse_date(m.get("end_date")),
                is_active=bool(m.get("is_active", True)),
            )
        )

    for a in fx.get("allergies", []):
        session.add(
            AllergyRow(
                id=a["id"],
                patient_id=patient.id,
                allergen=a["allergen"],
                reaction=a.get("reaction"),
                severity=a.get("severity", "unknown"),
            )
        )

    for s in load_monitoring_csv():
        session.add(
            MonitoringRecordRow(
                id=_uid("MON"),
                patient_id=patient.id,
                recorded_at=s.recorded_at,
                spo2=s.spo2,
                heart_rate=s.heart_rate,
                temperature_c=s.temperature_c,
                respiratory_rate=s.respiratory_rate,
                sleep_hours=s.sleep_hours,
                activity_steps=s.activity_steps,
                source=s.source,
            )
        )

    ci = fx.get("current_imaging")
    if ci:
        # Round-tripping through ImagingFinding runs the provenance validator,
        # which auto-fills the disclosure badge. Persisting that badge means a
        # row saved as mock stays labelled mock even if the server's configured
        # imaging mode changes later.
        finding = ImagingFinding.model_validate(
            {
                "image_id": ci["id"],
                "modality": ci.get("modality", "chest_xray"),
                "study_date": ci.get("study_date"),
                "source_mode": ci["source_mode"],
                "provenance": ci["provenance"],
                "findings": ci.get("findings", []),
                "abnormalities": [Abnormality.model_validate(a) for a in ci.get("abnormalities", [])],
                "has_acute_abnormality": any(
                    a.get("acuity") == "acute" for a in ci.get("abnormalities", [])
                ),
                "summary": ci.get("summary", ""),
            }
        )
        session.add(
            MedicalImageRow(
                id=ci["id"],
                patient_id=patient.id,
                study_date=finding.study_date,
                modality=finding.modality.value,
                file_path=None,
                report_text=ci.get("report_text"),
                source_mode=finding.source_mode.value,
                provenance=finding.provenance,
                badge=finding.badge,
                findings=finding.model_dump(mode="json"),
                created_at=datetime.now(timezone.utc),
            )
        )

    session.flush()
    return patient.id


def reset_and_seed(verbose: bool = True) -> str:
    drop_all()
    init_db()
    session = get_session_factory()()
    try:
        patient_id = seed(session)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    if verbose:
        _report(patient_id)
    return patient_id


def _report(patient_id: str) -> None:
    session = get_session_factory()()
    try:
        def count(model) -> int:
            return len(session.scalars(select(model).where(model.patient_id == patient_id)).all())

        records = session.scalars(
            select(MedicalRecordRow).where(MedicalRecordRow.patient_id == patient_id)
        ).all()
        clinical = [r for r in records if r.category != "lifestyle"]

        spo2 = session.scalars(
            select(MonitoringRecordRow.spo2)
            .where(MonitoringRecordRow.patient_id == patient_id)
            .order_by(MonitoringRecordRow.recorded_at)
        ).all()
        hr = session.scalars(
            select(MonitoringRecordRow.heart_rate)
            .where(MonitoringRecordRow.patient_id == patient_id)
            .order_by(MonitoringRecordRow.recorded_at)
        ).all()

        print(f"Seeded patient {patient_id}")
        print(f"  medical_records    {len(clinical)} clinical + {len(records) - len(clinical)} social history")
        print(f"  coded findings     {sum(1 for r in records if r.normalised)}  (normalised, read by History Agent)")
        print(f"  medications        {count(MedicationRow)}")
        print(f"  allergies          {count(AllergyRow)}")
        print(f"  monitoring_records {count(MonitoringRecordRow)}")
        print(f"  medical_images     {count(MedicalImageRow)}")
        print(f"  timeline_events    {count(TimelineEventRow)}  (materialised by History Agent, Phase 2)")
        print(f"  SpO2 series        {[int(v) if v is not None else None for v in spo2]}")
        print(f"  HR series          {[int(v) if v is not None else None for v in hr]}")
    finally:
        session.close()


if __name__ == "__main__":
    reset_and_seed()
