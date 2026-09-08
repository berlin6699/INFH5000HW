"""Database layer tests: schema creation, seed integrity and SQLite behaviour.

The seed fixture is the demo's script, so its contents are asserted exactly
rather than approximately — a silently altered SpO₂ series would still pass a
"row count" test while undermining the whole scenario.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    AgentResultRow,
    AllergyRow,
    AnalysisRunRow,
    MedicalImageRow,
    MedicalRecordRow,
    MedicationRow,
    MonitoringRecordRow,
    PatientRow,
    TimelineEventRow,
)
from app.db.seed import load_fixture, reset_and_seed, seed
from app.db.session import get_engine, get_session_factory, init_db

EXPECTED_TABLES = {
    "patients",
    "medical_records",
    "medications",
    "allergies",
    "timeline_events",
    "medical_images",
    "monitoring_records",
    "analysis_runs",
    "agent_results",
}

# The MVP scenario from the project specification.
EXPECTED_SPO2 = [98.0, 97.0, 95.0, 93.0, 91.0]
EXPECTED_HR = [72.0, 75.0, 82.0, 94.0, 103.0]
EXPECTED_TEMP = [36.8, 37.4, 38.1, 38.5, 38.7]
EXPECTED_RR = [14.0, 16.0, 18.0, 21.0, 24.0]


def _series(session, column) -> list[float]:
    return list(
        session.scalars(
            select(column)
            .where(MonitoringRecordRow.patient_id == "PT-DEMO-001")
            .order_by(MonitoringRecordRow.recorded_at)
        ).all()
    )


# --- Schema -------------------------------------------------------------------


def test_all_tables_are_created(tmp_db):
    created = set(inspect(get_engine()).get_table_names())
    assert EXPECTED_TABLES <= created, f"missing tables: {sorted(EXPECTED_TABLES - created)}"


def test_indexes_are_created(tmp_db):
    names = {
        row[0]
        for row in get_engine().connect().execute(
            text("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
        )
    }
    # The Monitoring Agent's core query is a time-ordered window per patient.
    assert "ix_monitoring_patient_time" in names
    assert "ix_agent_results_run" in names
    assert "ix_timeline_patient_order" in names


def test_sqlite_foreign_keys_are_enforced(tmp_db):
    """SQLite leaves FOREIGN KEY constraints off unless enabled per connection.

    Without the pragma listener the relationships would be decorative and
    orphaned rows could accumulate across repeated seeds.
    """
    with get_engine().connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


# --- Seed integrity -----------------------------------------------------------


def test_seed_row_counts(session):
    def count(model) -> int:
        return len(
            session.scalars(
                select(model).where(model.patient_id == "PT-DEMO-001")
            ).all()
        )

    records = session.scalars(
        select(MedicalRecordRow).where(MedicalRecordRow.patient_id == "PT-DEMO-001")
    ).all()
    clinical = [r for r in records if r.category != "lifestyle"]
    social = [r for r in records if r.category == "lifestyle"]

    # Asserted as two groups: a single total would let a dropped social-history
    # row hide behind an added clinical one.
    assert len(clinical) == 5
    assert len(social) == 2
    assert sum(1 for r in records if r.normalised) == 2
    assert count(MedicationRow) == 1
    assert count(AllergyRow) == 1
    assert count(MonitoringRecordRow) == 5
    assert count(MedicalImageRow) == 1


def test_social_history_is_persisted_without_a_fabricated_date(session):
    """Smoking status and BMI are context, not events.

    Inventing a date for them would put a false point on the timeline, so they
    are stored with occurred_on null and excluded from it.
    """
    social = session.scalars(
        select(MedicalRecordRow).where(MedicalRecordRow.category == "lifestyle")
    ).all()
    assert {r.title for r in social} == {"Former smoker", "BMI"}
    assert all(r.occurred_on is None for r in social)
    # The structured payload is what the risk-factor rules read; parsing
    # "~8 pack-years, quit 2018" as prose is exactly what this avoids.
    smoker = next(r for r in social if r.title == "Former smoker")
    assert smoker.detail["pack_years"] == 8
    assert smoker.detail["current_smoker"] is False
    bmi = next(r for r in social if r.title == "BMI")
    assert bmi.detail["bmi"] == 27.4


def test_normalised_blocks_round_trip_into_prior_findings(session):
    """The coded findings the History Agent reads must satisfy its contract."""
    from app.schemas import Acuity, ClinicalDimension, ImagingLabel, PriorFinding

    rows = session.scalars(
        select(MedicalRecordRow).where(MedicalRecordRow.normalised.is_not(None))
    ).all()
    assert len(rows) == 2

    findings = [
        PriorFinding.model_validate(
            {
                "dimension": r.clinical_dimension,
                "occurred_on": r.occurred_on,
                "source_record_id": r.id,
                **r.normalised,
            }
        )
        for r in rows
    ]
    by_label = {f.label: f for f in findings}
    assert set(by_label) == {"focal_opacity", "fibrosis"}

    # A chronic diagnosis is carried as a Condition on its dimension, not also as
    # a PriorFinding: a severity score only means something on an axis the
    # system can re-measure and diff, and "hypertension 1/5" is not one.
    assert all(f.dimension == ClinicalDimension.PULMONARY_IMAGING for f in findings)

    # Imaging-dimension labels must come from the closed vocabulary, or Phase 6
    # cannot match a prior study against a current abnormality by label.
    imaging_labels = {m.value for m in ImagingLabel}
    for f in findings:
        if f.dimension == ClinicalDimension.PULMONARY_IMAGING:
            assert f.label in imaging_labels, f.label

    # The scenario's premise: an opacity that resolved, leaving fibrosis.
    assert by_label["focal_opacity"].resolved is True
    assert by_label["focal_opacity"].severity_score == 2
    assert by_label["focal_opacity"].acuity == Acuity.ACUTE
    assert by_label["fibrosis"].resolved is False
    assert by_label["fibrosis"].acuity == Acuity.CHRONIC


def test_seed_vital_series_match_the_scenario(session):
    assert _series(session, MonitoringRecordRow.spo2) == EXPECTED_SPO2
    assert _series(session, MonitoringRecordRow.heart_rate) == EXPECTED_HR
    assert _series(session, MonitoringRecordRow.temperature_c) == EXPECTED_TEMP
    assert _series(session, MonitoringRecordRow.respiratory_rate) == EXPECTED_RR


def test_seed_shows_monotonic_deterioration(session):
    """The scenario's whole point: SpO2 falls and HR rises every single day.

    Phase 3's trend detection is tested against this, so a non-monotonic series
    would make those tests pass for the wrong reason.
    """
    spo2 = _series(session, MonitoringRecordRow.spo2)
    hr = _series(session, MonitoringRecordRow.heart_rate)
    assert all(a > b for a, b in zip(spo2, spo2[1:])), "SpO2 must fall every day"
    assert all(a < b for a, b in zip(hr, hr[1:])), "HR must rise every day"
    assert spo2[0] - spo2[-1] == 7.0, "demo patient drops 7 percentage points"


def test_seed_records_are_chronological(session):
    records = session.scalars(
        select(MedicalRecordRow).order_by(MedicalRecordRow.occurred_on)
    ).all()
    dated = [r.occurred_on for r in records if r.occurred_on is not None]
    assert dated == sorted(dated)
    assert dated[0].year == 2024 and dated[-1].year == 2026
    # Undated rows are social history only; nothing else is allowed to lose its
    # date, because the timeline and the longitudinal comparison both order by it.
    undated = [r for r in records if r.occurred_on is None]
    assert undated and all(r.category == "lifestyle" for r in undated)


def test_seed_prior_findings_span_two_years(session):
    """Longitudinal reasoning needs history on the same dimension as the present."""
    records = session.scalars(
        select(MedicalRecordRow).where(
            MedicalRecordRow.clinical_dimension == "pulmonary_imaging"
        )
    ).all()
    years = {r.occurred_on.year for r in records}
    assert 2025 in years, "a prior pulmonary imaging record must exist"


def test_timeline_is_left_for_the_history_agent(session):
    """The timeline is a materialised view owned by the History Agent (Phase 2).

    Seeding it here would mean two independent generators of the same rows.
    """
    assert session.scalars(select(TimelineEventRow)).all() == []


def test_seed_is_idempotent_after_reset(tmp_db):
    first = reset_and_seed(verbose=False)
    second = reset_and_seed(verbose=False)
    assert first == second == "PT-DEMO-001"

    s = get_session_factory()()
    try:
        assert len(s.scalars(select(PatientRow)).all()) == 1
        assert len(s.scalars(select(MonitoringRecordRow)).all()) == 5
    finally:
        s.close()


# --- Persisted provenance -----------------------------------------------------


def test_imaging_provenance_is_persisted_not_recomputed(session):
    """A row saved as mock must stay labelled mock even if the server's
    configured imaging mode changes afterwards."""
    img = session.scalars(select(MedicalImageRow)).one()
    assert img.source_mode == "mock_preset"
    assert img.badge
    assert "MOCK" in img.badge.upper()
    assert img.findings["source_mode"] == "mock_preset"


def test_imaging_findings_are_round_trippable(session):
    from app.schemas import ImagingFinding

    img = session.scalars(select(MedicalImageRow)).one()
    finding = ImagingFinding.model_validate(img.findings)
    assert finding.has_acute_abnormality is True
    labels = {a.label.value for a in finding.abnormalities}
    assert "multifocal_opacities" in labels
    assert finding.badge == img.badge


# --- Referential integrity ----------------------------------------------------


def test_orphan_row_is_rejected(tmp_db):
    init_db()
    s = get_session_factory()()
    try:
        s.add(
            MonitoringRecordRow(
                id="ORPHAN-1",
                patient_id="PT-DOES-NOT-EXIST",
                recorded_at=datetime.now(timezone.utc),
                spo2=99.0,
            )
        )
        with pytest.raises(IntegrityError):
            s.commit()
        s.rollback()
    finally:
        s.close()


def test_cascade_delete_removes_children(seeded_db):
    s = get_session_factory()()
    try:
        s.add(
            PatientRow(
                id="PT-CASCADE",
                full_name="cascade probe",
                birth_year=1990,
                sex="unknown",
                created_at=datetime.now(timezone.utc),
            )
        )
        s.flush()
        s.add(
            MonitoringRecordRow(
                id="CASC-1",
                patient_id="PT-CASCADE",
                recorded_at=datetime.now(timezone.utc),
                spo2=97.0,
            )
        )
        s.commit()

        s.delete(s.get(PatientRow, "PT-CASCADE"))
        s.commit()

        orphans = s.scalars(
            select(MonitoringRecordRow).where(MonitoringRecordRow.patient_id == "PT-CASCADE")
        ).all()
        assert orphans == []
        # The demo patient must be untouched.
        assert len(s.scalars(select(PatientRow)).all()) == 1
    finally:
        s.close()


# --- Timestamp handling -------------------------------------------------------


def test_timestamps_round_trip_timezone_aware(session):
    """SQLite discards the UTC offset for a plain DateTime(timezone=True).

    These comparisons matter because the system computes observation windows
    and days-between-observations; comparing a naive DB value against an aware
    `datetime.now(timezone.utc)` raises TypeError.
    """
    rows = session.scalars(select(MonitoringRecordRow)).all()
    assert all(r.recorded_at.tzinfo is not None for r in rows)


def test_timestamp_arithmetic_does_not_raise(session):
    rows = session.scalars(
        select(MonitoringRecordRow).order_by(MonitoringRecordRow.recorded_at)
    ).all()
    delta = rows[-1].recorded_at - rows[0].recorded_at
    assert delta.days == 4
    # Mixing a DB-read timestamp with an aware "now" must work.
    assert isinstance(datetime.now(timezone.utc) - rows[-1].recorded_at, object)


def test_nullable_timestamps_stay_null(tmp_db):
    init_db()
    reset_and_seed(verbose=False)
    s = get_session_factory()()
    try:
        s.add(AnalysisRunRow(id="RUN-1", patient_id="PT-DEMO-001", status="pending"))
        s.commit()
        run = s.get(AnalysisRunRow, "RUN-1")
        assert run.started_at is None and run.finished_at is None

        run.started_at = datetime.now(timezone.utc)
        s.commit()
        assert s.get(AnalysisRunRow, "RUN-1").started_at.tzinfo is not None
    finally:
        s.close()


def test_absent_json_payload_is_sql_null(tmp_db):
    """SQLAlchemy's default JSON stores None as the document 'null'.

    Reading the row back still gives None either way, so the difference is
    invisible until a query filters on it — then `IS NULL` matches nothing and
    `IS NOT NULL` matches everything. Several payloads here are legitimately
    absent (a pending run has no final_assessment), so the distinction is real.
    """
    init_db()
    reset_and_seed(verbose=False)
    s = get_session_factory()()
    try:
        s.add(AnalysisRunRow(id="RUN-PENDING", patient_id="PT-DEMO-001", status="pending"))
        s.add(AnalysisRunRow(
            id="RUN-DONE",
            patient_id="PT-DEMO-001",
            status="completed",
            final_assessment={"risk_level": "HIGH"},
        ))
        s.commit()

        pending = s.scalars(
            select(AnalysisRunRow).where(AnalysisRunRow.final_assessment.is_(None))
        ).all()
        assert [r.id for r in pending] == ["RUN-PENDING"]

        done = s.scalars(
            select(AnalysisRunRow).where(AnalysisRunRow.final_assessment.is_not(None))
        ).all()
        assert [r.id for r in done] == ["RUN-DONE"]
        assert s.get(AnalysisRunRow, "RUN-PENDING").final_assessment is None
    finally:
        s.close()


def test_records_without_a_coded_finding_are_queryable_as_null(session):
    """The History Agent selects coded findings with `normalised IS NOT NULL`."""
    coded = session.scalars(
        select(MedicalRecordRow).where(MedicalRecordRow.normalised.is_not(None))
    ).all()
    uncoded = session.scalars(
        select(MedicalRecordRow).where(MedicalRecordRow.normalised.is_(None))
    ).all()
    assert len(coded) == 2
    assert len(uncoded) == 5
    assert not {r.id for r in coded} & {r.id for r in uncoded}


def test_agent_results_link_to_runs(seeded_db):
    s = get_session_factory()()
    try:
        s.add(AnalysisRunRow(id="RUN-2", patient_id="PT-DEMO-001", status="completed"))
        s.flush()
        s.add(
            AgentResultRow(
                id="AR-1",
                run_id="RUN-2",
                seq=0,
                agent_name="monitoring",
                status="ok",
                output_payload={"rapid_deterioration": True},
                fired_rules=[{"rule_id": "R-MON-01", "contribution": 20.0}],
                duration_ms=7,
            )
        )
        s.commit()

        run = s.get(AnalysisRunRow, "RUN-2")
        assert len(run.agent_results) == 1
        assert run.agent_results[0].output_payload["rapid_deterioration"] is True
        assert run.agent_results[0].fired_rules[0]["contribution"] == 20.0
    finally:
        s.close()


# --- Fixtures -----------------------------------------------------------------


def test_fixture_json_is_valid_and_complete(demo_fixture):
    assert demo_fixture["patient"]["id"] == "PT-DEMO-001"
    assert demo_fixture["patient"]["birth_year"] == 1971
    assert len(demo_fixture["medical_records"]) == 5
    assert "current_symptoms" in demo_fixture
    assert "current_imaging" in demo_fixture


def test_fixture_has_a_single_representation_of_prior_findings(demo_fixture):
    """No top-level prior_findings block.

    Those facts are carried as `normalised` on the records that produced them.
    Two representations of the same finding would eventually disagree, which is
    the same reason the seed leaves timeline_events empty.
    """
    assert "prior_findings" not in demo_fixture
    coded = [
        r for r in demo_fixture["medical_records"] if r.get("normalised")
    ]
    assert len(coded) == 2
    assert all(r["category"] == "imaging_report" for r in coded)
    # The chronic diagnosis is a Condition, not also a coded finding.
    diagnoses = [r for r in demo_fixture["medical_records"] if r["category"] == "diagnosis"]
    assert diagnoses and all(not r.get("normalised") for r in diagnoses)
    for r in coded:
        # Everything PriorFinding needs beyond the normalised block must come
        # from the record's own columns, or the History Agent would have to guess.
        assert r["clinical_dimension"], r["id"]
        assert r["occurred_on"], r["id"]


def test_fixture_lifestyle_entries_are_structured(demo_fixture):
    """Risk-factor rules read numbers, not prose."""
    by_key = {lf["key"]: lf for lf in demo_fixture["lifestyle"]}
    assert by_key["former_smoker"]["pack_years"] == 8
    assert by_key["former_smoker"]["current_smoker"] is False
    assert by_key["bmi_overweight"]["bmi"] == 27.4


def test_fixture_current_imaging_is_explicitly_mock(demo_fixture):
    """The demo fixture must never claim to be real model output."""
    ci = demo_fixture["current_imaging"]
    assert ci["source_mode"] == "mock_preset"
    assert "no imaging model" in ci["provenance"].lower()


def test_fixture_declares_itself_synthetic(demo_fixture):
    assert demo_fixture["patient"]["is_synthetic"] is True
    assert "SYNTHETIC" in demo_fixture["_comment"].upper()


def test_monitoring_csv_parses_to_validated_samples(demo_monitoring):
    assert len(demo_monitoring) == 5
    assert [s.spo2 for s in demo_monitoring] == EXPECTED_SPO2
    assert [s.heart_rate for s in demo_monitoring] == EXPECTED_HR
    assert all(s.recorded_at.tzinfo is not None for s in demo_monitoring)


def test_monitoring_csv_timestamps_are_ordered(demo_monitoring):
    stamps = [s.recorded_at for s in demo_monitoring]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps), "timestamps must be unique"


def test_seed_function_accepts_an_injected_fixture(tmp_db):
    """Seeding from an explicit dict (not just the on-disk file) is what lets
    later phases build alternative test patients for the ablation study."""
    init_db()
    fx = load_fixture()
    fx["patient"]["id"] = "PT-ALT-001"
    s = get_session_factory()()
    try:
        assert seed(s, fx) == "PT-ALT-001"
        s.commit()
        assert s.get(PatientRow, "PT-ALT-001") is not None
    finally:
        s.close()
