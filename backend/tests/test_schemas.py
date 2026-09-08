"""Contract tests for the pydantic schemas.

These assert the invariants the rest of the system depends on. Several are
compliance or safety properties rather than ordinary correctness checks — they
are written as tests because a guarantee that is only documented tends to stop
being true.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas import (
    Abnormality,
    Acuity,
    AgentName,
    AgentStatus,
    AgentTrace,
    AnalysisResult,
    ChangeType,
    ClinicalDimension,
    Department,
    FinalAssessment,
    FiredRule,
    HistoricalChange,
    ImagingFinding,
    ImagingLabel,
    ImagingSourceMode,
    MOCK_BADGE_TEXT,
    MonitoringAssessment,
    PatientProfile,
    PriorFinding,
    REPORT_BADGE_TEXT,
    RetrievedEvidence,
    RiskLevel,
    RunConfig,
    RunStatus,
    Severity,
    Significance,
    Symptom,
    SymptomName,
    TrendDirection,
    TrendFeature,
    Urgency,
    VitalMetric,
)

BANNED_FIELDS = {
    "diagnosis",
    "diagnoses",
    "diagnostic_conclusion",
    "primary_diagnosis",
    "icd10_primary",
    "prescription",
    "medication_recommendation",
    "dosage_recommendation",
}


def make_assessment(**overrides) -> FinalAssessment:
    base = dict(
        run_id="RUN-TEST-1",
        patient_id="PT-DEMO-001",
        generated_at=datetime(2026, 9, 8, 8, 30, tzinfo=timezone.utc),
        risk_level=RiskLevel.HIGH,
        risk_score=72.5,
        recommended_department=Department.EMERGENCY,
        urgency=Urgency.IMMEDIATE,
    )
    base.update(overrides)
    return FinalAssessment(**base)


# --- Compliance: the schema cannot express a diagnosis -----------------------


@pytest.mark.parametrize("model", [FinalAssessment, AnalysisResult, HistoricalChange])
def test_no_diagnosis_vocabulary_in_schema(model):
    """The prototype must not be able to output a diagnosis or a prescription.

    Enforced structurally: with no field to hold one, there is nothing to leak.
    """
    present = set(model.model_fields) & BANNED_FIELDS
    assert not present, f"{model.__name__} exposes banned fields: {sorted(present)}"


def test_diagnosis_kwarg_is_rejected():
    """extra='forbid' means a diagnosis cannot be smuggled in as an extra field."""
    with pytest.raises(ValidationError):
        make_assessment(diagnosis="community-acquired pneumonia")


def test_prescription_kwarg_is_rejected():
    with pytest.raises(ValidationError):
        make_assessment(prescription="amoxicillin 500mg TDS")


# --- Contract strictness ----------------------------------------------------


def test_misspelled_field_is_rejected():
    """A typo'd field name must fail loudly rather than be silently dropped."""
    with pytest.raises(ValidationError) as exc:
        Symptom(name=SymptomName.FEVER, value=38.7, unit="C", severty="severe")
    assert "severty" in str(exc.value)


def test_unknown_symptom_cannot_be_invented():
    with pytest.raises(ValidationError):
        Symptom(name="shortness_of_breath")


# --- Imaging provenance labelling -------------------------------------------


def test_source_mode_is_required():
    with pytest.raises(ValidationError):
        ImagingFinding(provenance="unknown origin")


def test_mock_preset_badge_is_autofilled():
    finding = ImagingFinding(source_mode=ImagingSourceMode.MOCK_PRESET, provenance="preset")
    assert finding.badge == MOCK_BADGE_TEXT
    assert not finding.is_real_model


def test_uploaded_report_gets_its_own_accurate_badge():
    """Report parsing is not a model run, but it is not mock data either.

    Labelling it 'MOCK OUTPUT' would misdescribe what happened, so it carries a
    distinct disclosure.
    """
    finding = ImagingFinding(source_mode=ImagingSourceMode.UPLOADED_REPORT, provenance="parsed")
    assert finding.badge == REPORT_BADGE_TEXT
    assert finding.badge != MOCK_BADGE_TEXT
    assert not finding.is_real_model


def test_real_model_carries_no_mock_warning():
    finding = ImagingFinding(source_mode=ImagingSourceMode.REAL_MODEL, provenance="DenseNet121")
    assert finding.badge is None
    assert finding.is_real_model


def test_caller_may_override_badge_but_not_omit_it():
    custom = ImagingFinding(
        source_mode=ImagingSourceMode.MOCK_PRESET, provenance="p", badge="CUSTOM WARNING"
    )
    assert custom.badge == "CUSTOM WARNING"

    blank = ImagingFinding(source_mode=ImagingSourceMode.MOCK_PRESET, provenance="p", badge="   ")
    assert blank.badge == MOCK_BADGE_TEXT


def test_badge_survives_serialization():
    """The badge must reach the frontend; losing it in a round-trip would
    silently un-label mock output in the UI."""
    finding = ImagingFinding(source_mode=ImagingSourceMode.MOCK_PRESET, provenance="p")
    restored = ImagingFinding.model_validate_json(finding.model_dump_json())
    assert restored.badge == MOCK_BADGE_TEXT


# --- Longitudinal comparability ---------------------------------------------


@pytest.mark.parametrize("model", [PriorFinding, Abnormality])
@pytest.mark.parametrize("bad", [-1, 6, 100])
def test_severity_score_is_bounded(model, bad):
    """Both ends of a longitudinal comparison must share one bounded scale."""
    kwargs = (
        dict(dimension=ClinicalDimension.PULMONARY_IMAGING, label="focal_opacity", occurred_on=date(2025, 1, 1))
        if model is PriorFinding
        else dict(label=ImagingLabel.FOCAL_OPACITY)
    )
    with pytest.raises(ValidationError):
        model(severity_score=bad, **kwargs)


def test_prior_and_current_severity_are_directly_comparable():
    """The arithmetic behind change_type: two observations years apart on the
    same dimension must be subtractable."""
    prior = PriorFinding(
        dimension=ClinicalDimension.PULMONARY_IMAGING,
        label="focal_opacity",
        occurred_on=date(2025, 3, 12),
        acuity=Acuity.CHRONIC,
        severity_score=2,
        resolved=True,
    )
    current = Abnormality(
        label=ImagingLabel.MULTIFOCAL_OPACITIES,
        acuity=Acuity.ACUTE,
        severity_score=3,
    )
    assert current.severity_score - prior.severity_score == 1


def test_historical_change_records_its_arithmetic():
    change = HistoricalChange(
        dimension=ClinicalDimension.PULMONARY_IMAGING,
        change_type=ChangeType.WORSENING,
        significance=Significance.HIGH,
        prior_state="RLL focal opacity (resolved)",
        current_state="bilateral multifocal opacities",
        prior_severity=2,
        current_severity=3,
        severity_delta=1,
        basis="severity_score 2 -> 3 on pulmonary_imaging",
    )
    assert change.severity_delta == change.current_severity - change.prior_severity


def test_all_change_types_are_representable():
    """RECURRENT is modelled separately from NEW, so both must round-trip."""
    for change_type in ChangeType:
        c = HistoricalChange(
            dimension=ClinicalDimension.OXYGENATION,
            change_type=change_type,
            significance=Significance.LOW,
            current_state="x",
        )
        assert c.change_type == change_type


# --- Monitoring --------------------------------------------------------------


def _trend_kwargs(**overrides):
    """A complete, valid TrendFeature payload. Phase 3 added `unit` and
    `dimension` as required, so every construction site shares one definition."""
    kwargs = dict(
        metric=VitalMetric.SPO2,
        unit="%",
        dimension=ClinicalDimension.OXYGENATION,
        sample_count=5,
        current=91.0,
        baseline=98.0,
        delta=-7.0,
        delta_pct=-7.14,
        min_value=91.0,
        max_value=98.0,
        slope_per_day=-1.75,
        direction=TrendDirection.WORSENING,
    )
    kwargs.update(overrides)
    return kwargs


def test_trend_feature_still_requires_the_observed_shape():
    """`current`, `min_value`, `max_value` and `direction` are what a trend
    always has. Omitting any of them is a construction error."""
    for missing in ("current", "min_value", "max_value", "direction", "unit", "dimension"):
        kwargs = _trend_kwargs()
        del kwargs[missing]
        with pytest.raises(ValidationError):
            TrendFeature(**kwargs)


def test_baseline_may_be_absent_but_must_be_none_not_zero():
    """Phase 3 deliberately superseded the Phase 1 rule that a trend requires a
    baseline. A lone SpO₂ reading of 91% is clinically meaningful on its own —
    NEWS2 scores it 3 — and refusing to report it because no baseline exists
    would discard the one number that was actually observed.

    What must not happen is a *fabricated* baseline. Absent has to be None.
    """
    trend = TrendFeature(**_trend_kwargs(baseline=None, delta=None, delta_pct=None))
    assert trend.baseline is None
    assert trend.delta is None
    assert trend.current == 91.0


def test_unknown_slope_is_not_reported_as_a_flat_one():
    """The distinction the whole agent rests on: None means 'too few
    observations to fit', 0.0 means 'fitted, and flat'. A two-sample series
    yields r² = 1.0 by construction and must not be presented as a trend."""
    unfitted = TrendFeature(**_trend_kwargs(slope_per_day=None, r_squared=None,
                                           direction=TrendDirection.UNKNOWN))
    flat = TrendFeature(**_trend_kwargs(slope_per_day=0.0, r_squared=None,
                                       direction=TrendDirection.STABLE))
    assert unfitted.slope_per_day is None
    assert flat.slope_per_day == 0.0
    assert unfitted.slope_per_day != flat.slope_per_day
    assert unfitted.direction != flat.direction


def test_anomaly_score_is_normalised():
    with pytest.raises(ValidationError):
        TrendFeature(**_trend_kwargs(anomaly_score=1.4))
    assert TrendFeature(**_trend_kwargs(anomaly_score=0.93)).anomaly_score == 0.93


def test_threshold_requires_a_source_when_breached():
    """A threshold with no citation is an assertion, not evidence."""
    trend = TrendFeature(**_trend_kwargs(
        absolute_threshold_breach=True,
        threshold_value=92.0,
        threshold_source="NEWS2 (Royal College of Physicians, 2017)",
        news2_score=3,
        news2_band="<= 91%",
    ))
    assert trend.absolute_threshold_breach
    assert trend.threshold_source
    assert trend.news2_score == 3


def test_news2_score_is_none_for_metrics_it_does_not_cover():
    """Sleep and activity are not NEWS2 parameters. None means 'not scoreable';
    0 would mean 'normal', which is a different claim."""
    trend = TrendFeature(**_trend_kwargs(
        metric=VitalMetric.ACTIVITY_STEPS,
        unit="steps",
        dimension=ClinicalDimension.FUNCTIONAL_STATUS,
        current=600.0,
        news2_score=None,
    ))
    assert trend.news2_score is None
    assert trend.news2_score != 0


def test_partial_news2_caveat_cannot_be_omitted():
    """A partial NEWS2 total read as a full one is the specific misreading the
    validator exists to prevent."""
    from app.reasoning.thresholds import NEWS2_PARTIAL_SCORE_CAVEAT

    assessment = MonitoringAssessment(window_days=5, sample_count=5, news2_partial_total=7)
    assert assessment.news2_caveat == NEWS2_PARTIAL_SCORE_CAVEAT
    assert "NOT a NEWS2 total" in assessment.news2_caveat


# --- Run container -----------------------------------------------------------


def test_analysis_result_round_trips_with_typed_outputs():
    """The dashboard reads typed agent outputs, not `object`."""
    result = AnalysisResult(
        run_id="RUN-TEST-1",
        patient_id="PT-DEMO-001",
        status=RunStatus.COMPLETED,
        config=RunConfig(),
        history=PatientProfile(patient_id="PT-DEMO-001", full_name="Demo", age=55),
        monitoring=MonitoringAssessment(window_days=5, sample_count=5, rapid_deterioration=True),
        imaging=ImagingFinding(source_mode=ImagingSourceMode.MOCK_PRESET, provenance="preset"),
        assessment=make_assessment(
            score_breakdown=[
                FiredRule(
                    rule_id="R-MON-SPO2-ABS-01",
                    category="monitoring",
                    description="SpO2 below acute hypoxaemia threshold",
                    contribution=20.0,
                    evidence="91% vs 92%",
                    value=91.0,
                    threshold=92.0,
                )
            ],
            evidence=[
                RetrievedEvidence(
                    chunk_id="c1",
                    text="SpO2 below 92% indicates acute hypoxaemia.",
                    source="BTS",
                    relevance_score=0.81,
                    query="spo2 threshold",
                    retrieved_for="monitoring: spo2 worsening",
                )
            ],
            limitations=["Synthetic data"],
        ),
        traces=[
            AgentTrace(
                agent_name=AgentName.MONITORING, status=AgentStatus.OK, duration_ms=12
            )
        ],
    )

    restored = AnalysisResult.model_validate_json(result.model_dump_json())
    assert restored == result

    assert restored.monitoring.rapid_deterioration is True
    assert restored.assessment.risk_level == RiskLevel.HIGH
    assert restored.assessment.score_breakdown[0].contribution == 20.0
    assert restored.imaging.badge == MOCK_BADGE_TEXT


def test_score_breakdown_sums_to_reported_score():
    """The risk score must be derivable from the published rule contributions."""
    rules = [
        FiredRule(rule_id="A", category="monitoring", description="a", contribution=20.0),
        FiredRule(rule_id="B", category="triage", description="b", contribution=10.0),
        FiredRule(rule_id="C", category="imaging", description="c", contribution=15.0),
    ]
    assessment = make_assessment(
        risk_score=45.0, history_multiplier=1.0, score_breakdown=rules
    )
    assert sum(r.contribution for r in assessment.score_breakdown) == assessment.risk_score


# --- Enum stability ----------------------------------------------------------


def test_enum_values_are_stable():
    """The frontend hardcodes these strings for colour mapping; a silent rename
    would break the UI without any type error, because both sides regenerate."""
    assert [r.value for r in RiskLevel] == ["LOW", "MEDIUM", "HIGH"]
    assert [u.value for u in Urgency] == ["IMMEDIATE", "URGENT", "SEMI_URGENT", "ROUTINE"]
    assert [c.value for c in ChangeType] == [
        "new", "worsening", "improving", "stable", "recurrent",
    ]
    assert [a.value for a in AgentName] == [
        "history", "triage", "imaging", "monitoring", "knowledge", "coordinator",
    ]


def test_department_values_are_display_strings():
    """Department is rendered verbatim in the UI, so its values are labels."""
    assert Department.EMERGENCY.value == "Emergency Department"
    assert all(" " in d.value or d.value.isalpha() for d in Department)


def test_severity_and_symptom_vocabulary_cover_the_mvp_scenario():
    required = {"fever", "cough", "dyspnea", "chest_pain", "chest_tightness"}
    assert required <= {s.value for s in SymptomName}
    assert {s.value for s in Severity} == {"mild", "moderate", "severe"}
