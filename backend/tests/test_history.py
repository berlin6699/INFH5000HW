"""History Agent and its rule engine.

Two layers are tested separately, because they fail differently:

* `app.reasoning.history_rules` is pure — tested from constructed inputs, no
  database, so a threshold change shows up as a specific rule assertion rather
  than as a diff in a seeded profile.
* `app.agents.history` owns the ORM, the timeline and the trace — tested against
  the seeded temporary database from conftest.

The demo-scenario assertions (55-year-old, hypertension, prior pulmonary
abnormality) are kept in their own class. Those are the premises the project
spec states, and if a fixture edit silently breaks one, the whole longitudinal
story the dashboard tells stops being the story that was specified.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select

from app.agents.history import (
    UnknownPatientError,
    _human,
    build_timeline,
    load_record,
    persist_timeline,
    resolve_reference_date,
    run_history,
)
from app.db.models import MedicalRecordRow, TimelineEventRow
from app.db.session import get_session_factory
from app.reasoning.history_rules import (
    HISTORY_RULE_CATEGORY,
    MULTIPLIER_RULE_ID,
    HistoryInputs,
    _age_factor,
    _bmi_factor,
    _smoking_factor,
    compute_history_multiplier,
    derive_baseline_vulnerabilities,
    derive_data_gaps,
    derive_history,
    derive_risk_factors,
    history_fired_rules,
)
from app.reasoning.thresholds import (
    HISTORY_MULTIPLIER_CEILING,
    HISTORY_MULTIPLIER_FLOOR,
    WEIGHT_AGE_MIDDLE,
    WEIGHT_BMI_OVERWEIGHT,
    WEIGHT_CHRONIC_CARDIAC,
    WEIGHT_PRIOR_PULMONARY_FINDING,
)
from app.safety import DISCLAIMER, audit_prose
from app.schemas import (
    Acuity,
    Allergy,
    ClinicalDimension,
    Condition,
    Medication,
    PriorFinding,
    RiskFactor,
    SocialHistory,
)

DEMO_ID = "PT-DEMO-001"
REFERENCE = date(2026, 9, 8)


def _inputs(**overrides) -> HistoryInputs:
    """A minimal, gap-free history. Individual tests remove what they examine."""
    defaults = dict(
        reference_date=REFERENCE,
        age=55,
        chronic_conditions=(
            Condition(name="Essential hypertension", affects_dimension=ClinicalDimension.CARDIAC_RATE),
        ),
        previous_findings=(
            PriorFinding(
                dimension=ClinicalDimension.PULMONARY_IMAGING,
                label="focal_opacity",
                occurred_on=date(2025, 3, 12),
                severity_score=2,
                resolved=True,
            ),
        ),
        social_history=(
            SocialHistory(key="former_smoker", factor="Former smoker", current_smoker=False, pack_years=8),
            SocialHistory(key="bmi_overweight", factor="BMI", bmi=27.4),
        ),
        active_medications=(Medication(name="Amlodipine", dose="5 mg"),),
        allergies=(Allergy(allergen="Penicillin", severity="moderate"),),
        last_record_date=date(2026, 2, 10),
        last_prior_imaging_date=date(2025, 6, 20),
        monitoring_sample_count=5,
    )
    defaults.update(overrides)
    return HistoryInputs(**defaults)


def _blank_inputs(**overrides) -> HistoryInputs:
    """A record containing nothing, for asserting that gaps are reported.

    Merged as a dict rather than passed as kwargs, so a caller can put one
    field back without colliding with the blanking defaults.
    """
    blank: dict = dict(
        age=None,
        chronic_conditions=(),
        previous_findings=(),
        social_history=(),
        active_medications=(),
        allergies=(),
        last_record_date=None,
        last_prior_imaging_date=None,
        monitoring_sample_count=0,
    )
    blank.update(overrides)
    return _inputs(**blank)


# --- Risk-factor bands --------------------------------------------------------


class TestAgeBands:
    @pytest.mark.parametrize("age,expected", [(49, None), (50, "R-HIS-AGE-02"), (64, "R-HIS-AGE-02")])
    def test_middle_age_band(self, age, expected):
        factor = _age_factor(age)
        assert (factor.rule_id if factor else None) == expected

    def test_elderly_band_supersedes_middle(self):
        """65 matches both cutoffs; the heavier one must win."""
        factor = _age_factor(65)
        assert factor is not None
        assert factor.rule_id == "R-HIS-AGE-01"
        assert factor.weight > WEIGHT_AGE_MIDDLE.value

    def test_unknown_age_produces_no_factor(self):
        """A missing birth year is a gap, not a young patient."""
        assert _age_factor(None) is None

    def test_age_factor_states_its_threshold_and_source(self):
        factor = _age_factor(55)
        assert factor is not None
        assert "55" in factor.factor
        assert factor.source, "a weight without a source is an assertion, not evidence"


class TestSmokingBands:
    def test_light_former_exposure_is_considered_but_unweighted(self):
        """8 pack-years sits below the band. Reported at 1.0, not omitted."""
        factor = _smoking_factor(
            SocialHistory(key="former_smoker", factor="Former smoker", current_smoker=False, pack_years=8)
        )
        assert factor is not None
        assert factor.rule_id == "R-HIS-SMOKE-03"
        assert factor.weight == HISTORY_MULTIPLIER_FLOOR
        assert "no adjustment applied" in factor.factor

    def test_significant_exposure(self):
        factor = _smoking_factor(
            SocialHistory(key="smoker", factor="Current smoker", current_smoker=True, pack_years=30)
        )
        assert factor is not None
        assert factor.rule_id == "R-HIS-SMOKE-01"
        assert factor.weight > HISTORY_MULTIPLIER_FLOOR

    def test_unquantified_current_smoking_does_not_crash(self):
        """Regression: `current_smoker=True` with no pack-years used to reach
        `f"{None:g}"` and raise TypeError."""
        factor = _smoking_factor(
            SocialHistory(key="smoker", factor="Current smoker", current_smoker=True)
        )
        assert factor is not None
        assert factor.rule_id == "R-HIS-SMOKE-02"
        assert "exposure not quantified" in factor.factor
        assert "None" not in factor.factor


class TestBmiBands:
    @pytest.mark.parametrize(
        "bmi,expected",
        [(22.0, None), (24.9, None), (25.0, "R-HIS-BMI-02"), (27.4, "R-HIS-BMI-02"), (30.0, "R-HIS-BMI-01")],
    )
    def test_bands(self, bmi, expected):
        factor = _bmi_factor(SocialHistory(key="bmi", factor="BMI", bmi=bmi))
        assert (factor.rule_id if factor else None) == expected

    def test_absent_bmi_produces_no_factor(self):
        assert _bmi_factor(SocialHistory(key="smoker", factor="Current smoker", current_smoker=True)) is None


class TestMultiplier:
    def test_empty_history_is_neutral(self):
        assert compute_history_multiplier([]) == HISTORY_MULTIPLIER_FLOOR

    def test_product_of_weights(self):
        factors = [
            RiskFactor(rule_id="A", factor="a", category="demographic", weight=1.1),
            RiskFactor(rule_id="B", factor="b", category="comorbidity", weight=1.2),
        ]
        assert compute_history_multiplier(factors) == pytest.approx(1.32)

    def test_clamped_at_ceiling(self):
        """Five 1.15 modifiers multiply to ~2.0; the cap is what stops a long
        comorbidity list from dominating the score."""
        factors = [
            RiskFactor(rule_id=f"R{i}", factor=f"f{i}", category="comorbidity", weight=1.15)
            for i in range(5)
        ]
        assert compute_history_multiplier(factors) == HISTORY_MULTIPLIER_CEILING

    def test_never_below_floor(self):
        """Weights are constrained ge=1.0 by the schema, so this is defence in
        depth against a future hand-built factor."""
        factor = RiskFactor(rule_id="X", factor="x", category="historical", weight=1.0)
        assert compute_history_multiplier([factor]) >= HISTORY_MULTIPLIER_FLOOR


# --- Vulnerabilities and gaps -------------------------------------------------


class TestVulnerabilities:
    def test_derived_from_the_record_only(self):
        vulns = derive_baseline_vulnerabilities(_inputs())
        assert vulns == [ClinicalDimension.PULMONARY_IMAGING, ClinicalDimension.CARDIAC_RATE]

    def test_never_inferred_across_dimensions(self):
        """A prior pulmonary finding flags pulmonary_imaging, NOT oxygenation.
        Crossing that line would be reasoning from plausibility, not the record."""
        vulns = derive_baseline_vulnerabilities(_inputs())
        assert ClinicalDimension.OXYGENATION not in vulns
        assert ClinicalDimension.RESPIRATORY_RATE not in vulns

    def test_stable_enum_order_regardless_of_input_order(self):
        """The dashboard renders this list; a changing order looks like a
        changing patient."""
        cardiac_first = _inputs(
            previous_findings=(),
            chronic_conditions=(
                Condition(name="Essential hypertension", affects_dimension=ClinicalDimension.CARDIAC_RATE),
                Condition(name="COPD", affects_dimension=ClinicalDimension.RESPIRATORY_RATE),
            ),
        )
        respiratory_first = _inputs(
            previous_findings=(),
            chronic_conditions=(
                Condition(name="COPD", affects_dimension=ClinicalDimension.RESPIRATORY_RATE),
                Condition(name="Essential hypertension", affects_dimension=ClinicalDimension.CARDIAC_RATE),
            ),
        )
        assert derive_baseline_vulnerabilities(cardiac_first) == derive_baseline_vulnerabilities(
            respiratory_first
        )

    def test_blank_record_has_no_vulnerabilities(self):
        assert derive_baseline_vulnerabilities(_blank_inputs()) == []

    def test_condition_without_a_dimension_is_not_evidence(self):
        """`affects_dimension` is optional; a condition that was never mapped
        onto an axis cannot create one."""
        vulns = derive_baseline_vulnerabilities(
            _blank_inputs(chronic_conditions=(Condition(name="Migraine"),))
        )
        assert vulns == []


class TestDataGaps:
    def test_blank_record_reports_every_absence(self):
        gaps = derive_data_gaps(_blank_inputs())
        assert any("no personal baseline" in g for g in gaps)
        assert any("No prior chest imaging" in g for g in gaps)
        assert any("No continuous monitoring data" in g for g in gaps)
        assert any("No dated clinical records" in g for g in gaps)
        assert any("No active medications" in g for g in gaps)
        assert any("No allergies recorded" in g for g in gaps)
        assert any("No smoking history" in g for g in gaps)
        assert any("No BMI" in g for g in gaps)

    def test_complete_history_reports_only_the_long_interval(self):
        """The demo record is well populated; the only honest gap is that the
        prior imaging is 445 days old."""
        gaps = derive_data_gaps(_inputs())
        assert len(gaps) == 1
        assert "445 days old" in gaps[0]

    def test_empty_allergy_list_is_not_reassuring(self):
        """An absent allergy list means unknown, not none."""
        gaps = derive_data_gaps(_inputs(allergies=()))
        assert any("not evidence of no allergy" in g for g in gaps)

    def test_stale_record_interval_is_computed_in_months(self):
        gaps = derive_data_gaps(_inputs(last_record_date=date(2023, 1, 1)))
        assert any("months old" in g for g in gaps)

    def test_gaps_are_plain_strings_the_ui_can_render(self):
        for gap in derive_data_gaps(_blank_inputs()):
            assert isinstance(gap, str) and gap.endswith(".")


# --- Trace --------------------------------------------------------------------


class TestFiredRules:
    def test_one_rule_per_factor_plus_a_summary(self):
        factors = derive_risk_factors(_inputs())
        rules = history_fired_rules(factors, compute_history_multiplier(factors))
        assert len(rules) == len(factors) + 1
        assert rules[-1].rule_id == MULTIPLIER_RULE_ID

    def test_every_contribution_is_zero(self):
        """History modifies the score multiplicatively. A fake additive
        contribution here would break the property that summing score_breakdown
        reproduces risk_score."""
        rules = history_fired_rules(derive_risk_factors(_inputs()), 1.3698)
        assert all(r.contribution == 0.0 for r in rules)

    def test_every_factor_rule_id_appears_in_the_trace(self):
        """The UI anchors on rule ids, so a factor without a matching rule is
        unexplainable."""
        factors = derive_risk_factors(_inputs())
        rules = history_fired_rules(factors, compute_history_multiplier(factors))
        rule_ids = {r.rule_id for r in rules}
        assert {f.rule_id for f in factors} <= rule_ids

    def test_summary_rule_states_the_arithmetic(self):
        factors = derive_risk_factors(_inputs())
        rules = history_fired_rules(factors, compute_history_multiplier(factors))
        summary = rules[-1]
        assert summary.category == HISTORY_RULE_CATEGORY
        assert summary.threshold == HISTORY_MULTIPLIER_CEILING
        assert summary.value == pytest.approx(1.3698)
        assert "x" in summary.evidence and "clamped to" in summary.evidence

    def test_clamping_is_disclosed_when_it_happens(self):
        factors = [
            RiskFactor(rule_id=f"R{i}", factor=f"f{i}", category="comorbidity", weight=1.2)
            for i in range(5)
        ]
        rules = history_fired_rules(factors, compute_history_multiplier(factors))
        assert "clamped at ceiling" in rules[-1].description
        assert f"{len(factors)} of {len(factors)}" in rules[-1].evidence

    def test_unweighted_factors_are_excluded_from_the_product_display(self):
        factors = derive_risk_factors(_inputs())
        rules = history_fired_rules(factors, compute_history_multiplier(factors))
        active = [f for f in factors if f.weight != HISTORY_MULTIPLIER_FLOOR]
        assert f"{len(active)} of {len(factors)} factors adjusted" in rules[-1].evidence

    def test_no_rules_for_a_blank_record_still_yields_the_summary(self):
        rules = history_fired_rules([], HISTORY_MULTIPLIER_FLOOR)
        assert len(rules) == 1
        assert rules[0].value == HISTORY_MULTIPLIER_FLOOR


# --- End-to-end derivation ----------------------------------------------------


class TestDeriveHistory:
    def test_all_outputs_are_tuples(self):
        d = derive_history(_inputs())
        assert isinstance(d.risk_factors, tuple)
        assert isinstance(d.fired_rules, tuple)
        assert isinstance(d.baseline_vulnerabilities, tuple)
        assert isinstance(d.data_gaps, tuple)

    def test_deterministic(self):
        """Two derivations of the same inputs must be identical, or the
        dashboard would show a different patient on refresh."""
        inputs = _inputs()
        a, b = derive_history(inputs), derive_history(inputs)
        assert a.risk_factors == b.risk_factors
        assert a.history_multiplier == b.history_multiplier
        assert [r.rule_id for r in a.fired_rules] == [r.rule_id for r in b.fired_rules]

    def test_pure_and_side_effect_free(self):
        """The rules must not mutate the inputs they were handed."""
        conditions = [Condition(name="COPD", affects_dimension=ClinicalDimension.RESPIRATORY_RATE)]
        inputs = _blank_inputs(chronic_conditions=tuple(conditions))
        derive_history(inputs)
        assert conditions[0].model_dump() == Condition(
            name="COPD", affects_dimension=ClinicalDimension.RESPIRATORY_RATE
        ).model_dump()


class TestDemoScenario:
    """The premises stated in the project spec. These are the load-bearing
    facts the whole longitudinal narrative rests on."""

    def test_demo_patient_is_55_with_hypertension_and_a_prior_pulmonary_abnormality(self):
        d = derive_history(_inputs())
        assert any(f.rule_id == "R-HIS-AGE-02" for f in d.risk_factors)
        assert any(f.rule_id == "R-HIS-COMORB-CARD-01" for f in d.risk_factors)
        assert any(f.rule_id == "R-HIS-PRIOR-PULM-01" for f in d.risk_factors)

    def test_multiplier_is_the_product_of_four_weights(self):
        """1.05 (age 55) x 1.12 (chronic cardiac) x 1.12 (prior pulmonary)
        x 1.04 (BMI 27.4). The 8-pack-year smoking factor is considered and
        deliberately contributes nothing."""
        expected = (
            WEIGHT_AGE_MIDDLE.value
            * WEIGHT_CHRONIC_CARDIAC.value
            * WEIGHT_PRIOR_PULMONARY_FINDING.value
            * WEIGHT_BMI_OVERWEIGHT.value
        )
        d = derive_history(_inputs())
        assert d.history_multiplier == pytest.approx(expected, abs=1e-4)
        assert d.history_multiplier == pytest.approx(1.3698, abs=1e-4)

    def test_smoking_is_considered_without_adjusting(self):
        d = derive_history(_inputs())
        smoking = [f for f in d.risk_factors if f.category == "lifestyle" and "smoker" in f.factor.lower()]
        assert len(smoking) == 1
        assert smoking[0].weight == HISTORY_MULTIPLIER_FLOOR

    def test_demo_history_is_above_neutral_but_below_the_cap(self):
        """HIGH risk in the demo must come from the acute presentation, not from
        history maxing out on its own."""
        d = derive_history(_inputs())
        assert HISTORY_MULTIPLIER_FLOOR < d.history_multiplier < HISTORY_MULTIPLIER_CEILING


# --- Loading from the database ------------------------------------------------


class TestLoadRecord:
    def test_reference_date_comes_from_the_patients_own_data(self, session):
        """Not the wall clock. `datetime.now()` would age the demo patient every
        year the project sits unused and start firing the stale-record rules."""
        assert resolve_reference_date(session, DEMO_ID) == REFERENCE

    def test_explicit_reference_date_wins(self, session):
        assert resolve_reference_date(session, DEMO_ID, date(2020, 1, 1)) == date(2020, 1, 1)

    def test_unknown_patient_raises(self, session):
        with pytest.raises(UnknownPatientError):
            load_record(session, "PT-DOES-NOT-EXIST")

    def test_age_is_a_year_difference_not_a_precise_age(self, session):
        """The record holds a birth year, so whether the birthday has passed is
        unknowable. Claiming precision would invent a fact."""
        record = load_record(session, DEMO_ID)
        assert record.age == 55
        assert record.age == REFERENCE.year - record.birth_year

    def test_social_history_is_typed_not_a_raw_dict(self, session):
        record = load_record(session, DEMO_ID)
        assert len(record.social_history) == 2
        smoker = next(s for s in record.social_history if s.pack_years is not None)
        assert smoker.pack_years == 8 and smoker.current_smoker is False
        bmi = next(s for s in record.social_history if s.bmi is not None)
        assert bmi.bmi == 27.4

    def test_social_history_is_excluded_from_clinical_notes(self, session):
        record = load_record(session, DEMO_ID)
        assert len(record.notes) == 7
        assert len(record.clinical_notes) == 5

    def test_coded_findings_become_prior_findings(self, session):
        record = load_record(session, DEMO_ID)
        labels = {f.label for f in record.previous_findings}
        assert labels == {"focal_opacity", "fibrosis"}
        assert all(f.dimension == ClinicalDimension.PULMONARY_IMAGING for f in record.previous_findings)

    def test_notes_as_of_shrinks_monotonically_with_the_anchor(self, session):
        """Directly exercises the leakage guard the replay tests depend on."""
        anchors = (date(2026, 9, 8), date(2025, 4, 1), date(2024, 6, 1), date(2020, 1, 1))
        kept = [
            [n.id for n in load_record(session, DEMO_ID, a).notes_as_of] for a in anchors
        ]
        assert kept == [
            ["MR-001", "MR-002", "MR-003", "MR-004", "MR-005"],
            ["MR-001", "MR-002", "MR-003"],
            ["MR-001"],
            [],
        ]
        assert [len(k) for k in kept] == sorted((len(k) for k in kept), reverse=True)

    def test_undated_social_history_survives_every_anchor(self, session):
        """Social history rows carry no event date, so they cannot be filtered
        by one — dropping them would silently erase smoking and BMI exposure."""
        for anchor in (date(2026, 9, 8), date(2020, 1, 1)):
            record = load_record(session, DEMO_ID, anchor)
            assert len(record.social_history) == 2

    def test_studies_split_at_the_reference_date(self, session):
        record = load_record(session, DEMO_ID)
        assert [s.id for s in record.prior_studies] == []
        assert [s.id for s in record.current_studies] == ["IMG-CURRENT-001"]
        # The comparison anchor comes from the historical reports, which live in
        # medical_records rather than medical_images.
        assert record.last_prior_imaging_date == date(2025, 6, 20)

    def test_a_normalised_finding_without_a_dimension_is_rejected(self, session):
        """Without a dimension the finding cannot be compared against the
        present, which is the entire point of storing it coded."""
        row = session.scalar(
            select(MedicalRecordRow).where(MedicalRecordRow.id == "MR-002")
        )
        row.clinical_dimension = None
        session.flush()
        with pytest.raises(ValueError, match="no clinical_dimension"):
            load_record(session, DEMO_ID)

    def test_a_pulmonary_label_outside_the_vocabulary_is_rejected(self, session):
        """Phase 6 matches a prior study against a current abnormality by label.
        A label outside the closed vocabulary can never match, and would silently
        report a known finding as new."""
        row = session.scalar(select(MedicalRecordRow).where(MedicalRecordRow.id == "MR-004"))
        row.normalised = {**row.normalised, "label": "some_free_text_finding"}
        session.flush()
        with pytest.raises(ValueError, match="not an ImagingLabel value"):
            load_record(session, DEMO_ID)


# --- Timeline -----------------------------------------------------------------


class TestTimeline:
    def test_chronological_with_the_current_cluster_last(self, session):
        record = load_record(session, DEMO_ID)
        events = build_timeline(record)
        history = [e for e in events if not e.is_current]
        current = [e for e in events if e.is_current]
        assert [e.occurred_on for e in history] == sorted(e.occurred_on for e in history)
        assert events[-len(current):] == current, "the 'Today' cluster must sit at the head of the view"

    def test_sort_order_is_contiguous_from_zero(self, session):
        record = load_record(session, DEMO_ID)
        orders = [e.sort_order for e in build_timeline(record)]
        assert orders == list(range(len(orders)))

    def test_monitoring_is_one_window_event_not_one_per_sample(self, session):
        """Five daily rows would bury the clinical history."""
        record = load_record(session, DEMO_ID)
        events = build_timeline(record)
        assert sum(1 for e in events if e.event_type.value == "current") == 1
        window = next(e for e in events if e.event_type.value == "current")
        assert window.payload["sample_count"] == 5
        assert window.payload["window_days"] == 5
        assert len(window.payload["series"]) == 5

    def test_undated_records_are_omitted_rather_than_dated_falsely(self, session):
        """Social history has no event date; inventing one would misrepresent it."""
        record = load_record(session, DEMO_ID)
        ids = {e.payload.get("record_id") for e in build_timeline(record)}
        assert not any(i and i.endswith("-LS-former_smoker") for i in ids)

    def test_no_user_facing_label_contains_a_raw_enum_value(self, session):
        """'Chest X-ray — multifocal_opacities' is an implementation detail
        leaking into the clinical narrative."""
        record = load_record(session, DEMO_ID)
        for event in build_timeline(record):
            for field in (event.label, event.detail):
                if field:
                    assert "_" not in field or "x10e9" in field, f"{event.label!r}: {field!r}"

    def test_imaging_events_carry_their_provenance_badge(self, session):
        """A mock finding must stay labelled mock wherever it is rendered."""
        record = load_record(session, DEMO_ID)
        current = next(
            e for e in build_timeline(record) if e.payload.get("study_id") == "IMG-CURRENT-001"
        )
        assert current.payload["source_mode"] == "mock_preset"
        assert current.payload["badge"], "mock output reached the timeline unlabelled"

    def test_record_events_carry_their_structured_source_detail(self, session):
        """The UI renders lab values from data; the lossy prose fallback turns
        {"crp_mg_l": 84} into 'crp mg l: 84'."""
        record = load_record(session, DEMO_ID)
        lab = next(e for e in build_timeline(record) if e.payload.get("record_id") == "MR-003")
        assert lab.payload["source_detail"] == {
            "crp_mg_l": 84,
            "wbc_x10e9_l": 13.8,
            "reference_crp": "<5",
        }

    def test_human_readable_label_helper(self):
        assert _human("multifocal_opacities") == "multifocal opacities"
        assert _human(ClinicalDimension.PULMONARY_IMAGING) == "pulmonary imaging"
        assert _human(Acuity.ACUTE) == "acute"
        assert _human("RLL") == "RLL"


class TestTimelinePersistence:
    def test_rewriting_is_idempotent(self, session):
        """The timeline is a derived view rebuilt every run; a re-run must
        converge on identical rows rather than accumulate duplicates."""
        record = load_record(session, DEMO_ID)
        first = persist_timeline(session, DEMO_ID, build_timeline(record))
        session.commit()
        second = persist_timeline(session, DEMO_ID, build_timeline(record))
        session.commit()

        assert first == second
        rows = session.scalars(
            select(TimelineEventRow).where(TimelineEventRow.patient_id == DEMO_ID)
        ).all()
        assert len(rows) == first
        assert len({r.id for r in rows}) == len(rows)

    def test_ids_are_deterministic(self, session):
        record = load_record(session, DEMO_ID)
        events = build_timeline(record)
        persist_timeline(session, DEMO_ID, events)
        session.commit()
        ids = {
            r.id
            for r in session.scalars(
                select(TimelineEventRow).where(TimelineEventRow.patient_id == DEMO_ID)
            ).all()
        }
        assert ids == {f"TL-{DEMO_ID}-{e.sort_order:03d}" for e in events}

    def test_event_type_and_dimension_survive_the_round_trip(self, session):
        record = load_record(session, DEMO_ID)
        events = build_timeline(record)
        persist_timeline(session, DEMO_ID, events)
        session.commit()

        rows = session.scalars(
            select(TimelineEventRow)
            .where(TimelineEventRow.patient_id == DEMO_ID)
            .order_by(TimelineEventRow.sort_order)
        ).all()
        assert len(rows) == len(events)
        for row, event in zip(rows, events, strict=True):
            assert row.event_type == event.event_type.value
            assert row.occurred_on == event.occurred_on
            assert row.is_current == event.is_current
            assert row.payload == event.payload
            assert (row.clinical_dimension is None) == (event.dimension is None)


# --- Agent entry point --------------------------------------------------------


class TestRunHistory:
    def test_demo_profile_is_complete_and_ok(self, session):
        profile, trace = run_history(session, DEMO_ID)
        assert trace.status.value == "ok"
        assert trace.error is None
        assert profile is not None
        assert profile.patient_id == DEMO_ID
        assert profile.age == 55
        assert profile.history_multiplier == pytest.approx(1.3698, abs=1e-4)
        assert profile.baseline_vulnerabilities == [
            ClinicalDimension.PULMONARY_IMAGING,
            ClinicalDimension.CARDIAC_RATE,
        ]
        assert len(profile.timeline) == 8
        assert len(profile.social_history) == 2

    def test_trace_reports_llm_was_not_used(self, session):
        """Honesty requirement: the offline build performs no LLM inference, and
        the trace must not imply otherwise."""
        _, trace = run_history(session, DEMO_ID)
        assert trace.llm_used is False
        assert trace.agent_name.value == "history"

    def test_trace_notes_are_parseable_key_value_pairs(self, session):
        """Regression: '1 chronic conditions' pluralised badly, and
        'age=55 from birth_year=1971' was not parseable either."""
        _, trace = run_history(session, DEMO_ID)
        parsed = [n for n in trace.notes if "=" in n and not n.startswith("data_gap")]
        notes = dict(n.split("=", 1) for n in parsed)

        assert notes["reference_date"] == "2026-09-08"
        assert notes["age"] == "55"
        assert notes["birth_year"] == "1971"
        assert notes["prior_findings"] == "2"
        assert notes["chronic_conditions"] == "1"
        assert notes["timeline_events"] == "8"
        assert notes["history_multiplier"] == "x1.3698"

        # No parsed value carries trailing prose.
        assert all("(" not in v and " from " not in v for v in notes.values())
        # The rationale survives as its own prose note rather than being dropped.
        assert any("not the wall clock" in n for n in trace.notes)
        assert any(n.startswith("data_gap:") for n in trace.notes)

    def test_summary_passes_the_safety_guard(self, session):
        profile, _ = run_history(session, DEMO_ID)
        assert audit_prose({"history_summary": profile.summary}) == []

    def test_summary_states_history_without_asserting_a_diagnosis(self, session):
        profile, _ = run_history(session, DEMO_ID)
        assert profile.summary.startswith("55-year-old male with Essential hypertension")
        assert "prior findings" in profile.summary
        assert "x1.3698" in profile.summary
        lowered = profile.summary.lower()
        for forbidden in ("diagnosis is", "diagnosed with", "confirmed diagnosis", "pneumonia"):
            assert forbidden not in lowered

    def test_summary_joins_clauses_without_a_comma_before_with(self, session):
        """Regression: 'male, with Essential hypertension' is not the clinical register."""
        profile, _ = run_history(session, DEMO_ID)
        assert ", with " not in profile.summary

    def test_persist_false_writes_nothing(self, session):
        profile, trace = run_history(session, DEMO_ID, persist=False)
        assert profile is not None
        assert profile.timeline, "the profile still carries the timeline in memory"
        assert "timeline_events=0" in trace.notes
        assert session.scalars(select(TimelineEventRow)).all() == []

    def test_the_agent_flushes_but_does_not_commit(self, session):
        """The caller owns the transaction. Committing here would silently
        commit whatever else the surrounding pipeline had staged.

        Observed from a second session: an uncommitted flush is invisible outside
        the writing transaction, so if the agent had committed, these rows would
        be readable there.
        """
        profile, _ = run_history(session, DEMO_ID)
        assert profile is not None
        assert session.scalars(
            select(TimelineEventRow).where(TimelineEventRow.patient_id == DEMO_ID)
        ).all(), "the agent must flush, or the caller cannot see what it wrote"

        other = get_session_factory()()
        try:
            assert other.scalars(
                select(TimelineEventRow).where(TimelineEventRow.patient_id == DEMO_ID)
            ).all() == []
        finally:
            other.close()

    def test_unknown_patient_is_captured_in_the_trace_not_raised(self, session):
        """The workflow fans out to six agents; one failing must not take the
        others down."""
        profile, trace = run_history(session, "PT-DOES-NOT-EXIST")
        assert profile is None
        assert trace.status.value == "error"
        assert "UnknownPatientError" in (trace.error or "")
        assert trace.agent_name.value == "history"

    def test_a_corrupt_record_is_captured_in_the_trace(self, session):
        row = session.scalar(select(MedicalRecordRow).where(MedicalRecordRow.id == "MR-002"))
        row.clinical_dimension = None
        session.flush()

        profile, trace = run_history(session, DEMO_ID)
        assert profile is None
        assert trace.status.value == "error"
        assert "ValueError" in (trace.error or "")

    def test_the_session_is_usable_after_a_failure(self, session):
        """The error path rolls back, so the caller can continue rather than
        inheriting a poisoned transaction."""
        session.scalar(select(MedicalRecordRow).where(MedicalRecordRow.id == "MR-002")).clinical_dimension = None
        session.flush()
        _, failed = run_history(session, DEMO_ID)
        assert failed.status.value == "error"

        ok_profile, ok_trace = run_history(session, DEMO_ID)
        assert ok_trace.status.value == "ok"
        assert ok_profile is not None

    def test_an_earlier_reference_date_excludes_later_records(self, session):
        """Information leakage guard. Phase 10 replays the patient at an earlier
        date to ask what the system would have said then; a finding dated after
        that anchor must not feed the multiplier or the baseline."""
        profile, trace = run_history(session, DEMO_ID, reference_date=date(2025, 4, 1))
        assert trace.status.value == "ok"
        assert profile is not None
        assert profile.age == 54

        # MR-004 (2025-06-20 fibrosis) is in the future relative to this anchor,
        # so it must not appear in the set the Coordinator diffs against.
        assert {f.label for f in profile.previous_findings} == {"focal_opacity"}

        # The multiplier does not move, because MR-002 still evidences a prior
        # pulmonary abnormality. Asserted explicitly: the fix is observable in
        # the comparison set, not necessarily in the score.
        assert any(f.rule_id == "R-HIS-PRIOR-PULM-01" for f in profile.risk_factors)
        assert profile.history_multiplier == pytest.approx(1.3698, abs=1e-4)

        # The comparison anchor moves back to the only study then on file.
        record = load_record(session, DEMO_ID, date(2025, 4, 1))
        assert record.last_prior_imaging_date == date(2025, 3, 12)

    def test_an_anchor_before_any_record_yields_an_empty_clinical_history(self, session):
        profile, trace = run_history(session, DEMO_ID, reference_date=date(2020, 1, 1))
        assert trace.status.value == "ok"
        assert profile is not None
        assert profile.age == 49
        assert profile.previous_findings == []
        assert profile.chronic_conditions == []
        assert profile.baseline_vulnerabilities == []
        assert any("no personal baseline" in g for g in profile.data_gaps)
        assert any("No prior chest imaging" in g for g in profile.data_gaps)

        # Not neutral: social history rows carry no event date, so an undated
        # risk factor has no known validity period and still applies. Age 49
        # falls below the middle-age band, leaving only lifestyle weights — and
        # of those, 8 pack-years is considered but deliberately unweighted.
        assert profile.history_multiplier == pytest.approx(1.04, abs=1e-4)
        assert [f.rule_id for f in profile.risk_factors] == ["R-HIS-SMOKE-03", "R-HIS-BMI-02"]
        assert [f.weight for f in profile.risk_factors] == [
            HISTORY_MULTIPLIER_FLOOR,
            WEIGHT_BMI_OVERWEIGHT.value,
        ]

    def test_repeated_runs_produce_identical_profiles(self, session):
        first, _ = run_history(session, DEMO_ID)
        session.commit()
        second, _ = run_history(session, DEMO_ID)
        assert first is not None and second is not None
        assert first.model_dump(mode="json") == second.model_dump(mode="json")


class TestSafetyGuardEnforcement:
    def test_a_unsafe_summary_is_reported_as_an_agent_error(self, session, monkeypatch):
        """The guard is structural, not advisory: if generated prose ever
        asserted a diagnosis, the agent must fail rather than publish it."""
        import app.agents.history as history_module

        monkeypatch.setattr(history_module, "build_summary", lambda *a, **k: "The diagnosis is pneumonia.")
        profile, trace = run_history(session, DEMO_ID)
        assert profile is None
        assert trace.status.value == "error"
        assert "asserts_diagnosis" in (trace.error or "")

    def test_the_disclaimer_is_not_itself_flagged(self):
        """Documents the trap: the disclaimer says 'does not replace a qualified
        clinician', which only survives because the pattern matches 'doctor'."""
        assert audit_prose({"disclaimer": DISCLAIMER}) == []
