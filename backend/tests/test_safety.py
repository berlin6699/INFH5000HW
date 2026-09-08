"""Safety guard tests.

The guard's job is to block diagnostic and prescriptive wording without
censoring correct clinical observations. Both halves are tested: an audit that
only ever fires is as useless as one that never fires, because it would force
the Phase 6 summary templates into evasive nonsense.
"""

from __future__ import annotations

import pytest

from app.safety import (
    DISCLAIMER,
    SafetyGuardError,
    assert_prose_is_safe,
    audit_prose,
)

# Prose the coordinator is legitimately expected to produce.
CLEAN_SAMPLES = [
    "The patient has a fever of 38.7 C and a productive cough.",
    "You have elevated blood pressure relative to your recorded baseline.",
    "SpO2 has fallen from 98% to 91% over five days, a worsening trend.",
    "Findings are consistent with an acute lower respiratory process.",
    "Community-acquired pneumonia should be considered in the differential.",
    "Compared with the 2025-03-12 chest X-ray, the opacity has increased.",
    "Recommend same-day evaluation by a clinician.",
    "Risk level HIGH; suggested department: Emergency Department.",
    "Pneumonia, sepsis and pulmonary embolism remain in the differential.",
]

# Prose that crosses into diagnosis or prescription.
BLOCKED_SAMPLES = [
    ("asserts_diagnosis", "The diagnosis is community-acquired pneumonia."),
    ("asserts_diagnosis", "You have pneumonia and require admission."),
    ("asserts_diagnosis", "The patient has tuberculosis."),
    ("asserts_diagnosis", "She was diagnosed with asthma in 2019."),
    ("asserts_diagnosis", "This is a confirmed diagnosis of sepsis."),
    ("asserts_diagnosis_zh", "该患者确诊为肺炎。"),
    ("asserts_diagnosis_zh", "诊断为社区获得性肺炎。"),
    ("asserts_diagnosis_zh", "患者患有肺癌。"),
    ("prescribes_treatment", "We prescribe amoxicillin 500 mg three times daily."),
    ("prescribes_treatment", "Take 20 mg of prednisone each morning."),
    ("prescribes_treatment", "You should increase your dose of amlodipine."),
    ("prescribes_treatment", "Stop taking your anticoagulant before imaging."),
    ("prescribes_treatment_zh", "建议服用阿莫西林。"),
    ("prescribes_treatment_zh", "可加大剂量至每日两次。"),
    ("claims_replacement", "This system replaces your doctor."),
    ("claims_replacement", "There is no need to see a doctor about this."),
    ("claims_replacement", "The result is guaranteed accurate."),
    ("claims_replacement", "Our triage is 100% accurate."),
]


@pytest.mark.parametrize("text", CLEAN_SAMPLES)
def test_clean_clinical_prose_passes(text: str) -> None:
    assert audit_prose({"summary": text}) == []


@pytest.mark.parametrize("category,text", BLOCKED_SAMPLES)
def test_blocked_prose_is_caught(category: str, text: str) -> None:
    violations = audit_prose({"summary": text})
    assert violations, f"expected {text!r} to be blocked"
    assert any(v.category == category for v in violations)


def test_possessive_frame_needs_a_named_disease() -> None:
    """Regression test for an over-broad pattern.

    `you have` used to match anything that was not literally "a fever", so
    "you have elevated blood pressure" — a correct, in-scope observation — was
    blocked. The verb alone must not be enough.
    """
    assert audit_prose({"summary": "You have elevated inflammatory markers."}) == []
    assert audit_prose({"summary": "The patient has a chronic cough."}) == []
    assert audit_prose({"summary": "You have pneumonia."}) != []
    assert audit_prose({"summary": "The patient has pneumonia."}) != []


def test_naming_a_disease_is_not_diagnosing() -> None:
    """A differential may name conditions; asserting one may not."""
    assert audit_prose({"summary": "pneumonia"}) == []
    assert audit_prose({"summary": "Consider pneumonia or pulmonary embolism."}) == []


def test_mentioning_a_drug_is_not_prescribing() -> None:
    """Guideline-sourced medication context is allowed; directives are not."""
    assert audit_prose({"summary": "Amlodipine is recorded as a current medication."}) == []
    assert audit_prose({"summary": "Antibiotic therapy is a clinical decision."}) == []


def test_violation_records_field_and_match() -> None:
    violations = audit_prose({
        "summary": "The diagnosis is pneumonia.",
        "recommendation": "Recommend urgent review.",
    })
    assert len(violations) == 1
    assert violations[0].field == "summary"
    assert violations[0].matched.lower().startswith("diagnosis")


def test_audit_scans_every_item_in_a_list() -> None:
    violations = audit_prose({
        "key_evidence": [
            "SpO2 fell 7 points in 5 days.",
            "The diagnosis is pneumonia.",
        ],
    })
    assert len(violations) == 1


def test_audit_tolerates_none_and_non_string_entries() -> None:
    # Prose fields on the assessment models are optional; a None must not crash
    # the guard, and the guard must not silently stringify a non-prose value.
    assert audit_prose({"summary": None, "notes": [None, "clean text"]}) == []


def test_assert_prose_is_safe_raises_with_detail() -> None:
    with pytest.raises(SafetyGuardError) as exc:
        assert_prose_is_safe({
            "summary": "The diagnosis is pneumonia.",
            "advice": "Take 20 mg of prednisone.",
        })
    message = str(exc.value)
    # The message must say which field and which rule fired, or debugging a
    # blocked run means reading regexes.
    assert "summary" in message
    assert "asserts_diagnosis" in message
    assert "advice" in message
    assert "prescribes_treatment" in message


def test_assert_prose_is_safe_passes_clean_output() -> None:
    assert_prose_is_safe({"summary": CLEAN_SAMPLES[0], "evidence": CLEAN_SAMPLES[2]})


def test_disclaimer_itself_is_clean() -> None:
    """The banner is rendered on every screen; it must not trip its own guard.

    This is a real trap: "does not replace a qualified clinician" is safe only
    because the replacement pattern matches `doctor`. Rewording the disclaimer
    to "does not replace your doctor" would flag it.
    """
    assert audit_prose({"disclaimer": DISCLAIMER}) == []


def test_evidence_text_would_false_positive_so_callers_exclude_it() -> None:
    """Pins the reason `RetrievedEvidence.text` is never passed to the guard.

    Guideline source material legitimately contains assertive clinical framing.
    Nothing structural stops a caller from auditing it, so the failure mode is
    documented here as an executable example.
    """
    guideline_quote = "In adults the patient has pneumonia when consolidation is lobar."
    assert audit_prose({"evidence": guideline_quote}) != []
    # Prose *about* the citation stays clean.
    assert audit_prose({"summary": "Guideline excerpt retrieved from NICE NG138."}) == []


def test_every_violation_category_is_reachable() -> None:
    """No dead regex in the table.

    Each category must be exercised by at least one blocked sample, otherwise
    the pattern has drifted from the vocabulary it was meant to cover.
    """
    categories = {category for category, _ in BLOCKED_SAMPLES}
    fired = {v.category for _, text in BLOCKED_SAMPLES for v in audit_prose({"s": text})}
    assert fired == categories
