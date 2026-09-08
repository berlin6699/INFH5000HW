"""Safety and scope guardrails.

This system is an educational research prototype for triage decision-support.
It is not a medical device and never produces a diagnosis. The guardrails here
are enforced in code rather than left to convention, because a prototype that
*can* emit diagnostic wording will eventually emit diagnostic wording.

Two distinct protections:

1. STRUCTURAL -- there is no `diagnosis` field anywhere in the schemas. The
   vocabulary itself is limited to risk level, urgency, and recommended
   department.

2. LEXICAL -- `audit_prose()` scans the prose fields this system generates and
   rejects wording that asserts a diagnosis or prescribes treatment.

Only generated prose is audited. `RetrievedEvidence.text` is quoted source
material from clinical guidelines and legitimately contains disease names;
scanning it would produce false positives and would mean censoring citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DISCLAIMER = (
    "Educational research prototype for triage decision-support. "
    "Output is NOT a medical diagnosis and does NOT replace a qualified "
    "clinician. Seek professional medical assessment for any health concern."
)

# Imaging disclosure wording lives in app.schemas.imaging (MOCK_BADGE_TEXT /
# REPORT_BADGE_TEXT), next to the validator that enforces it, so there is a
# single source of truth.

# Specific disease names. Used to keep the possessive patterns below narrow.
_DISEASES = (
    r"(?:pneumonia|cancer|carcinoma|tumou?r|lymphoma|tuberculosis|sepsis|"
    r"septicaemia|copd|asthma|pulmonary\s+embolism|myocardial\s+infarction|"
    r"heart\s+failure|pneumothorax)"
)

# Asserted-diagnosis and treatment-directive phrasing. Deliberately narrow:
# "you have" and "patient has" must be followed by a named disease, because
# "you have elevated blood pressure" or "the patient has a fever" are correct,
# in-scope observations. Blocking on the verb alone would censor legitimate
# findings rather than catch overreach.
_PROSE_VIOLATIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("asserts_diagnosis", re.compile(
        rf"\b(diagnosis\s+is|diagnosed\s+with|definitive\s+diagnosis|"
        rf"confirmed\s+diagnosis|(?:you|the\s+patient)\s+ha(?:s|ve)\s+"
        rf"{_DISEASES})\b", re.I)),
    ("asserts_diagnosis_zh", re.compile(
        r"(确诊|诊断为|患有(?:肺炎|癌症|肺癌|结核|肿瘤)|可以确定是)")),
    ("prescribes_treatment", re.compile(
        r"\b(prescrib(?:e|es|ing)|take\s+\d+\s*mg|increase\s+(?:your\s+)?dose|"
        r"reduce\s+(?:your\s+)?dose|stop\s+taking\s+your)\b", re.I)),
    ("prescribes_treatment_zh", re.compile(
        r"(开具处方|建议服用|加大剂量|减少剂量|停用(?:你的)?药物|服用\d+\s*毫克)")),
    ("claims_replacement", re.compile(
        r"\b(replaces?\s+(?:your\s+)?doctor|no\s+need\s+to\s+see\s+a\s+doctor|"
        r"guaranteed\s+(?:accurate|correct)|100%\s+accurate)\b", re.I)),
)


@dataclass(frozen=True)
class Violation:
    category: str
    matched: str
    field: str


def audit_prose(fields: dict[str, str | list[str] | None]) -> list[Violation]:
    """Scan generated prose for disallowed wording.

    Accepts a mapping of field name -> string or list of strings. Returns an
    empty list when the text is clean.
    """
    violations: list[Violation] = []
    for field, value in fields.items():
        if value is None:
            continue
        chunks = value if isinstance(value, list) else [value]
        for chunk in chunks:
            if not isinstance(chunk, str):
                continue
            for category, pattern in _PROSE_VIOLATIONS:
                m = pattern.search(chunk)
                if m:
                    violations.append(
                        Violation(category=category, matched=m.group(0), field=field)
                    )
    return violations


def assert_prose_is_safe(fields: dict[str, str | list[str] | None]) -> None:
    """Raise when generated prose crosses into diagnosis or prescription.

    Called on the coordinator output path when MEDAI_ENFORCE_SAFETY_GUARD is
    on. Failing loudly here is the point: a silently rewritten assessment
    would hide the bug.
    """
    violations = audit_prose(fields)
    if violations:
        detail = "; ".join(
            f"{v.field}: [{v.category}] {v.matched!r}" for v in violations
        )
        raise SafetyGuardError(f"Output blocked by safety guard — {detail}")


class SafetyGuardError(RuntimeError):
    """Raised when generated text violates the prototype's scope limits."""
