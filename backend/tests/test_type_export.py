"""Type-export pipeline tests.

`make types` turns the pydantic contracts into `frontend/src/types/generated.ts`.
The two transforms that make the output usable (`_strip_leaf_titles`,
`_simplify_refs`) exist because json-schema-to-typescript mishandles valid
JSON Schema 2020-12; without them the generator emitted 259 declarations
instead of 47 and suffixed duplicates (`Acuity1`, `AnalysisResult1`). These
tests pin the invariants the generator depends on, so a future pydantic upgrade
that changes its schema output fails here rather than silently producing a
frontend full of duplicate types.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.tools import export_types
from app.tools.export_types import (
    ENUMS,
    MODELS,
    REF_TEMPLATE,
    _simplify_refs,
    _strip_leaf_titles,
    build_schema,
    main,
)

SUFFIXED = re.compile(r"^.+\d$")


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return build_schema()


@pytest.fixture(scope="module")
def defs(schema: dict[str, Any]) -> dict[str, Any]:
    return schema["$defs"]


def _walk(node: Any) -> Iterator[tuple[Any, Any]]:
    """Yield (parent, child) for every dict/list member in the tree."""
    if isinstance(node, dict):
        for value in node.values():
            yield node, value
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield node, value
            yield from _walk(value)


def test_document_shape(schema: dict[str, Any]) -> None:
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    # Title-only root. A root $ref made the generator emit a second copy of the
    # referenced model; no title made it invent an empty `Schema` interface that
    # collided with this project's pydantic base class.
    assert schema["title"] == "MedAISchemas"
    assert "$ref" not in schema
    assert set(schema) == {"$schema", "title", "$defs"}


def test_definitions_are_sorted(defs: dict[str, Any]) -> None:
    assert list(defs) == sorted(defs)


def test_every_declared_model_and_enum_is_exported(defs: dict[str, Any]) -> None:
    expected = {m.__name__ for m in MODELS} | {e.__name__ for e in ENUMS}
    assert expected <= set(defs), f"missing: {expected - set(defs)}"


def test_no_numeric_suffix_duplicates(defs: dict[str, Any]) -> None:
    """The visible symptom of the generator inlining duplicated subschemas."""
    offenders = [name for name in defs if SUFFIXED.match(name)]
    assert offenders == [], f"suspicious names: {offenders}"


def test_analysis_result_emitted_exactly_once(defs: dict[str, Any]) -> None:
    assert [n for n in defs if n.startswith("AnalysisResult")] == ["AnalysisResult"]


def test_no_ref_has_siblings(schema: dict[str, Any]) -> None:
    """json-schema-to-typescript inlines `$ref` when siblings are present.

    Pydantic emits `{"$ref": ..., "default": ...}` for enum fields that carry a
    default, which is legal JSON Schema but produces `Acuity1`/`Acuity2` in the
    generated TypeScript.
    """
    offenders = [
        node for _, node in _walk(schema)
        if isinstance(node, dict) and "$ref" in node and set(node) != {"$ref"}
    ]
    assert offenders == []


def test_title_appears_only_at_definition_roots(schema: dict[str, Any]) -> None:
    """A titled subschema becomes its own exported declaration.

    Pydantic derives a `title` for every property from its field name, so
    leaving them in exploded the definition count from 47 to 259.
    """
    for name, definition in schema["$defs"].items():
        assert definition.get("title") == name
        inner = {k: v for k, v in definition.items() if k != "title"}
        stray = [node for _, node in _walk(inner)
                 if isinstance(node, dict) and "title" in node]
        assert stray == [], f"{name} still carries nested titles"


def test_descriptions_survive_stripping(defs: dict[str, Any]) -> None:
    """Field docstrings become JSDoc in generated.ts; they are worth keeping."""
    assert "description" in defs["AnalysisResult"]
    props = defs["TrendFeature"]["properties"]
    assert "description" in props["anomaly_score"]


def test_all_refs_resolve(defs: dict[str, Any]) -> None:
    for _, node in _walk(defs):
        if isinstance(node, dict) and "$ref" in node:
            target = node["$ref"]
            assert target.startswith("#/$defs/"), target
            assert target[len("#/$defs/"):] in defs, f"dangling {target}"


def test_ref_template_matches_the_resolver() -> None:
    assert REF_TEMPLATE == "#/$defs/{model}"


def test_enums_export_their_python_members(defs: dict[str, Any]) -> None:
    """The frontend hardcodes enum values for colour mapping.

    If a member is renamed in Python and the export drops it, the dashboard's
    risk badge silently renders with no colour instead of failing to compile.
    """
    for enum in ENUMS:
        definition = defs[enum.__name__]
        assert definition.get("enum") == [m.value for m in enum], enum.__name__


def test_risk_levels_are_exported_verbatim(defs: dict[str, Any]) -> None:
    """Uppercase by design — these are display values, not internal codes."""
    assert defs["RiskLevel"]["enum"] == ["LOW", "MEDIUM", "HIGH"]


def test_transforms_are_idempotent(schema: dict[str, Any]) -> None:
    """Re-running either transform must be a no-op on already-cleaned output."""
    once = _simplify_refs(_strip_leaf_titles(schema))
    twice = _simplify_refs(_strip_leaf_titles(once))
    assert once == twice


def test_strip_leaf_titles_keeps_descriptions() -> None:
    node = {"title": "Risk Level", "description": "keep me", "enum": ["low"]}
    assert _strip_leaf_titles(node) == {"description": "keep me", "enum": ["low"]}


def test_simplify_refs_drops_only_siblings() -> None:
    node = {"properties": {"level": {"$ref": "#/$defs/RiskLevel", "default": "low"}}}
    assert _simplify_refs(node) == {"properties": {"level": {"$ref": "#/$defs/RiskLevel"}}}


def test_main_writes_valid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`main` is what `make types` invokes; write it to a temp path, not the tree."""
    out = tmp_path / "schema.json"
    monkeypatch.setattr(export_types, "OUTPUT_PATH", out)
    assert main() == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == build_schema()
    assert "$defs" in written

    log = capsys.readouterr().out
    assert str(len(written["$defs"])) in log
    assert "json-schema-to-typescript" in log


def test_exported_names_cover_the_demo_scenario(defs: dict[str, Any]) -> None:
    """The types the acceptance demo needs must all be reachable."""
    required = {
        "AnalysisResult", "FinalAssessment", "HistoricalChange", "PatientProfile",
        "PriorFinding", "SymptomAssessment", "ImagingFinding", "Abnormality",
        "MonitoringAssessment", "TrendFeature", "KnowledgeResult",
        "RetrievedEvidence", "AgentTrace", "FiredRule", "Department",
        "RiskLevel", "ChangeType", "ClinicalDimension",
    }
    assert required <= set(defs), f"missing: {required - set(defs)}"
