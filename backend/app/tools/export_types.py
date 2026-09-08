"""Export the pydantic contracts as one JSON Schema document.

Consumed by `make types`, which runs json-schema-to-typescript over the output
to produce `frontend/src/types/generated.ts`.

This exists to kill a specific class of bug: a backend field renamed or
retyped while the frontend keeps using the old shape. With generation, the
contract has exactly one source of truth (the pydantic models) and the
TypeScript side is rebuilt from it rather than maintained by hand.

Run directly with:  python -m app.tools.export_types
"""

from __future__ import annotations

import json
import sys
from typing import Any

from pydantic import BaseModel, TypeAdapter

from app.config import PROJECT_ROOT
from app.schemas import (
    Abnormality,
    Acuity,
    AgentName,
    AgentStatus,
    AgentTrace,
    Allergy,
    AnalysisRequest,
    AnalysisResult,
    ChangeType,
    ClinicalDimension,
    ClinicalDomain,
    Condition,
    Department,
    FinalAssessment,
    FiredRule,
    HistoricalChange,
    ImagingFinding,
    ImagingLabel,
    ImagingModality,
    ImagingSourceMode,
    KnowledgeResult,
    Medication,
    MonitoringAssessment,
    Patient,
    PatientProfile,
    PriorFinding,
    RetrievedEvidence,
    RetrieverKind,
    RiskFactor,
    RiskLevel,
    RunConfig,
    RunStatus,
    Severity,
    Significance,
    SocialHistory,
    Symptom,
    SymptomAssessment,
    SymptomInput,
    SymptomName,
    TimelineEvent,
    TimelineEventType,
    TrendDirection,
    TrendFeature,
    Urgency,
    VitalMetric,
    VitalSample,
    WarningSign,
    WarningSignSeverity,
)

REF_TEMPLATE = "#/$defs/{model}"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "src" / "types" / "schema.json"

MODELS: list[type[BaseModel]] = [
    # run-level
    AnalysisResult,
    AnalysisRequest,
    RunConfig,
    AgentTrace,
    FiredRule,
    # coordinator
    FinalAssessment,
    HistoricalChange,
    # history
    PatientProfile,
    Patient,
    Condition,
    Medication,
    Allergy,
    SocialHistory,
    PriorFinding,
    RiskFactor,
    TimelineEvent,
    # triage
    SymptomAssessment,
    SymptomInput,
    Symptom,
    WarningSign,
    # imaging
    ImagingFinding,
    Abnormality,
    # monitoring
    MonitoringAssessment,
    TrendFeature,
    VitalSample,
    # knowledge
    KnowledgeResult,
    RetrievedEvidence,
]

ENUMS: list[type] = [
    Acuity,
    AgentName,
    AgentStatus,
    ChangeType,
    ClinicalDimension,
    ClinicalDomain,
    Department,
    ImagingLabel,
    ImagingModality,
    ImagingSourceMode,
    RetrieverKind,
    RiskLevel,
    RunStatus,
    Severity,
    Significance,
    SymptomName,
    TimelineEventType,
    TrendDirection,
    Urgency,
    VitalMetric,
    WarningSignSeverity,
]


def _strip_leaf_titles(node: Any) -> Any:
    """Remove `title` from a schema tree.

    Pydantic derives a `title` for every property from its field name.
    json-schema-to-typescript promotes any titled subschema into its own
    exported declaration, so those property titles turned 47 definitions into
    259 top-level types and forced numeric suffixes (`Acuity1`, `PatientId3`)
    to disambiguate them. A title is only meaningful at the root of a `$def`,
    where it names the generated interface, so it is stripped everywhere and
    restored there by `build_schema`. `description` is kept: it becomes JSDoc.
    """
    if isinstance(node, dict):
        return {k: _strip_leaf_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_leaf_titles(v) for v in node]
    return node


def _simplify_refs(node: Any) -> Any:
    """Reduce any schema containing `$ref` to the `$ref` alone.

    Pydantic emits `{"$ref": "#/$defs/Acuity", "default": "indeterminate"}` for
    enum fields that carry a default. JSON Schema 2020-12 allows `$ref`
    siblings, but json-schema-to-typescript does not handle them: it inlines a
    copy of the referenced definition, producing duplicate `Acuity1`/`Acuity2`
    declarations with scrambled doc comments.

    Dropping the siblings costs nothing here. `default` is a pydantic runtime
    concern with no meaning in a TypeScript type, and the enum-typed fields in
    these schemas carry no descriptions worth preserving.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            return {"$ref": node["$ref"]}
        return {k: _simplify_refs(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_simplify_refs(v) for v in node]
    return node


def build_schema() -> dict[str, Any]:
    defs: dict[str, Any] = {}

    def add(name: str, schema: dict[str, Any]) -> None:
        cleaned = _simplify_refs(_strip_leaf_titles(schema))
        cleaned["title"] = name
        defs[name] = cleaned

    for model in MODELS:
        schema = model.model_json_schema(ref_template=REF_TEMPLATE)
        # Nested dependencies come back in the model's own $defs; hoist them so
        # every reference resolves against one flat namespace.
        for name, definition in schema.pop("$defs", {}).items():
            if name not in defs:
                add(name, definition)
        add(model.__name__, schema)

    for enum in ENUMS:
        schema = TypeAdapter(enum).json_schema(ref_template=REF_TEMPLATE)
        for name, definition in schema.pop("$defs", {}).items():
            if name not in defs:
                add(name, definition)
        if enum.__name__ not in defs:
            add(enum.__name__, schema)

    # A root title but deliberately NO root $ref. The $ref made the generator
    # emit a second copy of the referenced model (AnalysisResult1); with no title
    # at all it invented an empty root interface named `Schema`, colliding with
    # this project's pydantic base class. Title-only gives one empty root
    # interface and every definition exactly once. --unreachableDefinitions does
    # the actual work of emitting the $defs.
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "MedAISchemas",
        "$defs": dict(sorted(defs.items())),
    }


def main() -> int:
    schema = build_schema()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Cosmetic, and the file is already written: don't let a path outside the
    # project root (a temp dir, a CI workspace) fail the export.
    try:
        shown = OUTPUT_PATH.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = OUTPUT_PATH

    print(f"Wrote {shown}")
    print(f"  {len(MODELS)} models + {len(ENUMS)} enums -> {len(schema['$defs'])} definitions")
    print("  Next: json-schema-to-typescript (make types runs both steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
