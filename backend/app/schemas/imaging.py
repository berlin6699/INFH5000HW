"""Imaging Agent contracts. Chest X-ray only for the MVP.

Two invariants are enforced here rather than left to convention:

1. `source_mode` is required. An imaging result with unknown provenance cannot
   be constructed, so mock output can never be mistaken for model output.
2. Any result whose `source_mode` is not `real_model` is guaranteed to carry a
   disclosure badge. The validator fills in mode-appropriate wording when the
   caller omits it, so a predefined finding cannot reach the UI unlabelled even
   if someone forgets to pass the constant.

`Abnormality.severity_score` uses the same 0-5 ordinal scale as
`PriorFinding.severity_score`. That shared scale is what allows the
LongitudinalComparator to diff a 2025 report against a 2026 study numerically.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import Field, model_validator

from app.schemas.common import Acuity, ImagingModality, ImagingSourceMode, Schema

MOCK_BADGE_TEXT = "DEMO / MOCK OUTPUT — not produced by a real medical imaging model"
REPORT_BADGE_TEXT = (
    "PARSED FROM USER-SUPPLIED RADIOLOGY REPORT — no imaging model was run by this system"
)


class ImagingLabel(StrEnum):
    """Closed vocabulary of chest X-ray findings.

    Exact labels (not free text) are required so that a prior study and a
    current study can be matched on the same axis. Free-text descriptions live
    in `Abnormality.description`.
    """

    NORMAL = "normal"
    FOCAL_OPACITY = "focal_opacity"
    MULTIFOCAL_OPACITIES = "multifocal_opacities"
    CONSOLIDATION = "consolidation"
    GROUND_GLASS_OPACITY = "ground_glass_opacity"
    PLEURAL_EFFUSION = "pleural_effusion"
    PNEUMOTHORAX = "pneumothorax"
    ATELECTASIS = "atelectasis"
    PULMONARY_EDEMA = "pulmonary_edema"
    CARDIOMEGALY = "cardiomegaly"
    HYPERINFLATION = "hyperinflation"
    FIBROSIS = "fibrosis"
    NODULE = "nodule"
    MASS = "mass"


class Abnormality(Schema):
    label: ImagingLabel
    description: str | None = None
    location: str | None = Field(default=None, description="Anatomical site, e.g. 'RLL', 'bilateral'.")
    acuity: Acuity = Acuity.INDETERMINATE
    severity_score: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Same ordinal scale as PriorFinding.severity_score, so that "
        "findings years apart can be compared by subtraction.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    matched_prior_label: ImagingLabel | None = Field(
        default=None,
        description="Set by the LongitudinalComparator when the same label "
        "appears in a prior study.",
    )


class ImagingFinding(Schema):
    """Imaging Agent output."""

    image_id: str | None = None
    modality: ImagingModality = ImagingModality.CHEST_XRAY
    study_date: date | None = None

    source_mode: ImagingSourceMode
    provenance: str = Field(
        description="Plain statement of where this result came from, e.g. "
        "'Preset finding for synthetic patient PT-DEMO-001' or "
        "'Parsed from radiology report uploaded 2026-09-08'."
    )
    badge: str | None = Field(
        default=None,
        description="Disclosure text the UI must render. Auto-filled for any "
        "source_mode other than real_model; always None for real_model.",
    )

    findings: list[str] = Field(default_factory=list, description="Observations, including normal ones.")
    abnormalities: list[Abnormality] = Field(default_factory=list)
    confidence: dict[str, float] = Field(
        default_factory=dict, description="Per-label confidence, mirrored from `abnormalities`."
    )

    comparable_prior_study_id: str | None = Field(
        default=None,
        description="Prior study selected as the comparison anchor. Null means "
        "no longitudinal imaging comparison was possible, which is itself "
        "reported rather than silently skipped.",
    )
    prior_comparison_note: str | None = None

    has_acute_abnormality: bool = False
    summary: str = ""

    @model_validator(mode="after")
    def _enforce_provenance_labelling(self) -> ImagingFinding:
        """Guarantee that non-real imaging output always carries a disclosure.

        Enforced by construction rather than by convention: a caller may
        override the wording, but cannot omit it. The failure mode this prevents
        is a predefined finding reaching the UI with no indication that no model
        produced it.
        """
        if self.source_mode == ImagingSourceMode.REAL_MODEL:
            self.badge = None
            return self
        if (self.badge or "").strip():
            return self
        self.badge = (
            MOCK_BADGE_TEXT
            if self.source_mode == ImagingSourceMode.MOCK_PRESET
            else REPORT_BADGE_TEXT
        )
        return self

    @property
    def is_real_model(self) -> bool:
        return self.source_mode == ImagingSourceMode.REAL_MODEL
