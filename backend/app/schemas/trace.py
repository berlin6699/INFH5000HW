"""Explainability primitives shared by every agent.

`FiredRule` is the atom of this system's auditability. Risk levels are not
asserted, they are accumulated: each rule that fires records what it matched,
how much it contributed, and the evidence behind it. The UI renders these
directly, so "why did the AI decide this?" resolves to a concrete list rather
than a paragraph of prose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from app.schemas.common import AgentName, AgentStatus, Schema


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FiredRule(Schema):
    rule_id: str = Field(
        description="Stable identifier, e.g. R-MON-SPO2-ABS-01. Used as the key "
        "in reasoning/rules.py and as the UI anchor for this rule's definition."
    )
    category: str = Field(
        description="Grouping for display: monitoring | triage | imaging | "
        "history | longitudinal | override"
    )
    description: str = Field(description="Human-readable statement of what matched.")
    contribution: float = Field(
        default=0.0,
        description="Signed weight added to the risk score. Zero for rules that "
        "inform without scoring (e.g. an override, which sets the level directly).",
    )
    evidence: str = Field(
        default="",
        description="The observed values behind the match, with units and source, "
        "e.g. 'SpO2=91% on day 5; acute hypoxaemia threshold 92% (BTS)'.",
    )
    value: float | None = Field(
        default=None, description="The measured value that triggered the rule, if numeric."
    )
    threshold: float | None = Field(
        default=None, description="The threshold it was compared against, if any."
    )


class AgentTrace(Schema):
    """Uniform envelope returned by every agent and persisted per run.

    `output` is loosely typed on purpose: this envelope is what gets stored in
    `agent_results`, streamed over SSE, and rendered as an execution timeline,
    and those consumers care about ordering, timing and status rather than the
    payload shape. Typed access to each agent's result lives on
    `AnalysisResult`, which holds the concrete models.
    """

    agent_name: AgentName
    status: AgentStatus
    output: dict[str, Any] = Field(default_factory=dict)
    fired_rules: list[FiredRule] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    llm_used: bool = False
    duration_ms: int = 0
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
