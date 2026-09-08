"""Knowledge Agent contracts: retrieval-augmented evidence.

Other agents are not permitted to reason from LLM parametric memory alone. Any
clinical threshold or guideline claim that reaches the final assessment should
arrive through this module carrying its source, so the UI can show the citation
rather than an unsourced assertion.

`coverage_gaps` is as important as `evidence`. Recording what could not be
found prevents the system from looking uniformly confident across questions it
can and cannot support.
"""

from __future__ import annotations

from pydantic import Field

from app.schemas.common import RetrieverKind, Schema


class RetrievedEvidence(Schema):
    chunk_id: str
    text: str = Field(description="Verbatim passage from the corpus. Never rewritten.")
    source: str = Field(description="Document title, e.g. 'BTS Community-Acquired Pneumonia Guideline'.")
    guideline_ref: str | None = Field(
        default=None, description="Section, table or page locator within the source."
    )
    url: str | None = None
    relevance_score: float = Field(
        default=0.0, ge=0.0, description="Retriever score, normalised to 0-1 where supported."
    )
    retriever: RetrieverKind = RetrieverKind.BM25

    query: str = Field(default="", description="The query string that retrieved this chunk.")
    retrieved_for: str = Field(
        default="",
        description="The upstream finding that motivated the query, e.g. "
        "'monitoring: spo2 worsening' or 'triage: dyspnea'. Links a citation "
        "back to the observation it supports.",
    )
    topic: str | None = Field(default=None, description="Corpus-assigned topic tag.")


class KnowledgeResult(Schema):
    """Knowledge Agent output."""

    queries: list[str] = Field(
        default_factory=list,
        description="Every query issued. Generated programmatically from the "
        "structured output of the other agents, not from free text.",
    )
    evidence: list[RetrievedEvidence] = Field(default_factory=list)

    coverage_gaps: list[str] = Field(
        default_factory=list,
        description="Questions asked that the corpus could not support.",
    )
    deduplicated_from: int = Field(
        default=0, description="Count of duplicate chunks removed across queries."
    )

    retriever: RetrieverKind = RetrieverKind.BM25
    corpus_size: int = 0
    top_k: int = 0
    summary: str = ""
