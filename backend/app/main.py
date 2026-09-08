"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.safety import DISCLAIMER
from app.schemas import MOCK_BADGE_TEXT, REPORT_BADGE_TEXT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("medai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("LLM provider=%s (llm_enabled=%s)", settings.llm_provider, settings.llm_enabled)
    log.info("Imaging mode=%s | RAG retriever=%s", settings.imaging_mode, settings.rag_retriever)
    if not settings.llm_enabled:
        log.info("Running in deterministic offline mode: all structured fields are "
                 "computed by the rule engine, prose by templates.")
    yield


app = FastAPI(
    title="Multimodal Multi-Agent Healthcare Assistant",
    description=(
        "Educational research prototype for intelligent triage and continuous "
        "health monitoring. Not a medical device; output is not a diagnosis."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/system/info")
def system_info() -> dict[str, object]:
    """Capability manifest the frontend uses to render banners and badges.

    The UI must be able to show, at all times, whether the LLM layer is live
    and whether imaging output is real or mocked. Deriving that from the
    backend keeps the badge honest instead of hardcoded in JSX.
    """
    s = get_settings()
    return {
        "llm": {
            "provider": s.llm_provider,
            "enabled": s.llm_enabled,
            "model": (
                s.openai_model if s.llm_provider == "openai"
                else s.deepseek_model if s.llm_provider == "deepseek"
                else None
            ),
            "note": (
                "Deterministic offline mode. Structured fields are rule-computed; "
                "prose is template-generated."
                if not s.llm_enabled
                else "LLM active for prose synthesis only. Structured fields "
                     "remain rule-computed."
            ),
        },
        "imaging": {
            "mode": s.imaging_mode,
            # No CXR model is integrated in this project; see README "Imaging
            # output provenance". Kept as a field so the UI never has to infer it.
            "is_real_model": False,
            "badge": (
                MOCK_BADGE_TEXT if s.imaging_mode == "mock_preset" else REPORT_BADGE_TEXT
            ),
            "supported_modalities": ["chest_xray"],
        },
        "rag": {"retriever": s.rag_retriever, "top_k": s.rag_top_k},
        "safety": {
            "disclaimer": DISCLAIMER,
            "guard_enforced": s.enforce_safety_guard,
            "produces_diagnosis": False,
        },
    }


# Routers are attached in later phases:
#   Phase 1b -> patients, monitoring, imaging
#   Phase 6  -> analysis
#   Phase 5  -> knowledge
