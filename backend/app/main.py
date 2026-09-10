"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.agents.history import run_history
from app.db.models import MedicalImageRow, MonitoringRecordRow, PatientRow
from app.db.seed import load_fixture, seed
from app.db.session import get_session, init_db
from app.mock_pipeline import run_mock_analysis
from app.safety import DISCLAIMER
from app.schemas import AnalysisRequest, AnalysisResult, MOCK_BADGE_TEXT, REPORT_BADGE_TEXT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("medai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
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


def _ensure_demo_patient(session: Session) -> PatientRow:
    patient = session.get(PatientRow, "PT-DEMO-001")
    if patient is None:
        seed(session)
        session.flush()
        patient = session.get(PatientRow, "PT-DEMO-001")
    if patient is None:  # pragma: no cover - defensive invariant
        raise HTTPException(status_code=500, detail="Demo patient could not be initialized")
    return patient


@app.get("/api/demo")
def demo_data(session: Session = Depends(get_session)) -> dict[str, object]:
    """Everything required to render the single-patient offline demo."""
    patient = _ensure_demo_patient(session)
    profile, trace = run_history(session, patient.id)
    if profile is None:
        raise HTTPException(status_code=500, detail=trace.error or "History Agent failed")
    monitoring = session.scalars(
        select(MonitoringRecordRow)
        .where(MonitoringRecordRow.patient_id == patient.id)
        .order_by(MonitoringRecordRow.recorded_at)
    ).all()
    image = session.scalars(
        select(MedicalImageRow)
        .where(MedicalImageRow.patient_id == patient.id)
        .order_by(MedicalImageRow.study_date.desc())
    ).first()
    fixture = load_fixture()
    return {
        "patient": profile.model_dump(mode="json"),
        "symptoms": fixture["current_symptoms"],
        "monitoring": [
            {
                "recorded_at": row.recorded_at.isoformat(),
                "spo2": row.spo2,
                "heart_rate": row.heart_rate,
                "temperature_c": row.temperature_c,
                "respiratory_rate": row.respiratory_rate,
                "sleep_hours": row.sleep_hours,
                "activity_steps": row.activity_steps,
            }
            for row in monitoring
        ],
        "imaging": image.findings if image and image.findings else None,
    }


@app.post("/api/analysis/run", response_model=AnalysisResult)
def analyse(request: AnalysisRequest, session: Session = Depends(get_session)) -> AnalysisResult:
    _ensure_demo_patient(session)
    if request.patient_id != "PT-DEMO-001":
        raise HTTPException(status_code=404, detail="The first MVP supports the demo patient only")
    try:
        return run_mock_analysis(session, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# In the one-port production-style demo FastAPI serves the Vite build itself.
# API routes are registered first so the catch-all static mount cannot shadow them.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
