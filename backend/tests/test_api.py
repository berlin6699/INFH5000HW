"""API contract tests.

`/api/system/info` drives every honesty badge in the UI, so its shape is part
of the contract, not an implementation detail. These tests pin the keys the
frontend's hand-written `SystemInfo` interface declares, and assert the two
invariants that must never flip: imaging is not a real model, and the system
does not produce a diagnosis.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.safety import DISCLAIMER
from app.schemas import MOCK_BADGE_TEXT, REPORT_BADGE_TEXT


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient over forced-default settings.

    `get_settings` is lru_cached and `.env` is read at construction, so a
    developer's local key would otherwise leak into assertions about offline
    behaviour. The cache is cleared on the way out.
    """
    monkeypatch.setenv("MEDAI_LLM_PROVIDER", "null")
    monkeypatch.setenv("MEDAI_OPENAI_API_KEY", "")
    monkeypatch.setenv("MEDAI_DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("MEDAI_IMAGING_MODE", "mock_preset")
    monkeypatch.setenv("MEDAI_RAG_RETRIEVER", "bm25")
    monkeypatch.setenv("MEDAI_RAG_TOP_K", "4")
    monkeypatch.setenv("MEDAI_ENFORCE_SAFETY_GUARD", "true")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_system_info_top_level_keys(client: TestClient) -> None:
    body = client.get("/api/system/info").json()
    assert set(body) == {"llm", "imaging", "rag", "safety"}


@pytest.mark.parametrize(
    ("section", "keys"),
    [
        ("llm", {"provider", "enabled", "model", "note"}),
        ("imaging", {"mode", "is_real_model", "badge", "supported_modalities"}),
        ("rag", {"retriever", "top_k"}),
        ("safety", {"disclaimer", "guard_enforced", "produces_diagnosis"}),
    ],
)
def test_system_info_section_keys(client: TestClient, section: str, keys: set[str]) -> None:
    """Mirror `SystemInfo` in frontend/src/api/client.ts.

    The frontend interface is hand-written (it describes the envelope, not a
    pydantic model), so a backend key rename would compile fine on both sides
    and fail only at runtime.
    """
    body = client.get("/api/system/info").json()
    assert set(body[section]) == keys


def test_llm_reports_offline_deterministic_mode(client: TestClient) -> None:
    llm = client.get("/api/system/info").json()["llm"]
    assert llm["provider"] == "null"
    assert llm["enabled"] is False
    assert llm["model"] is None
    assert "Deterministic offline mode" in llm["note"]


def test_selected_provider_without_a_key_stays_disabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider name alone must not make the UI claim the LLM is live."""
    monkeypatch.setenv("MEDAI_LLM_PROVIDER", "openai")
    get_settings.cache_clear()
    llm = client.get("/api/system/info").json()["llm"]
    assert llm["provider"] == "openai"
    assert llm["enabled"] is False
    get_settings.cache_clear()


def test_provider_with_a_key_is_enabled_and_names_its_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEDAI_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("MEDAI_DEEPSEEK_API_KEY", "sk-test-not-a-real-key")
    get_settings.cache_clear()
    llm = client.get("/api/system/info").json()["llm"]
    assert llm["enabled"] is True
    assert llm["model"] == "deepseek-chat"
    # The note must still say structured fields are rule-computed: enabling
    # prose synthesis does not make the LLM authoritative over risk_level.
    assert "rule-computed" in llm["note"]
    get_settings.cache_clear()


def test_imaging_never_claims_a_real_model(client: TestClient) -> None:
    """Hardcoded False on purpose — no CXR model is integrated in this project."""
    imaging = client.get("/api/system/info").json()["imaging"]
    assert imaging["is_real_model"] is False
    assert imaging["supported_modalities"] == ["chest_xray"]


def test_mock_preset_mode_carries_the_mock_badge(client: TestClient) -> None:
    imaging = client.get("/api/system/info").json()["imaging"]
    assert imaging["mode"] == "mock_preset"
    assert imaging["badge"] == MOCK_BADGE_TEXT


def test_uploaded_report_mode_carries_the_report_badge(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parsing a real radiology report is still not a real imaging model.

    The badge differs so the UI does not label honest report-derived findings
    as fabricated presets, but `is_real_model` stays False in both modes.
    """
    monkeypatch.setenv("MEDAI_IMAGING_MODE", "uploaded_report")
    get_settings.cache_clear()
    imaging = client.get("/api/system/info").json()["imaging"]
    assert imaging["mode"] == "uploaded_report"
    assert imaging["badge"] == REPORT_BADGE_TEXT
    assert imaging["is_real_model"] is False
    get_settings.cache_clear()


def test_rag_settings_are_surfaced(client: TestClient) -> None:
    rag = client.get("/api/system/info").json()["rag"]
    assert rag == {"retriever": "bm25", "top_k": 4}


def test_safety_section_matches_the_guard(client: TestClient) -> None:
    safety = client.get("/api/system/info").json()["safety"]
    assert safety["disclaimer"] == DISCLAIMER
    assert safety["produces_diagnosis"] is False
    assert safety["guard_enforced"] is True


def test_guard_can_be_disabled_for_ablation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 10 ablations need to show what the guard actually removes."""
    monkeypatch.setenv("MEDAI_ENFORCE_SAFETY_GUARD", "false")
    get_settings.cache_clear()
    safety = client.get("/api/system/info").json()["safety"]
    assert safety["guard_enforced"] is False
    # Turning the lexical guard off must not make the system claim diagnoses.
    assert safety["produces_diagnosis"] is False
    get_settings.cache_clear()


def test_openapi_exposes_no_diagnosis_field(client: TestClient) -> None:
    """The structural guarantee, checked at the API surface.

    Schemas are audited for this in test_schemas.py, but a router added later
    could introduce a response model that bypasses them.
    """
    spec = client.get("/openapi.json").json()
    blob = spec["components"]["schemas"] if spec.get("components") else {}
    for model_name, model in blob.items():
        for prop in model.get("properties", {}):
            assert "diagnosis" not in prop.lower(), f"{model_name}.{prop}"


def test_openapi_description_states_the_scope_limit(client: TestClient) -> None:
    """The generated docs are public-facing; they must not oversell."""
    spec = client.get("/openapi.json").json()
    assert "not a diagnosis" in spec["info"]["description"].lower()
