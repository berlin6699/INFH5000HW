"""End-to-end checks for the small, fully-offline first MVP."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_demo_bootstrap_contains_all_dashboard_inputs(
    seeded_db: Path,
) -> None:
    with TestClient(app) as client:
        response = client.get("/api/demo")
    assert response.status_code == 200
    body = response.json()
    assert body["patient"]["patient_id"] == "PT-DEMO-001"
    assert len(body["monitoring"]) == 5
    assert body["symptoms"]["symptoms"]
    assert body["imaging"]["source_mode"] == "mock_preset"


def test_full_mock_workflow_is_high_risk_and_transparent(
    seeded_db: Path,
) -> None:
    with TestClient(app) as client:
        demo = client.get("/api/demo").json()
        response = client.post(
            "/api/analysis/run",
            json={
                "patient_id": "PT-DEMO-001",
                "symptoms": demo["symptoms"]["symptoms"],
                "free_text": demo["symptoms"]["free_text"],
                "imaging_mode": "mock_preset",
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert result["assessment"]["risk_level"] == "HIGH"
    assert len(result["traces"]) == 6
    assert all(trace["llm_used"] is False for trace in result["traces"])
    assert result["imaging"]["source_mode"] == "mock_preset"
    assert "MOCK OUTPUT" in result["imaging"]["badge"]
    assert result["assessment"]["historical_changes"]
    assert any("No LLM" in item for item in result["assessment"]["limitations"])
