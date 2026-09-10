"""API checks for local-image inference without loading the heavy model in tests."""

from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.models import MedicalImageRow
from app.db.session import get_session_factory
from app.main import app
from app.schemas import Abnormality, ImagingFinding, ImagingLabel, ImagingSourceMode


def _fake_result(*, image_id: str, study_date: date) -> ImagingFinding:
    return ImagingFinding(
        image_id=image_id,
        study_date=study_date,
        source_mode=ImagingSourceMode.REAL_MODEL,
        provenance="Local test model; no external API.",
        findings=["Atelectasis: model score 70.0%"],
        abnormalities=[
            Abnormality(label=ImagingLabel.ATELECTASIS, confidence=0.7, severity_score=3)
        ],
        confidence={"Atelectasis": 0.7},
        summary="Research-model signal only.",
    )


def test_image_upload_persists_structured_real_model_result(
    seeded_db, monkeypatch
) -> None:
    monkeypatch.setattr("app.main.analyse_image_bytes", lambda _data, **kwargs: _fake_result(**kwargs))
    with TestClient(app) as client:
        response = client.post(
            "/api/imaging/analyze",
            data={"patient_id": "PT-DEMO-001"},
            files={"file": ("chest.png", b"fake-png", "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["source_mode"] == "real_model"
    assert body["badge"] is None

    with get_session_factory()() as session:
        rows = session.scalars(
            select(MedicalImageRow).where(MedicalImageRow.source_mode == "real_model")
        ).all()
    assert len(rows) == 1
    assert rows[0].file_path is None


def test_image_upload_rejects_unsupported_content_type(seeded_db) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/imaging/analyze",
            data={"patient_id": "PT-DEMO-001"},
            files={"file": ("chest.gif", b"gif", "image/gif")},
        )
    assert response.status_code == 415
