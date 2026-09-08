"""Shared test fixtures.

Every database test runs against a temporary SQLite file rather than the
development database, so `make test` can never clobber a seeded demo and tests
stay order-independent.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.config import BACKEND_ROOT
from app.db.seed import load_fixture, load_monitoring_csv, reset_and_seed
from app.db.session import get_session_factory, init_db, reset_engine

DATA_DIR = BACKEND_ROOT / "data" / "synthetic"


@pytest.fixture
def demo_fixture() -> dict:
    """The raw synthetic-patient fixture, unparsed by the ORM."""
    return load_fixture()


@pytest.fixture
def demo_monitoring():
    """The 5-day vital series, validated into VitalSample objects."""
    return load_monitoring_csv()


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the app at a throwaway database and build the schema."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("MEDAI_DATABASE_URL", f"sqlite:///{db_path}")
    # reset_engine disposes any cached engine and clears the lru_cache'd
    # Settings, so the new URL takes effect.
    reset_engine()
    init_db()
    try:
        yield db_path
    finally:
        reset_engine()


@pytest.fixture
def seeded_db(tmp_db: Path) -> Iterator[Path]:
    """A temporary database already populated with the demo patient."""
    reset_and_seed(verbose=False)
    yield tmp_db


@pytest.fixture
def session(seeded_db: Path) -> Iterator[Session]:
    """An open session over the seeded temporary database."""
    s = get_session_factory()()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def fixture_dir() -> Path:
    return DATA_DIR
