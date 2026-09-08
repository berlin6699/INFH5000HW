"""Engine, session factory and declarative base.

The engine is created lazily so tests can point MEDAI_DATABASE_URL at a
temporary file before anything connects.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_conn: Any, _record: Any) -> None:
    """SQLite does not enforce FOREIGN KEY constraints unless asked per connection.

    Without this the schema's relationships would be decorative: orphaned
    medical_records or monitoring rows could be inserted and deleted patients
    would leave their children behind. Enabling it makes referential integrity
    actually hold, which matters because `make seed` and `make distclean`
    recreate these tables repeatedly.
    """
    if dbapi_conn.__class__.__module__.startswith("sqlite3"):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_settings().sqlalchemy_url
        kwargs: dict[str, Any] = {}
        if url.startswith("sqlite"):
            # FastAPI serves requests from a threadpool; SQLite connections are
            # thread-affine by default.
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, future=True, **kwargs)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
        )
    return _session_factory


def reset_engine() -> None:
    """Dispose cached engine/session state. Used by tests after changing the URL."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    get_settings.cache_clear()


def init_db() -> None:
    # Imported here so every model module is registered on Base.metadata
    # before create_all runs.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())


def drop_all() -> None:
    from app.db import models  # noqa: F401

    Base.metadata.drop_all(bind=get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI dependency. Commits on success, rolls back on exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
