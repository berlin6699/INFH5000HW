"""Runtime configuration.

Every field has an offline-safe default: the system must run end-to-end with
no .env file and no network access.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent

LLMProviderName = Literal["null", "openai", "deepseek"]
ImagingMode = Literal["mock_preset", "uploaded_report"]
RetrieverName = Literal["bm25", "embedding"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDAI_",
        env_file=(PROJECT_ROOT / ".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    database_url: str = "sqlite:///./data/medai.db"

    llm_provider: LLMProviderName = "null"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    imaging_mode: ImagingMode = "mock_preset"

    rag_retriever: RetrieverName = "bm25"
    rag_top_k: int = Field(default=4, ge=1, le=20)

    enforce_safety_guard: bool = True

    @field_validator("cors_origins")
    @classmethod
    def _strip_origins(cls, v: str) -> str:
        return v.strip()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sqlalchemy_url(self) -> str:
        """Resolve relative sqlite paths against backend/ so the DB lands in
        the same place no matter what cwd the server is started from."""
        url = self.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix):
            raw = url[len(prefix):]
            if raw and raw != ":memory:":
                p = Path(raw)
                if not p.is_absolute():
                    p = (BACKEND_ROOT / p).resolve()
                p.parent.mkdir(parents=True, exist_ok=True)
                return f"{prefix}{p}"
        return url

    @property
    def llm_enabled(self) -> bool:
        """True only if a provider is selected AND its credential is present.

        A missing key silently degrades to deterministic mode rather than
        crashing mid-demo.
        """
        if self.llm_provider == "openai":
            return bool(self.openai_api_key.strip())
        if self.llm_provider == "deepseek":
            return bool(self.deepseek_api_key.strip())
        return False


@lru_cache
def get_settings() -> Settings:
    return Settings()
