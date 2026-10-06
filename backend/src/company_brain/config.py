"""Central, typed configuration. Everything is overridable through environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root when running from source (backend/src/company_brain/config.py -> repo root).
# Inside Docker everything is passed explicitly via env vars, so this is only a default.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # LLM
    google_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    llm_mode: Literal["gemini", "mock"] = "gemini"
    # Reasoning depth. "low" keeps tool-using Q&A fast; "" lets the model decide.
    gemini_thinking_level: Literal["", "minimal", "low", "medium", "high"] = "low"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = ""
    qdrant_collection: str = "company_brain_kb"

    # Embeddings
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    sparse_model: str = "Qdrant/bm25"
    embedding_cache_dir: str = ".cache/models"

    # Retrieval
    retrieval_top_k: int = Field(5, ge=1, le=20)
    retrieval_candidates: int = Field(20, ge=5, le=100)
    relevance_min: float = Field(0.35, ge=0, le=1)
    relevance_high: float = Field(0.55, ge=0, le=1)

    # Guardrails
    max_tool_calls_per_turn: int = Field(12, ge=1)  # retrieval tools: search / get_document / list
    max_action_calls_per_turn: int = Field(6, ge=1)  # check_compliance / save_creative_brief
    max_llm_calls_per_turn: int = Field(14, ge=2)
    max_message_chars: int = Field(2000, ge=50)

    # API
    cors_origins: str = "http://localhost:5173,http://localhost:8080"
    ws_auth_token: str = ""
    ws_max_messages_per_minute: int = Field(20, ge=1)
    session_db_url: str = "sqlite+aiosqlite:///data/runtime/sessions.db"
    kb_dir: str = "data/knowledge_base"
    output_dir: str = "data/runtime/briefs"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    auto_ingest: bool = True

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    # -- derived helpers ---------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def resolve(self, value: str) -> Path:
        """Resolve a possibly-relative path against the project root."""
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def kb_path(self) -> Path:
        return self.resolve(self.kb_dir)

    @property
    def compliance_rules_path(self) -> Path:
        return self.kb_path / "compliance" / "compliance_rules.yaml"

    @property
    def output_path(self) -> Path:
        return self.resolve(self.output_dir)

    @property
    def embedding_cache_path(self) -> Path:
        return self.resolve(self.embedding_cache_dir)

    @property
    def resolved_session_db_url(self) -> str:
        """Make relative sqlite file URLs absolute so the CWD does not matter."""
        prefix = "sqlite+aiosqlite:///"
        if self.session_db_url.startswith(prefix):
            raw = self.session_db_url[len(prefix) :]
            if raw and raw != ":memory:" and not Path(raw).is_absolute():
                path = PROJECT_ROOT / raw
                path.parent.mkdir(parents=True, exist_ok=True)
                return f"{prefix}{path}"
        return self.session_db_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Export .env into os.environ so google-genai picks up GOOGLE_API_KEY too.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings()
