"""Shared fixtures. The vector store uses the REAL local embedding models (cached after first
download) so retrieval tests exercise the same code path as production, against in-memory Qdrant."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")

from company_brain.agent import build_deps  # noqa: E402
from company_brain.config import Settings  # noqa: E402
from company_brain.knowledge.chunker import chunk_documents  # noqa: E402
from company_brain.knowledge.ingest import build_embedder  # noqa: E402
from company_brain.knowledge.loader import load_documents  # noqa: E402
from company_brain.knowledge.store import KnowledgeStore  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    return Settings(
        llm_mode="mock",
        qdrant_url="",
        qdrant_path="",
        output_dir=str(tmp_path_factory.mktemp("briefs")),
        ws_max_messages_per_minute=100,
    )


@pytest.fixture(scope="session")
def store(settings) -> KnowledgeStore:
    kb = KnowledgeStore(QdrantClient(":memory:"), build_embedder(settings), "test_kb")
    kb.sync_chunks(chunk_documents(load_documents(settings.kb_path)))
    return kb


@pytest.fixture(scope="session")
def deps(settings, store):
    return build_deps(settings, store)


@pytest.fixture
def tool_context():
    """Tools only touch `.state`; a plain namespace with a dict is enough."""
    return SimpleNamespace(state={})
