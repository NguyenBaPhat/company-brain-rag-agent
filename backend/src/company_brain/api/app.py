"""FastAPI application factory.

Run with:  uvicorn company_brain.api.app:create_app --factory --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from google.adk.sessions import BaseSessionService

from .. import __version__
from ..agent import APP_NAME, build_deps, build_runner
from ..agent.prompts import LANGUAGES
from ..config import Settings, get_settings
from ..knowledge.ingest import build_store, ingest
from ..knowledge.store import KnowledgeStore
from ..logging_config import setup_logging
from .ws import SESSION_ID_RE, USER_ID, register_ws

logger = logging.getLogger(__name__)

_BRIEF_ID = re.compile(r"^[a-z0-9-]{3,100}$")
_CHUNK_ID = re.compile(r"^[a-z0-9_-]+#[a-z0-9_-]+$")


def _prepare_store(store: KnowledgeStore, settings: Settings) -> None:
    """Blocking startup work: ensure collection, optional first-time ingest, warm the models."""
    store.ensure_collection()
    if store.count() == 0 and settings.auto_ingest:
        logger.info("Collection is empty: running first-time ingest")
        ingest(settings, store)
    if store.count() > 0:
        store.search("warm up", top_k=1)  # loads dense + sparse models so the first user is not slow


def create_app(
    settings: Settings | None = None,
    *,
    store: KnowledgeStore | None = None,
    session_service: BaseSessionService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        kb_store = store or build_store(settings)
        await asyncio.to_thread(_prepare_store, kb_store, settings)
        svc = session_service
        if svc is None:
            from google.adk.sessions import DatabaseSessionService

            svc = DatabaseSessionService(db_url=settings.resolved_session_db_url)
        deps = build_deps(settings, kb_store)
        app.state.store = kb_store
        app.state.deps = deps
        app.state.session_service = svc
        app.state.runner = build_runner(deps, svc)
        app.state.ready = True
        logger.info(
            "Company Brain ready (llm_mode=%s model=%s docs=%d chunks)",
            settings.llm_mode, settings.gemini_model, kb_store.count(),
        )
        yield
        app.state.ready = False

    app = FastAPI(title="Company Brain", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "DELETE"],
        allow_headers=["*"],
    )

    def llm_configured() -> bool:
        return settings.llm_mode == "mock" or bool(settings.google_api_key)

    # ---------------------------------------------------------------- health
    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    async def readyz(response: Response) -> dict[str, object]:
        """Readiness: vector store reachable, KB indexed, LLM credentials present."""
        kb: KnowledgeStore | None = getattr(app.state, "store", None)
        qdrant_ok = bool(kb and await asyncio.to_thread(kb.healthy))
        chunks = await asyncio.to_thread(kb.count) if qdrant_ok and kb else 0
        ready = bool(getattr(app.state, "ready", False)) and qdrant_ok and chunks > 0
        if not ready:
            response.status_code = 503
        return {
            "ready": ready,
            "vector_store": qdrant_ok,
            "indexed_chunks": chunks,
            "llm_configured": llm_configured(),
        }

    # ------------------------------------------------------------------- api
    @app.get("/api/config", tags=["api"])
    async def config() -> dict[str, object]:
        return {
            "version": __version__,
            "model": settings.gemini_model if settings.llm_mode == "gemini" else "mock",
            "llm_mode": settings.llm_mode,
            "llm_configured": llm_configured(),
            "languages": list(LANGUAGES),
            "max_message_chars": settings.max_message_chars,
        }

    @app.get("/api/sources", tags=["api"])
    async def sources() -> dict[str, object]:
        docs = await asyncio.to_thread(app.state.store.list_documents)
        return {"count": len(docs), "documents": docs}

    @app.get("/api/sessions", tags=["api"])
    async def list_sessions() -> dict[str, object]:
        """Conversation history, newest first. Read straight from the ADK session service."""
        resp = await app.state.session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID)
        items = [
            {
                "id": sess.id,
                "title": sess.state.get("title"),
                "updated_at": sess.last_update_time,
                "language": sess.state.get("language", "en"),
            }
            for sess in resp.sessions
            if sess.state.get("title")  # sessions opened but never used have no title: hide them
        ]
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return {"count": len(items), "sessions": items[:100]}

    @app.delete("/api/sessions/{session_id}", status_code=204, tags=["api"])
    async def delete_session(session_id: str) -> Response:
        if not SESSION_ID_RE.match(session_id):
            raise HTTPException(400, "invalid session id")
        await app.state.session_service.delete_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        return Response(status_code=204)

    @app.get("/api/chunks", tags=["api"])
    async def chunk(chunk_id: str) -> dict[str, object]:
        """Fetch one passage by citation id. `#` must be URL-encoded (%23) by the caller."""
        if not _CHUNK_ID.match(chunk_id):
            raise HTTPException(400, "invalid chunk id")
        found = await asyncio.to_thread(app.state.store.get_chunk, chunk_id)
        if not found:
            raise HTTPException(404, "chunk not found")
        return found.model_dump(exclude={"content_hash"})

    @app.get("/api/briefs/{brief_id}", tags=["api"])
    async def brief(brief_id: str, download: bool = False) -> Response:
        if not _BRIEF_ID.match(brief_id):
            raise HTTPException(400, "invalid brief id")
        path = settings.output_path / f"{brief_id}.md"
        if not path.is_file():
            raise HTTPException(404, "brief not found")
        headers = {"Content-Disposition": f'attachment; filename="{brief_id}.md"'} if download else {}
        return Response(path.read_text("utf-8"), media_type="text/markdown; charset=utf-8", headers=headers)

    register_ws(app)
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        "company_brain.api.app:create_app",
        factory=True,
        host="0.0.0.0",  # noqa: S104 - container-friendly default
        port=8000,
    )


if __name__ == "__main__":
    main()
