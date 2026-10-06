"""WebSocket chat endpoint: streams the ADK agent's tokens, tool activity and guardrail verdicts."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
import time
import uuid
from collections import deque
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.adk.sessions import BaseSessionService, Session
from google.genai import types
from pydantic import ValidationError

from ..agent import APP_NAME, build_run_config
from ..agent.prompts import LANGUAGES
from ..logging_config import session_id_var
from .events import TurnState, translate_event
from .schemas import CancelMessage, PingMessage, UserMessage, client_message_adapter

logger = logging.getLogger(__name__)

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
USER_ID = "web-user"  # single-tenant demo; in production derive from the authenticated identity
WS_POLICY_VIOLATION = 1008


def classify_llm_error(exc: BaseException) -> tuple[str, str]:
    """Map provider / runtime exceptions to a stable error code the UI can localise."""
    code = getattr(exc, "code", None)
    text = str(exc).lower()
    # Walk the cause chain: connection / TLS failures are usually wrapped by the SDK layers.
    chain, cur = [], exc
    while cur is not None and len(chain) < 8:
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    if any(isinstance(e, (httpx.ConnectError, httpx.TimeoutException, ssl.SSLError, ConnectionError)) for e in chain):
        return "LLM_UNREACHABLE", "Cannot reach the model service (network or TLS certificate problem)."
    if "missing key inputs" in text or "api key" in text and "not" in text and "valid" not in text:
        return "LLM_NOT_CONFIGURED", "The model API key is not configured on the server."
    if code in (401, 403) or "api key not valid" in text or "permission" in text and "denied" in text:
        return "LLM_AUTH", "The model rejected the server's credentials."
    if code == 404 or "is not found" in text or "not found for api version" in text:
        return "LLM_MODEL_NOT_FOUND", "The configured model name was not found."
    if code == 429 or "resource_exhausted" in text or "quota" in text:
        return "LLM_RATE_LIMIT", "The model is rate-limited right now. Please retry shortly."
    if isinstance(code, int) and code >= 500:
        return "LLM_UNAVAILABLE", "The model service is temporarily unavailable."
    if "max_llm_calls" in text or "llm call" in text and "limit" in text:
        return "TURN_LIMIT", "The agent reached its step limit for this request."
    return "AGENT_ERROR", "Something went wrong while generating the answer."


TITLE_MAX_CHARS = 60


def make_title(text: str) -> str:
    """Conversation title = the first user message, whitespace-collapsed and truncated."""
    flat = " ".join(text.split())
    return flat if len(flat) <= TITLE_MAX_CHARS else flat[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def history_from_session(session: Session) -> list[dict[str, Any]]:
    """Rebuild a UI-friendly transcript (user/assistant text only) from persisted ADK events."""
    items: list[dict[str, Any]] = []
    for ev in session.events:
        parts = ev.content.parts if ev.content and ev.content.parts else []
        if any(p.function_call or p.function_response for p in parts):
            continue
        text = "".join(p.text for p in parts if p.text and not p.thought)
        if not text:
            continue
        if ev.author == "user":
            items.append({"role": "user", "text": text})
        else:
            meta = ev.custom_metadata or {}
            items.append({"role": "assistant", "text": text, "guardrails": meta.get("guardrails")})
    return items


class _RateLimiter:
    """Sliding-window limiter, one instance per connection."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.stamps: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        while self.stamps and now - self.stamps[0] > 60:
            self.stamps.popleft()
        if len(self.stamps) >= self.per_minute:
            return False
        self.stamps.append(now)
        return True


def _origin_allowed(ws: WebSocket, allowed: list[str]) -> bool:
    """Browsers always send Origin on WebSocket handshakes. Accept: no Origin (non-browser
    clients), an explicitly allowed origin, or same-origin (Origin host == Host header), which is
    what a SPA served behind the same reverse proxy produces."""
    origin = ws.headers.get("origin")
    if origin is None or "*" in allowed or origin in allowed:
        return True
    host = ws.headers.get("host", "")
    return bool(host) and urlsplit(origin).netloc == host


async def _get_or_create_session(svc: BaseSessionService, session_id: str) -> Session:
    session = await svc.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    if session is None:
        session = await svc.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    return session


def register_ws(app: FastAPI) -> None:
    @app.websocket("/ws/chat")
    async def chat(ws: WebSocket) -> None:
        settings = app.state.settings
        if not _origin_allowed(ws, settings.cors_origin_list):
            await ws.close(code=WS_POLICY_VIOLATION)
            return
        if settings.ws_auth_token and ws.query_params.get("token") != settings.ws_auth_token:
            await ws.close(code=WS_POLICY_VIOLATION)
            return
        await ws.accept()

        requested = ws.query_params.get("session_id", "")
        session_id = requested if SESSION_ID_RE.match(requested) else uuid.uuid4().hex
        session_id_var.set(session_id[:8])
        svc: BaseSessionService = app.state.session_service
        send_lock = asyncio.Lock()

        async def send(event: dict[str, Any]) -> None:
            async with send_lock:
                await ws.send_text(json.dumps(event, ensure_ascii=False, default=str))

        session = await _get_or_create_session(svc, session_id)
        try:
            await send(
                {
                    "type": "session",
                    "session_id": session_id,
                    "history": history_from_session(session),
                    "model": settings.gemini_model if settings.llm_mode == "gemini" else "mock",
                    "llm_mode": settings.llm_mode,
                    "languages": list(LANGUAGES),
                }
            )
        except (WebSocketDisconnect, RuntimeError):
            # Client left before the handshake finished (e.g. React StrictMode's double mount in dev).
            logger.info("ws closed before session event")
            return
        logger.info("ws connected (history=%d events)", len(session.events))

        titled = bool(session.state.get("title"))
        limiter = _RateLimiter(settings.ws_max_messages_per_minute)
        active: asyncio.Task[None] | None = None
        abort = asyncio.Event()

        async def run_turn(msg: UserMessage) -> None:
            nonlocal titled
            message_id = msg.message_id or uuid.uuid4().hex[:12]
            turn = TurnState()
            started = time.perf_counter()
            await send({"type": "turn_start", "message_id": message_id})
            try:
                content = types.Content(role="user", parts=[types.Part(text=msg.text)])
                async for event in app.state.runner.run_async(
                    user_id=USER_ID,
                    session_id=session_id,
                    new_message=content,
                    # The title is stored in ADK session state, which is what the history sidebar lists.
                    state_delta={"language": msg.language, **({} if titled else {"title": make_title(msg.text)})},
                    run_config=build_run_config(app.state.deps),
                    abort_signal=abort,
                ):
                    for out in translate_event(event, turn):
                        await send(out)
                titled = True
                if abort.is_set():
                    await send({"type": "error", "code": "CANCELLED", "message": "Generation cancelled."})
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - convert every failure to a protocol error
                err_code, err_msg = classify_llm_error(exc)
                logger.exception("turn failed (%s)", err_code)
                await send({"type": "error", "code": err_code, "message": err_msg})
            finally:
                latency = time.perf_counter() - started
                logger.info(
                    "turn done in %.1fs tool_calls=%d tokens_in=%d tokens_out=%d",
                    latency, turn.tool_calls, turn.prompt_tokens, turn.output_tokens,
                )
                await send(
                    {
                        "type": "turn_end",
                        "message_id": message_id,
                        "usage": {
                            "prompt_tokens": turn.prompt_tokens,
                            "output_tokens": turn.output_tokens,
                            "tool_calls": turn.tool_calls,
                            "latency_ms": int(latency * 1000),
                        },
                    }
                )

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = client_message_adapter.validate_json(raw)
                except ValidationError:
                    await send({"type": "error", "code": "BAD_MESSAGE", "message": "Malformed message."})
                    continue

                if isinstance(msg, PingMessage):
                    await send({"type": "pong"})
                elif isinstance(msg, CancelMessage):
                    abort.set()
                elif isinstance(msg, UserMessage):
                    msg.text = msg.text.strip()
                    if not msg.text:
                        await send({"type": "error", "code": "BAD_MESSAGE", "message": "Empty message."})
                    elif len(msg.text) > settings.max_message_chars:
                        await send(
                            {
                                "type": "error",
                                "code": "MESSAGE_TOO_LONG",
                                "message": f"Message exceeds {settings.max_message_chars} characters.",
                            }
                        )
                    elif active and not active.done():
                        await send(
                            {"type": "error", "code": "BUSY", "message": "Still answering the previous message."}
                        )
                    elif not limiter.allow():
                        await send(
                            {"type": "error", "code": "RATE_LIMITED", "message": "Too many messages. Slow down."}
                        )
                    else:
                        abort = asyncio.Event()
                        active = asyncio.create_task(run_turn(msg))
        except WebSocketDisconnect:
            logger.info("ws disconnected")
        finally:
            abort.set()
            if active and not active.done():
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
