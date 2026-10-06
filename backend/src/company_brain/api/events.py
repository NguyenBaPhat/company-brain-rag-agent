"""Translate ADK runner events into the small, stable WebSocket protocol the UI consumes.

Keeping this a pure function (event in, list of dicts out) makes the streaming behaviour unit
testable without a socket or an LLM, and isolates the UI from ADK's internal event schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from google.adk.events import Event

MAX_SOURCE_CHARS = 900


@dataclass
class TurnState:
    """Mutable per-turn bookkeeping while translating a stream of ADK events."""

    streamed_chars: int = 0
    final_sent: bool = False
    tool_names: dict[str, str] = field(default_factory=dict)
    prompt_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    llm_calls: int = 0


def _summarise(name: str, resp: dict[str, Any]) -> str:
    status = resp.get("status", "ok")
    if status == "error":
        return str(resp.get("error_code", "error"))
    if name == "search_knowledge_base":
        n = len(resp.get("results", []))
        return f"{n} passages · confidence {resp.get('confidence', 'none')}"
    if name == "get_document":
        return str(resp.get("title") or resp.get("message", ""))
    if name == "list_knowledge_sources":
        return f"{resp.get('count', 0)} documents"
    if name == "check_compliance":
        v = resp.get("violations", [])
        return "passed" if resp.get("passed") else f"{len(v)} violation(s)"
    if name == "save_creative_brief":
        return str(resp.get("brief_id") or f"{len(resp.get('problems', []))} problem(s)")
    return str(status)


def _sources_event(resp: dict[str, Any]) -> dict[str, Any] | None:
    items = resp.get("results") or resp.get("sections") or []
    if not items:
        return None
    return {
        "type": "sources",
        "chunks": [
            {
                "chunk_id": r["chunk_id"],
                "document": r.get("document"),
                "section": r.get("section"),
                "status": r.get("status"),
                "owner": r.get("owner"),
                "relevance": r.get("relevance"),
                "text": (r.get("text") or "")[:MAX_SOURCE_CHARS],
            }
            for r in items
        ],
    }


def translate_event(event: Event, turn: TurnState) -> list[dict[str, Any]]:
    """Convert one ADK `Event` into zero or more protocol messages."""
    out: list[dict[str, Any]] = []

    if event.usage_metadata and not event.partial:
        turn.prompt_tokens += int(event.usage_metadata.prompt_token_count or 0)
        turn.output_tokens += int(event.usage_metadata.candidates_token_count or 0)

    if getattr(event, "error_code", None) or getattr(event, "error_message", None):
        out.append(
            {
                "type": "error",
                "code": str(event.error_code or "MODEL_ERROR"),
                "message": str(event.error_message or "The model returned an error."),
            }
        )
        return out

    parts = event.content.parts if event.content and event.content.parts else []

    # 1. Streaming text deltas ------------------------------------------------------------
    if event.partial:
        delta = "".join(p.text for p in parts if p.text and not p.thought)
        if delta:
            turn.streamed_chars += len(delta)
            out.append({"type": "token", "delta": delta})
        return out

    # 2. Tool calls (complete, non-partial) ------------------------------------------------
    calls = [p.function_call for p in parts if p.function_call]
    for fc in calls:
        turn.tool_calls += 1
        call_id = fc.id or f"call-{turn.tool_calls}"
        turn.tool_names[call_id] = fc.name
        out.append({"type": "tool_call", "id": call_id, "name": fc.name, "args": dict(fc.args or {})})

    # 3. Tool results ----------------------------------------------------------------------
    for p in parts:
        fr = p.function_response
        if not fr:
            continue
        resp = dict(fr.response or {})
        out.append(
            {
                "type": "tool_result",
                "id": fr.id or "",
                "name": fr.name,
                "status": resp.get("status", "ok"),
                "summary": _summarise(fr.name, resp),
                **(
                    {"download_path": resp["download_path"]}
                    if resp.get("download_path")
                    else {}
                ),
            }
        )
        if fr.name in ("search_knowledge_base", "get_document"):
            src = _sources_event(resp)
            if src:
                out.append(src)

    # 4. Final answer: a non-partial text event with no tool call --------------------------
    text = "".join(p.text for p in parts if p.text and not p.thought)
    if text and not calls and not turn.final_sent and event.author != "user":
        turn.final_sent = True
        meta = event.custom_metadata or {}
        out.append({"type": "final", "text": text, "guardrails": meta.get("guardrails")})
    return out
