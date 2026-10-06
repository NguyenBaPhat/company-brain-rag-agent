import json

import httpx
import pytest
from fastapi.testclient import TestClient
from google.adk.sessions import InMemorySessionService
from starlette.websockets import WebSocketDisconnect

from company_brain.api.app import create_app
from company_brain.api.ws import classify_llm_error, make_title

pytestmark = pytest.mark.slow


@pytest.fixture
def client(settings, store):
    app = create_app(settings, store=store, session_service=InMemorySessionService())
    with TestClient(app) as c:
        yield c


def turn(ws, text, language="en", **extra):
    ws.send_text(json.dumps({"type": "user_message", "text": text, "language": language, **extra}))
    events = []
    while True:
        ev = json.loads(ws.receive_text())
        events.append(ev)
        if ev["type"] == "turn_end":
            return events


def types_of(events):
    return [e["type"] for e in events]


def test_health_ready_and_config(client):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["indexed_chunks"] > 60
    cfg = client.get("/api/config").json()
    assert cfg["languages"] == ["en", "vi"] and cfg["llm_mode"] == "mock"


def test_sources_and_chunk_endpoints(client):
    docs = client.get("/api/sources").json()
    assert docs["count"] >= 15
    chunk = client.get("/api/chunks", params={"chunk_id": "product-glow-serum#how-to-use"})
    assert chunk.status_code == 200 and "2 pumps" in chunk.json()["text"]
    assert client.get("/api/chunks", params={"chunk_id": "bad id"}).status_code == 400
    assert client.get("/api/chunks", params={"chunk_id": "nope#nope"}).status_code == 404


def test_streaming_turn_has_expected_event_order(client):
    with client.websocket_connect("/ws/chat") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "session" and hello["history"] == [] and hello["languages"] == ["en", "vi"]
        events = turn(ws, "How do I use the Glow Serum?", message_id="m1")
    kinds = types_of(events)
    assert kinds[0] == "turn_start" and kinds[-1] == "turn_end"
    assert kinds.index("tool_call") < kinds.index("tool_result") < kinds.index("sources") < kinds.index("token") < kinds.index("final")
    streamed = "".join(e["delta"] for e in events if e["type"] == "token")
    final = next(e for e in events if e["type"] == "final")
    assert streamed.strip() == final["text"].strip()  # no guardrail edits -> streamed text equals final text
    assert final["guardrails"]["citations"]["verified"] and not final["guardrails"]["citations"]["unverified"]
    assert events[-1]["usage"]["tool_calls"] == 1 and events[0]["message_id"] == "m1"


def test_guardrail_edit_arrives_in_final_even_though_tokens_streamed_first(client):
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_text()
        events = turn(ws, "/demo violation")
    streamed = "".join(e["delta"] for e in events if e["type"] == "token")
    final = next(e for e in events if e["type"] == "final")
    assert "Clinically proven to cure acne" in streamed  # draft was streamed...
    assert "Copy blocked by compliance guardrail" in final["text"]  # ...and the authoritative final replaced it
    assert final["guardrails"]["compliance"]["blocked"]


def test_brief_flow_reports_download_path(client, settings):
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_text()
        events = turn(ws, "/demo brief")
    saved = next(e for e in events if e["type"] == "tool_result" and e["name"] == "save_creative_brief")
    assert saved["status"] == "saved" and saved["download_path"].startswith("/api/briefs/")
    res = client.get(saved["download_path"])
    assert res.status_code == 200 and "## Objective" in res.text
    assert "attachment" in client.get(saved["download_path"] + "?download=true").headers["content-disposition"]
    assert client.get("/api/briefs/does-not-exist").status_code == 404
    assert client.get("/api/briefs/..%2Fsecret").status_code in (400, 404)


def test_session_history_is_restored_on_reconnect(client):
    sid = "test-session-1234"
    with client.websocket_connect(f"/ws/chat?session_id={sid}") as ws:
        assert json.loads(ws.receive_text())["session_id"] == sid
        turn(ws, "Glow Serum price")
    with client.websocket_connect(f"/ws/chat?session_id={sid}") as ws:
        hello = json.loads(ws.receive_text())
    roles = [m["role"] for m in hello["history"]]
    assert roles == ["user", "assistant"] and hello["history"][1]["guardrails"]["grounding"]["kb_searches"] == 1


def test_invalid_session_id_gets_a_fresh_one(client):
    with client.websocket_connect("/ws/chat?session_id=../../etc") as ws:
        assert len(json.loads(ws.receive_text())["session_id"]) == 32


@pytest.mark.parametrize(
    "payload,code",
    [
        ("not json", "BAD_MESSAGE"),
        (json.dumps({"type": "nope"}), "BAD_MESSAGE"),
        (json.dumps({"type": "user_message", "text": ""}), "BAD_MESSAGE"),
        (json.dumps({"type": "user_message", "text": "x" * 3000}), "MESSAGE_TOO_LONG"),
        (json.dumps({"type": "user_message", "text": "hi", "language": "fr"}), "BAD_MESSAGE"),
    ],
)
def test_bad_input_is_rejected_safely_and_connection_survives(client, payload, code):
    with client.websocket_connect("/ws/chat") as ws:
        ws.receive_text()
        ws.send_text(payload)
        assert json.loads(ws.receive_text())["code"] == code
        ws.send_text(json.dumps({"type": "ping"}))
        assert json.loads(ws.receive_text())["type"] == "pong"


def test_rate_limit(settings, store):
    limited = settings.model_copy(update={"ws_max_messages_per_minute": 1})
    app = create_app(limited, store=store, session_service=InMemorySessionService())
    with TestClient(app) as c, c.websocket_connect("/ws/chat") as ws:
        ws.receive_text()
        turn(ws, "Glow Serum price")
        ws.send_text(json.dumps({"type": "user_message", "text": "again"}))
        assert json.loads(ws.receive_text())["code"] == "RATE_LIMITED"


def test_origin_and_token_are_enforced(settings, store):
    secured = settings.model_copy(update={"ws_auth_token": "s3cret", "cors_origins": "http://app.example"})
    app = create_app(secured, store=store, session_service=InMemorySessionService())
    with TestClient(app) as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws/chat?token=s3cret", headers={"origin": "http://evil.example"}):
                pass
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws/chat?token=wrong", headers={"origin": "http://app.example"}):
                pass
        with c.websocket_connect("/ws/chat?token=s3cret", headers={"origin": "http://app.example"}) as ws:
            assert json.loads(ws.receive_text())["type"] == "session"
        # Same-origin (what a SPA behind the same proxy sends) is accepted without being listed...
        with c.websocket_connect("/ws/chat?token=s3cret", headers={"origin": "http://testserver"}) as ws:
            assert json.loads(ws.receive_text())["type"] == "session"
        # ...but a different host/port is still rejected.
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws/chat?token=s3cret", headers={"origin": "http://testserver:9999"}):
                pass


def _wrapped(inner: Exception) -> Exception:
    outer = RuntimeError("request failed")
    outer.__cause__ = inner
    return outer


class _Err(Exception):
    def __init__(self, code, msg=""):
        super().__init__(msg)
        self.code = code


@pytest.mark.parametrize(
    "exc,expected",
    [
        (ValueError("Missing key inputs argument! To use the Google AI API, provide (`api_key`)"), "LLM_NOT_CONFIGURED"),
        (_Err(403, "forbidden"), "LLM_AUTH"),
        (_Err(404, "models/x is not found for API version"), "LLM_MODEL_NOT_FOUND"),
        (_Err(429, "RESOURCE_EXHAUSTED"), "LLM_RATE_LIMIT"),
        (_Err(503, "overloaded"), "LLM_UNAVAILABLE"),
        (RuntimeError("boom"), "AGENT_ERROR"),
        (_wrapped(httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]")), "LLM_UNREACHABLE"),
    ],
)
def test_llm_errors_are_classified_for_the_ui(exc, expected):
    assert classify_llm_error(exc)[0] == expected


def test_history_survives_restart_with_the_sqlite_session_store(settings, store, tmp_path):
    """The production session store (ADK DatabaseSessionService on SQLite), not the in-memory fake."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}"
    cfg = settings.model_copy(update={"session_db_url": url})
    sid = "persisted-session-1"
    with TestClient(create_app(cfg, store=store)) as c, c.websocket_connect(f"/ws/chat?session_id={sid}") as ws:
        ws.receive_text()
        turn(ws, "Glow Serum price")
    with TestClient(create_app(cfg, store=store)) as c, c.websocket_connect(f"/ws/chat?session_id={sid}") as ws:  # "restart"
        hello = json.loads(ws.receive_text())
    assert [m["role"] for m in hello["history"]] == ["user", "assistant"]


def test_make_title_collapses_whitespace_and_truncates():
    assert make_title("  What   is\nthe price? ") == "What is the price?"
    long = make_title("x" * 200)
    assert len(long) == 60 and long.endswith("…")


def test_history_sidebar_lists_titles_newest_first_and_hides_unused_sessions(client):
    def chat(sid, text):
        with client.websocket_connect(f"/ws/chat?session_id={sid}") as ws:
            ws.receive_text()
            if text:
                turn(ws, text)

    chat("history-session-a1", "Glow Serum price")
    chat("history-session-b2", "How do I use   the Glow Serum?")
    chat("history-session-unused", None)  # connected, never asked anything

    sessions = client.get("/api/sessions").json()["sessions"]
    ids = [s["id"] for s in sessions]
    assert ids[:2] == ["history-session-b2", "history-session-a1"]  # newest first
    assert "history-session-unused" not in ids
    assert sessions[0]["title"] == "How do I use the Glow Serum?"
    assert sessions[0]["language"] == "en" and sessions[0]["updated_at"] > 0


def test_title_is_set_once_from_the_first_message(client):
    sid = "history-title-once-1"
    with client.websocket_connect(f"/ws/chat?session_id={sid}") as ws:
        ws.receive_text()
        turn(ws, "First question about shipping")
        turn(ws, "A completely different second question")
    titles = {s["id"]: s["title"] for s in client.get("/api/sessions").json()["sessions"]}
    assert titles[sid] == "First question about shipping"


def test_delete_session_removes_it_and_its_history(client):
    sid = "history-delete-me-1"
    with client.websocket_connect(f"/ws/chat?session_id={sid}") as ws:
        ws.receive_text()
        turn(ws, "Glow Serum price")
    assert sid in [s["id"] for s in client.get("/api/sessions").json()["sessions"]]
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert sid not in [s["id"] for s in client.get("/api/sessions").json()["sessions"]]
    with client.websocket_connect(f"/ws/chat?session_id={sid}") as ws:  # a fresh, empty session
        assert json.loads(ws.receive_text())["history"] == []
    assert client.delete("/api/sessions/bad id!").status_code in (400, 404)
