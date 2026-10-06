from types import SimpleNamespace

from google.adk.events import Event
from google.genai import types

from company_brain.agent.prompts import LANGUAGES, build_system_prompt, instruction_provider
from company_brain.api.events import TurnState, translate_event


def test_prompt_contains_every_required_section_and_no_template_leftovers():
    p = build_system_prompt("en")
    for heading in ("# ROLE", "# SCOPE", "# GROUND TRUTH", "# TOOLS", "# OPERATING PROCEDURE", "# CITATIONS",
                    "# INSUFFICIENT INFORMATION PROTOCOL", "# COMPLIANCE", "# OUTPUT FORMATS", "# LANGUAGE", "# SAFETY"):
        assert heading in p
    assert "{language}" not in p and "{" not in p.replace("{", "", 0) or "{language}" not in p


def test_prompt_enforces_english_search_and_selected_answer_language():
    vi = build_system_prompt("vi")
    assert "Vietnamese" in vi and "queries MUST be written in English" in vi
    assert "English" in build_system_prompt("en")


def test_instruction_provider_follows_session_language_and_falls_back():
    assert "Vietnamese" in instruction_provider(SimpleNamespace(state={"language": "vi"}))
    assert LANGUAGES["en"] in instruction_provider(SimpleNamespace(state={}))
    assert instruction_provider(SimpleNamespace(state={"language": "xx"})) == build_system_prompt("en")


def _event(parts, partial=None, **kw):
    return Event(author="company_brain", content=types.Content(role="model", parts=parts), partial=partial, **kw)


def test_partial_text_becomes_token_events_and_skips_thoughts():
    turn = TurnState()
    out = translate_event(_event([types.Part(text="Hel"), types.Part(text="hidden", thought=True)], partial=True), turn)
    assert out == [{"type": "token", "delta": "Hel"}] and turn.streamed_chars == 3


def test_tool_call_and_result_and_sources():
    turn = TurnState()
    call = _event([types.Part(function_call=types.FunctionCall(id="c1", name="search_knowledge_base", args={"query": "q"}))])
    assert translate_event(call, turn)[0] == {"type": "tool_call", "id": "c1", "name": "search_knowledge_base", "args": {"query": "q"}}
    resp = {"status": "ok", "confidence": "high", "results": [{"chunk_id": "a#b", "document": "D", "section": "S", "status": "active", "text": "t" * 2000}]}
    result_ev = _event([types.Part(function_response=types.FunctionResponse(id="c1", name="search_knowledge_base", response=resp))])
    out = translate_event(result_ev, turn)
    assert out[0]["type"] == "tool_result" and out[0]["summary"] == "1 passages · confidence high"
    assert out[1]["type"] == "sources" and len(out[1]["chunks"][0]["text"]) == 900  # truncated for the wire


def test_final_text_emitted_once_with_guardrails_and_not_for_tool_turns():
    turn = TurnState()
    meta = {"guardrails": {"modified": False}}
    out = translate_event(_event([types.Part(text="Answer")], custom_metadata=meta), turn)
    assert out == [{"type": "final", "text": "Answer", "guardrails": {"modified": False}}]
    assert translate_event(_event([types.Part(text="Again")]), turn) == []  # only one final per turn

    t2 = TurnState()
    mixed = _event([types.Part(text="Let me look"), types.Part(function_call=types.FunctionCall(name="x", args={}))])
    assert [e["type"] for e in translate_event(mixed, t2)] == ["tool_call"]  # narration is not a final answer


def test_error_events_are_translated():
    out = translate_event(_event([], error_code="429", error_message="quota"), TurnState())
    assert out == [{"type": "error", "code": "429", "message": "quota"}]
