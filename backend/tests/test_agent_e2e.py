"""Full ADK agent loop (real Runner, real tools, real store, MockLlm) with no network or API key."""

import pytest
from google.adk.sessions import InMemorySessionService
from google.genai import types

from company_brain.agent import APP_NAME, build_run_config, build_runner

pytestmark = pytest.mark.slow


async def run(deps, text, lang="en", session_id="s1", svc=None):
    svc = svc or InMemorySessionService()
    runner = build_runner(deps, svc)
    if not await svc.get_session(app_name=APP_NAME, user_id="u", session_id=session_id):
        await svc.create_session(app_name=APP_NAME, user_id="u", session_id=session_id)
    events = []
    async for ev in runner.run_async(
        user_id="u", session_id=session_id, state_delta={"language": lang},
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
        run_config=build_run_config(deps),
    ):
        events.append(ev)
    return events


def calls(events):
    return [(p.function_call.name, dict(p.function_call.args or {})) for e in events if e.content for p in e.content.parts or [] if p.function_call]


def final(events):
    done = [e for e in events if not e.partial and e.content and e.content.role == "model" and any(p.text for p in e.content.parts or [])]
    return done[-1]


async def test_grounded_answer_cites_only_retrieved_chunks(deps):
    events = await run(deps, "How do I use the Glow Serum?")
    assert calls(events)[0][0] == "search_knowledge_base"
    ev = final(events)
    g = ev.custom_metadata["guardrails"]
    assert g["citations"]["verified"] and g["citations"]["unverified"] == []
    assert g["grounding"]["kb_searches"] == 1 and g["grounding"]["best_confidence"] in ("medium", "high")


async def test_non_english_query_is_rejected_then_retried_in_english(deps):
    events = await run(deps, "cách dùng serum vitamin C", lang="vi")
    searches = [a["query"] for n, a in calls(events) if n == "search_knowledge_base"]
    assert len(searches) == 2
    assert not searches[0].isascii() and searches[1].isascii()  # guard bounced the first, agent retried in English


async def test_out_of_scope_question_gets_insufficient_information(deps):
    events = await run(deps, "what is the capital of France")
    ev = final(events)
    assert ev.custom_metadata["guardrails"]["grounding"]["best_confidence"] == "none"
    assert "could not find" in ev.content.parts[-1].text


async def test_guardrail_blocks_non_compliant_copy_in_final_answer(deps):
    ev = final(await run(deps, "/demo violation"))
    text = ev.content.parts[-1].text
    g = ev.custom_metadata["guardrails"]
    assert "Copy blocked by compliance guardrail" in text and "```copy\nClinically" not in text
    assert g["modified"] and g["compliance"]["blocked"]


async def test_brief_flow_saves_validated_brief(deps, settings):
    events = await run(deps, "/demo brief")
    names = [n for n, _ in calls(events)]
    assert names == ["search_knowledge_base", "save_creative_brief"]
    assert "Brief saved" in final(events).content.parts[-1].text
    assert any(settings.output_path.glob("*mock-demo-brief*.md"))


async def test_language_is_stored_in_session_state_and_drives_the_prompt(deps):
    svc = InMemorySessionService()
    events = await run(deps, "Glow Serum price", lang="vi", svc=svc)
    assert "bản giả lập" in final(events).content.parts[-1].text  # mock reads the Vietnamese system prompt
    session = await svc.get_session(app_name=APP_NAME, user_id="u", session_id="s1")
    assert session.state["language"] == "vi" and session.state["kb_seen_chunks"]
