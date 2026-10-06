from types import SimpleNamespace

import pytest
from google.adk.models import LlmResponse
from google.genai import types

from company_brain.agent.guardrails import make_after_model_callback, make_before_tool_callback


def _resp(text=None, partial=False, call=None):
    part = types.Part(text=text) if text is not None else types.Part(function_call=types.FunctionCall(name=call, args={}))
    return LlmResponse(content=types.Content(role="model", parts=[part]), partial=partial)


def _ctx(**state):
    return SimpleNamespace(state=state)


@pytest.fixture(scope="module")
def after(deps):
    return make_after_model_callback(deps.compliance)


def _text(r):
    return r.content.parts[-1].text


def test_ignores_partial_chunks_and_tool_calls(after):
    assert after(_ctx(), _resp("par", partial=True)) is None
    assert after(_ctx(), _resp(call="search_knowledge_base")) is None


def test_verified_citation_is_kept(after):
    ctx = _ctx(kb_seen_chunks={"product-glow-serum#how-to-use": {}}, **{"temp:kb_searches": 1})
    out = after(ctx, _resp("Apply 2 pumps [product-glow-serum#how-to-use]."))
    g = out.custom_metadata["guardrails"]
    assert "[product-glow-serum#how-to-use]" in _text(out)
    assert g["citations"]["unverified"] == [] and g["modified"] is False


def test_fabricated_citation_is_neutralised(after):
    ctx = _ctx(kb_seen_chunks={"a#b": {}}, **{"temp:kb_searches": 1})
    out = after(ctx, _resp("Fact [a#b] and invented [made-up#source]."))
    g = out.custom_metadata["guardrails"]["citations"]
    assert "[a#b]" in _text(out) and "[unverified]" in _text(out) and "made-up#source" not in _text(out)
    assert g["unverified"] == ["made-up#source"] and g["verified"] == ["a#b"]


def test_blocking_copy_is_redacted_and_reported(after):
    text = "Draft:\n```copy\nClinically proven to cure acne.\n```"
    out = after(_ctx(**{"temp:kb_searches": 1}), _resp(text))
    g = out.custom_metadata["guardrails"]
    assert "Clinically proven to cure" not in _text(out).replace('("Clinically proven")', "")
    assert g["modified"] and {b["rule_id"] for b in g["compliance"]["blocked"]} >= {"no-endorsement-claims"}


def test_quoting_policy_outside_copy_is_not_blocked(after):
    out = after(_ctx(**{"temp:kb_searches": 1}), _resp('"Clinically proven" is prohibited. Use "helps visibly brighten".'))
    g = out.custom_metadata["guardrails"]
    assert g["compliance"]["blocked"] == [] and g["modified"] is False


def test_missing_disclaimer_reported_for_copy(after):
    text = "```copy\nIn a consumer study, 82% said skin looked brighter.\n```"
    g = after(_ctx(**{"temp:kb_searches": 1}), _resp(text)).custom_metadata["guardrails"]
    assert g["compliance"]["missing_disclaimers"] == ["results-disclaimer"]


def test_long_answer_without_any_kb_search_is_flagged_ungrounded(after):
    g = after(_ctx(), _resp("x" * 400)).custom_metadata["guardrails"]
    assert g["grounding"]["ungrounded"] is True
    g2 = after(_ctx(), _resp("Hello! How can I help?")).custom_metadata["guardrails"]
    assert g2["grounding"]["ungrounded"] is False


async def test_tool_budget_is_enforced(settings):
    before = make_before_tool_callback(settings.model_copy(update={"max_tool_calls_per_turn": 2}))
    ctx, tool = _ctx(), SimpleNamespace(name="search_knowledge_base")
    assert await before(tool, {"query": "a"}, ctx) is None
    assert await before(tool, {"query": "b"}, ctx) is None
    blocked = await before(tool, {"query": "c"}, ctx)
    assert blocked["error_code"] == "TOOL_BUDGET_EXHAUSTED"


async def test_compliance_step_is_never_starved_by_retrieval_budget(settings):
    cfg = settings.model_copy(update={"max_tool_calls_per_turn": 2, "max_action_calls_per_turn": 1})
    before = make_before_tool_callback(cfg)
    ctx = _ctx()
    search, check = SimpleNamespace(name="search_knowledge_base"), SimpleNamespace(name="check_compliance")
    assert await before(search, {}, ctx) is None and await before(search, {}, ctx) is None
    assert (await before(search, {}, ctx))["error_code"] == "TOOL_BUDGET_EXHAUSTED"  # retrieval is capped...
    assert await before(check, {}, ctx) is None  # ...but the compliance check still runs
    assert (await before(check, {}, ctx))["error_code"] == "TOOL_BUDGET_EXHAUSTED"  # and is itself bounded
