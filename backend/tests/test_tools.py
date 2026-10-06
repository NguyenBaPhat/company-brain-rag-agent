import pytest

from company_brain.agent.tools import REQUIRED_BRIEF_SECTIONS, build_tools

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tools(deps):
    return {fn.__name__: fn for fn in build_tools(deps)}


async def test_search_rejects_non_english_query(tools, tool_context):
    res = await tools["search_knowledge_base"]("cách dùng serum", tool_context)
    assert res["status"] == "error" and res["error_code"] == "QUERY_NOT_ENGLISH"


@pytest.mark.parametrize(
    "query,code",
    [("", "EMPTY_QUERY"), ("x" * 400, "QUERY_TOO_LONG")],
)
async def test_search_validates_input(tools, tool_context, query, code):
    assert (await tools["search_knowledge_base"](query, tool_context))["error_code"] == code


async def test_search_rejects_unknown_doc_type(tools, tool_context):
    res = await tools["search_knowledge_base"]("serum", tool_context, doc_types=["bogus"])
    assert res["error_code"] == "INVALID_DOC_TYPE"


async def test_search_ok_returns_citable_chunks_and_records_state(tools, tool_context):
    res = await tools["search_knowledge_base"]("Glow Serum usage instructions how to apply", tool_context)
    assert res["status"] == "ok" and res["confidence"] in ("medium", "high")
    top = res["results"][0]
    assert top["chunk_id"] == "product-glow-serum#how-to-use" and "2 pumps" in top["text"]
    assert top["chunk_id"] in tool_context.state["kb_seen_chunks"]
    assert tool_context.state["temp:kb_searches"] == 1


async def test_gate_returns_no_relevant_results_for_off_domain(tools, tool_context):
    res = await tools["search_knowledge_base"]("what is the capital of France", tool_context)
    assert res["status"] == "no_relevant_results" and res["results"] == []
    assert "Do NOT answer from general knowledge" in res["guidance"]
    assert tool_context.state.get("kb_seen_chunks") is None  # nothing retrievable was recorded


async def test_get_document_and_catalogue(tools, tool_context):
    doc = await tools["get_document"]("compliance-claims-policy", tool_context)
    assert doc["status"] == "ok" and len(doc["sections"]) >= 5
    assert (await tools["get_document"]("does-not-exist", tool_context))["status"] == "not_found"
    cat = await tools["list_knowledge_sources"]("product")
    assert cat["count"] == 3


async def test_check_compliance_tool(tools):
    bad = await tools["check_compliance"]("Clinically proven to cure acne")
    assert not bad["passed"] and {v["rule_id"] for v in bad["violations"]} >= {"no-endorsement-claims", "no-disease-claims"}
    assert (await tools["check_compliance"]("Glow, without the stinging."))["passed"]


GOOD_BRIEF = """## Objective
Lower CAC below $35 [product-glow-serum#overview].
## Audience
Sensitive-Skin Sara, cold audience.
## Key message
Glow, without the stinging.
## Proof points
- Fragrance-free gel serum [product-glow-serum#key-ingredients].
## Claims, disclaimers and prohibited claims
Approved: helps visibly brighten the look of dull skin. Prohibited: treatment claims.
## Hook ideas
```copy
Glow, without the stinging.
```
## Format and placement
15-second vertical UGC video.
## Call to action
```copy
Try it for 30 days.
```
## Success metrics and test plan
KPI CAC; one variable; $1,500 minimum per variant; kill at frequency 3.5.
"""


@pytest.fixture
async def seen_context(tools, tool_context):
    await tools["search_knowledge_base"]("Glow Serum ingredients overview", tool_context)
    tool_context.state["kb_seen_chunks"].update(
        {"product-glow-serum#overview": {}, "product-glow-serum#key-ingredients": {}}
    )
    return tool_context


async def test_save_brief_success(tools, seen_context, settings):
    res = await tools["save_creative_brief"]("Glow Serum UGC test", GOOD_BRIEF, seen_context)
    assert res["status"] == "saved", res
    saved = settings.output_path / f"{res['brief_id']}.md"
    assert saved.is_file() and "compliance_rules_version" in saved.read_text()


async def test_save_brief_rejects_fabricated_citation(tools, seen_context):
    body = GOOD_BRIEF.replace("product-glow-serum#key-ingredients", "invented-doc#made-up")
    res = await tools["save_creative_brief"]("Glow Serum UGC test", body, seen_context)
    assert res["status"] == "rejected" and any("fabrication" in p for p in res["problems"])


async def test_save_brief_rejects_non_compliant_copy(tools, seen_context):
    body = GOOD_BRIEF.replace("Try it for 30 days.", "Clinically proven miracle serum")
    res = await tools["save_creative_brief"]("Glow Serum UGC test", body, seen_context)
    assert res["status"] == "rejected"
    assert any("no-endorsement-claims" in p for p in res["problems"])


async def test_save_brief_rejects_missing_sections(tools, seen_context):
    res = await tools["save_creative_brief"]("Glow Serum UGC test", "## Objective\n" + "text " * 120, seen_context)
    assert res["status"] == "rejected" and any("missing required sections" in p for p in res["problems"])


def test_required_sections_cover_the_sop():
    assert len(REQUIRED_BRIEF_SECTIONS) == 9
