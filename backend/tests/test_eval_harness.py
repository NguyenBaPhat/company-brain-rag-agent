"""The eval harnesses are production code too: make sure they run and the retrieval gate holds."""

import json

import pytest

from company_brain.evaluation.agent_eval import Outcome, check, run_case
from company_brain.evaluation.retrieval import evaluate, load_cases

pytestmark = pytest.mark.slow


def test_retrieval_quality_and_gate_meet_targets(store, settings):
    report = evaluate(store, load_cases(settings.resolve("data/eval/retrieval_cases.jsonl")), settings)
    assert report.hit_at[3] >= 0.95 and report.mrr >= 0.9
    gate = {r["threshold"]: r for r in report.gate_sweep([settings.relevance_min])}[settings.relevance_min]
    assert gate["answerable_pass"] >= 0.95  # the gate must not reject answerable questions
    assert gate["off_domain_reject"] == 1.0  # ...and must reject clearly out-of-domain ones


async def test_agent_eval_harness_runs_with_mock_llm(deps):
    cases = [json.loads(line) for line in deps.settings.resolve("data/eval/agent_cases.jsonl").read_text().splitlines()]
    assert len(cases) >= 12 and len({c["id"] for c in cases}) == len(cases)
    case = next(c for c in cases if c["id"] == "e01")
    out = await run_case(deps, case)
    assert out.tool_calls and out.text and not out.error


def test_check_flags_each_kind_of_failure(deps):
    out = Outcome(text="Paris. Dermatologist recommended [made-up#id]", guardrails={"citations": {"unverified": ["made-up#id"]}})
    fails = check({"expect_tools": ["search_knowledge_base"], "must_not_match": ["Paris"], "insufficient": True,
                   "cite_docs_any": ["product-glow-serum"], "answer_language": "vi"}, out, deps)
    joined = " | ".join(fails)
    for expected in ("tool not called", "unverified citations", "must_not_match", "not enough information", "cited", "Vietnamese"):
        assert expected in joined
