"""Behavioural evaluation of the whole agent against a real LLM.  Run with `cb-eval-agent`.

Each case asserts *behaviour* (tools used, English-only search queries, valid citations of the right
documents, refusal when the KB is silent, compliant copy, answer language), not exact wording, so it
is robust to model phrasing. Requires GOOGLE_API_KEY (LLM_MODE=gemini); with LLM_MODE=mock it only
smoke-tests the harness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from google.adk.sessions import InMemorySessionService
from google.genai import types

from ..agent import APP_NAME, AgentDeps, build_deps, build_run_config, build_runner
from ..agent.citations import extract_citations
from ..agent.compliance import scan_copy_blocks
from ..config import Settings, get_settings
from ..knowledge.ingest import build_store, ingest

CASES_PATH = "data/eval/agent_cases.jsonl"
HEDGE = re.compile(
    r"(?i)(not (contain|cover|include|have|specif|mention|find|enough|in the knowledge)|no (information|data|evidence)|"
    r"cannot (answer|find|confirm|help|write)|can't (answer|find|confirm|help|write)|only (help|assist)|unable to|"
    r"outside (of )?(lumera|the|my|our)|"
    r"doesn't (contain|cover|include|say)|does not (contain|cover|include|say)|"
    r"không (có|tìm thấy|đề cập|đủ|chứa|thể)|chưa có|ngoài phạm vi)"
)
VI_CHARS = re.compile(r"[ăâđêôơưàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹ]", re.I)


@dataclass
class Outcome:
    text: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    tool_results: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    guardrails: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0


async def run_case(deps: AgentDeps, case: dict[str, Any]) -> Outcome:
    svc = InMemorySessionService()
    runner = build_runner(deps, svc)
    sid = uuid.uuid4().hex
    await svc.create_session(app_name=APP_NAME, user_id="eval", session_id=sid)
    out, started = Outcome(), time.perf_counter()
    try:
        async for ev in runner.run_async(
            user_id="eval",
            session_id=sid,
            new_message=types.Content(role="user", parts=[types.Part(text=case["input"])]),
            state_delta={"language": case.get("language", "en")},
            run_config=build_run_config(deps),
        ):
            for p in (ev.content.parts if ev.content and ev.content.parts else []):
                if p.function_call:
                    out.tool_calls.append((p.function_call.name, dict(p.function_call.args or {})))
                elif p.function_response:
                    out.tool_results.append((p.function_response.name, dict(p.function_response.response or {})))
            if not ev.partial and ev.content and ev.content.role == "model":
                text = "".join(p.text or "" for p in (ev.content.parts or []) if not p.thought)
                if text and not any(p.function_call for p in ev.content.parts):
                    out.text = text
                    out.guardrails = (ev.custom_metadata or {}).get("guardrails", {})
    except Exception as exc:  # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"
    out.seconds = time.perf_counter() - started
    return out


def check(case: dict[str, Any], out: Outcome, deps: AgentDeps) -> list[str]:
    """Return the list of failed expectations (empty = pass)."""
    if out.error:
        return [f"agent error: {out.error[:160]}"]
    fails: list[str] = []
    names = {n for n, _ in out.tool_calls}
    for tool in case.get("expect_tools", []):
        if tool not in names:
            fails.append(f"tool not called: {tool}")
    if case.get("expect_no_tools") and out.tool_calls:
        fails.append(f"out-of-scope request should not use tools, called {sorted(names)}")
    # Every search query that was accepted by the tool must have been English.
    for (_, args), (_, res) in zip(
        [c for c in out.tool_calls if c[0] == "search_knowledge_base"],
        [r for r in out.tool_results if r[0] == "search_knowledge_base"],
        strict=False,
    ):
        if res.get("error_code") != "QUERY_NOT_ENGLISH" and not str(args.get("query", "")).isascii():
            fails.append("accepted a non-English search query")
    cited = extract_citations(out.text)
    if out.guardrails.get("citations", {}).get("unverified"):
        fails.append(f"unverified citations: {out.guardrails['citations']['unverified']}")
    if case.get("cite_docs_any") and not any(c.split("#")[0] in case["cite_docs_any"] for c in cited):
        fails.append(f"none of {case['cite_docs_any']} cited (got {cited})")
    for needle in [case["contains_any"]] if case.get("contains_any") else []:
        if not any(s.lower() in out.text.lower() for s in needle):
            fails.append(f"expected one of {needle} in the answer")
    for pat in case.get("must_match", []):
        if not re.search(pat, out.text):
            fails.append(f"must_match failed: {pat}")
    for pat in case.get("must_not_match", []):
        if re.search(pat, out.text):
            fails.append(f"must_not_match hit: {pat}")
    if case.get("insufficient") and not HEDGE.search(out.text):
        fails.append("expected a clear 'not enough information' answer")
    if case.get("no_blocked_copy"):
        report = scan_copy_blocks(deps.compliance, out.text)
        if report is not None and not report.passed:
            fails.append("final answer contains non-compliant copy")
        if out.guardrails.get("compliance", {}).get("blocked"):
            fails.append("guardrail had to block copy (the agent should self-correct first)")
    saved = any(r.get("status") == "saved" for n, r in out.tool_results if n == "save_creative_brief")
    if case.get("brief_saved") and not saved:
        fails.append("brief was not saved")
    lang = case.get("answer_language")
    if lang == "vi" and not VI_CHARS.search(out.text):
        fails.append("answer is not in Vietnamese")
    if lang == "en" and VI_CHARS.search(out.text):
        fails.append("answer is not in English")
    return fails


async def evaluate(settings: Settings, only: set[str] | None, verbose: bool = False, repeat: int = 1) -> int:
    store = build_store(settings)
    if store.count() == 0:
        ingest(settings, store)
    deps = build_deps(settings, store)
    cases = [json.loads(line) for line in settings.resolve(CASES_PATH).read_text().splitlines() if line.strip()]
    if only:
        cases = [c for c in cases if c["id"] in only]
    failed = 0
    print(f"model={'mock' if settings.llm_mode == 'mock' else settings.gemini_model}  cases={len(cases)}\n")
    cases = [c for c in cases for _ in range(repeat)]
    for case in cases:
        out = await run_case(deps, case)
        fails = check(case, out, deps)
        failed += bool(fails)
        verdict = "PASS" if not fails else "FAIL"
        print(f"[{verdict}] {case['id']} ({out.seconds:4.1f}s, {len(out.tool_calls)} tool calls) {case['input'][:70]}")
        for f in fails:
            print(f"         - {f}")
        if fails and verbose:
            print("         answer: " + out.text[:600].replace("\n", " "))
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behavioural evals against the live agent.")
    parser.add_argument("--only", default="", help="Comma-separated case ids.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print the answer of failing cases.")
    parser.add_argument("--repeat", type=int, default=1, help="Run each case N times to measure flakiness.")
    args = parser.parse_args()
    logging.basicConfig(level="WARNING")
    settings = get_settings()
    if settings.llm_mode == "gemini" and not settings.google_api_key:
        sys.exit("GOOGLE_API_KEY is not set. Set it in .env, or use LLM_MODE=mock to smoke-test the harness.")
    only = {c.strip() for c in args.only.split(",") if c.strip()} or None
    sys.exit(1 if asyncio.run(evaluate(settings, only, args.verbose, args.repeat)) else 0)


if __name__ == "__main__":
    main()
