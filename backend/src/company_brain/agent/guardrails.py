"""Runtime guardrails implemented as ADK callbacks.

Defence in depth: the prompt asks the model to behave, the tools enforce hard rules, and these
callbacks verify the model's *output* independently of what the model claims it did.

* ``before_tool_callback``  — per-turn tool budget (stops runaway loops and cost blow-ups).
* ``after_model_callback``  — on the final text of a turn:
    1. verifies every ``[chunk_id]`` citation was actually retrieved in this session;
    2. scans ```copy blocks with the compliance engine and redacts blocking violations;
    3. flags answers that were produced without consulting the knowledge base;
    4. attaches a structured report (``custom_metadata["guardrails"]``) that the API streams to the UI.
"""

from __future__ import annotations

import logging
from typing import Any

from google.adk.agents.context import Context as CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import BaseTool, ToolContext
from google.genai import types

from ..config import Settings
from .citations import SEEN_CHUNKS_KEY, extract_citations, replace_unverified
from .compliance import ComplianceEngine, extract_copy_blocks, redact_blocked_copy

logger = logging.getLogger(__name__)

UNGROUNDED_MIN_CHARS = 280


# Tools that finish the job (verify + persist). They get their own budget so a research-heavy turn
# (e.g. a creative brief that needs many searches) can never starve the compliance step.
ACTION_TOOLS = frozenset({"check_compliance", "save_creative_brief"})


def make_before_tool_callback(settings: Settings):
    async def before_tool(
        tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        is_action = tool.name in ACTION_TOOLS
        key, limit = (
            ("temp:action_calls", settings.max_action_calls_per_turn)
            if is_action
            else ("temp:tool_calls", settings.max_tool_calls_per_turn)
        )
        calls = int(tool_context.state.get(key, 0)) + 1
        tool_context.state[key] = calls
        logger.info("tool_call name=%s n=%d args=%s", tool.name, calls, _short(args))
        if calls > limit:
            logger.warning("%s budget exhausted (%d calls)", "action" if is_action else "retrieval", calls)
            return {
                "status": "error",
                "error_code": "TOOL_BUDGET_EXHAUSTED",
                "message": "Tool-call budget for this turn is exhausted. Answer now with the evidence "
                "you already have, and clearly state what you could not verify.",
            }
        return None

    return before_tool


def make_after_model_callback(engine: ComplianceEngine):
    def after_model(
        callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        # Streaming chunks and tool-calling steps are not final answers.
        if llm_response.partial or not llm_response.content or not llm_response.content.parts:
            return None
        parts = llm_response.content.parts
        if any(p.function_call for p in parts):
            return None
        text_parts = [p for p in parts if p.text and not p.thought]
        if not text_parts:
            return None

        original = "".join(p.text for p in text_parts)
        text = original
        state = callback_context.state

        # 1. Citation verification -------------------------------------------------------
        seen: dict[str, Any] = state.get(SEEN_CHUNKS_KEY) or {}
        cited = extract_citations(text)
        unverified = {c for c in cited if c not in seen}
        text = replace_unverified(text, unverified)

        # 2. Compliance on customer-facing copy ------------------------------------------
        text, blocked = redact_blocked_copy(engine, text)
        copy_blocks = extract_copy_blocks(original)
        warnings: list[dict[str, str]] = []
        missing_disclaimers: list[str] = []
        if copy_blocks:
            report = engine.scan("\n\n".join(b[2] for b in copy_blocks))
            warnings = [
                {"rule_id": v.rule_id, "match": v.match}
                for v in report.violations
                if v.severity == "warn"
            ]
            missing_disclaimers = [d.id for d in report.missing_disclaimers]

        # 3. Grounding ---------------------------------------------------------------------
        searches = int(state.get("temp:kb_searches", 0))
        confidence = state.get("temp:best_confidence", "none")
        ungrounded = searches == 0 and len(original) >= UNGROUNDED_MIN_CHARS

        report_payload = {
            "citations": {
                "cited": cited,
                "verified": [c for c in cited if c in seen],
                "unverified": sorted(unverified),
            },
            "compliance": {
                "copy_blocks": len(copy_blocks),
                "blocked": blocked,
                "warnings": warnings,
                "missing_disclaimers": missing_disclaimers,
                "rules_version": engine.version,
            },
            "grounding": {
                "kb_searches": searches,
                "best_confidence": confidence,
                "ungrounded": ungrounded,
            },
            "modified": text != original,
        }
        if unverified or blocked or ungrounded:
            logger.warning(
                "guardrail triggered unverified=%s blocked=%s ungrounded=%s",
                sorted(unverified),
                [b["rule_id"] for b in blocked],
                ungrounded,
            )

        thought_parts = [p for p in parts if p.thought]
        new_content = types.Content(
            role=llm_response.content.role or "model",
            parts=[*thought_parts, types.Part(text=text)],
        )
        return llm_response.model_copy(
            update={
                "content": new_content,
                "custom_metadata": {**(llm_response.custom_metadata or {}), "guardrails": report_payload},
            }
        )

    return after_model


def _short(args: dict[str, Any], limit: int = 160) -> str:
    s = str(args)
    return s if len(s) <= limit else s[:limit] + "…"
