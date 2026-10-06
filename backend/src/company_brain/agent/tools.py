"""Tools exposed to the ADK agent.

Design notes
------------
* Tools are built by :func:`build_tools` as closures over :class:`AgentDeps`, so there is no global
  state and tests can inject an in-memory store.
* Tool *docstrings are prompts*: ADK sends them to the model as the function description, so they
  state precisely when to call the tool, what the arguments mean and how to read the result.
* Every tool returns a JSON-serialisable ``dict`` with a ``status`` field and **never raises** to the
  model: failures come back as ``{"status": "error", "error_code": ..., "message": ...}`` so the
  agent can recover (for example by rewriting a query in English).
* Hard business rules live here, in code, and not only in the prompt: English-only queries, the
  relevance gate, citation validation and compliance blocking when saving a brief.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from google.adk.tools import ToolContext

from ..knowledge.chunker import slugify
from ..knowledge.models import DOC_TYPES, SearchHit
from .citations import SEEN_CHUNKS_KEY, extract_citations
from .compliance import extract_copy_blocks
from .deps import AgentDeps

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 300
MAX_DOCUMENT_CHARS = 7000
MAX_BRIEF_CHARS = 20000

# A saved brief must contain the sections mandated by the "Creative Brief Process" SOP.
REQUIRED_BRIEF_SECTIONS: dict[str, re.Pattern[str]] = {
    "Objective": re.compile(r"objective", re.I),
    "Audience": re.compile(r"audience|persona", re.I),
    "Key message": re.compile(r"key\s+message", re.I),
    "Proof points": re.compile(r"proof\s+points?", re.I),
    "Claims & disclaimers": re.compile(r"claims|disclaimer", re.I),
    "Hook ideas": re.compile(r"hooks?", re.I),
    "Format & placement": re.compile(r"format|placement", re.I),
    "Call to action": re.compile(r"call\s+to\s+action|\bcta\b", re.I),
    "Success metrics & test plan": re.compile(r"success\s+metrics?|test\s+plan|kpi", re.I),
}


def _is_english_like(text: str) -> bool:
    """True when the text has no non-ASCII letters (e.g. Vietnamese diacritics, CJK)."""
    return not any(ch.isalpha() and ord(ch) > 127 for ch in text)


def _remember(tool_context: ToolContext, hits: list[dict[str, Any]]) -> None:
    """Record retrieved chunk ids in session state so citations can be verified later."""
    seen = dict(tool_context.state.get(SEEN_CHUNKS_KEY) or {})
    for h in hits:
        seen[h["chunk_id"]] = {
            "document": h["document"],
            "section": h["section"],
            "status": h["status"],
        }
    if len(seen) > 400:  # keep the state bounded on very long sessions
        seen = dict(list(seen.items())[-400:])
    tool_context.state[SEEN_CHUNKS_KEY] = seen


def _hit_to_result(hit: SearchHit) -> dict[str, Any]:
    c = hit.chunk
    result: dict[str, Any] = {
        "chunk_id": c.chunk_id,
        "document": c.title,
        "doc_id": c.doc_id,
        "doc_type": c.doc_type,
        "section": c.heading_path,
        "status": c.status,
        "last_updated": c.updated,
        "owner": c.owner,
        "relevance": round(hit.relevance, 3),
        "text": c.text,
    }
    if c.status == "deprecated":
        result["warning"] = "DEPRECATED document: outdated, do not present as current policy/data."
    return result


def build_tools(deps: AgentDeps) -> list[Callable[..., Any]]:
    settings, store, engine = deps.settings, deps.store, deps.compliance

    async def search_knowledge_base(
        query: str,
        tool_context: ToolContext,
        doc_types: list[str] | None = None,
        include_deprecated: bool = False,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search Lumera's internal knowledge base (hybrid semantic + keyword search).

        Call this BEFORE answering any question about Lumera products, prices, claims, brand voice,
        SOPs, policies, customer research or past creative results. You may call it several times
        with different focused queries.

        Args:
            query: A focused search query WRITTEN IN ENGLISH, even if the user wrote in another
                language (translate it first). Use concrete nouns, e.g. "Glow Serum approved
                benefit claims" rather than a full conversational sentence. Max 300 characters.
            doc_types: Optional filter. Any of: product, brand, compliance, sop, research,
                learnings. Omit to search everything.
            include_deprecated: Leave False. Set True only when the user explicitly asks about
                historical/archived information; deprecated results are outdated.
            top_k: Number of passages to return (1-8). Default 5.

        Returns:
            A dict with `status` ("ok", "no_relevant_results" or "error"), `confidence` ("high",
            "medium", "none"), and `results`: passages each having `chunk_id` (the citation key you
            must use as [chunk_id]), `document`, `section`, `status`, `owner`, `relevance` (0-1) and
            `text`. If status is "no_relevant_results" the knowledge base has nothing relevant:
            tell the user so, do not answer from general knowledge.
        """
        query = (query or "").strip()
        if not query:
            return {"status": "error", "error_code": "EMPTY_QUERY", "message": "query must not be empty."}
        if len(query) > MAX_QUERY_CHARS:
            return {
                "status": "error",
                "error_code": "QUERY_TOO_LONG",
                "message": f"query exceeds {MAX_QUERY_CHARS} characters; make it shorter and more focused.",
            }
        if not _is_english_like(query):
            return {
                "status": "error",
                "error_code": "QUERY_NOT_ENGLISH",
                "message": "The knowledge base is in English. Translate the query to English and call "
                "search_knowledge_base again. Do not mention this error to the user.",
            }
        bad_types = [t for t in (doc_types or []) if t not in DOC_TYPES]
        if bad_types:
            return {
                "status": "error",
                "error_code": "INVALID_DOC_TYPE",
                "message": f"Unknown doc_types {bad_types}. Valid values: {list(DOC_TYPES)}.",
            }

        top_k = max(1, min(int(top_k), 8))
        try:
            hits = await asyncio.to_thread(
                store.search,
                query,
                top_k=top_k,
                candidates=settings.retrieval_candidates,
                doc_types=doc_types or None,
                include_deprecated=include_deprecated,
            )
        except Exception:  # noqa: BLE001 - never leak infra errors to the model
            logger.exception("knowledge base search failed")
            return {
                "status": "error",
                "error_code": "SEARCH_UNAVAILABLE",
                "message": "The knowledge base is temporarily unavailable. Tell the user you cannot "
                "answer reliably right now and to retry shortly. Do not answer from memory.",
            }

        tool_context.state["temp:kb_searches"] = int(tool_context.state.get("temp:kb_searches", 0)) + 1
        best = max((h.relevance for h in hits), default=0.0)
        relevant = [h for h in hits if h.relevance >= settings.relevance_min]

        if not relevant:
            confidence = "none"
        elif best >= settings.relevance_high:
            confidence = "high"
        else:
            confidence = "medium"
        # Track the strongest evidence seen this turn (used by the output guardrail / UI badge).
        order = {"none": 0, "medium": 1, "high": 2}
        prev = tool_context.state.get("temp:best_confidence", "none")
        if order[confidence] > order[prev]:
            tool_context.state["temp:best_confidence"] = confidence

        if not relevant:
            return {
                "status": "no_relevant_results",
                "confidence": "none",
                "query": query,
                "results": [],
                "closest_documents": sorted({h.chunk.title for h in hits})[:3],
                "guidance": "Nothing in the knowledge base is relevant enough to this query. If rephrasing "
                "once with different keywords also fails, tell the user the knowledge base does not "
                "contain this information. Do NOT answer from general knowledge or guess.",
            }

        results = [_hit_to_result(h) for h in relevant]
        _remember(tool_context, results)
        guidance = (
            "Use only these passages as evidence and cite each claim as [chunk_id]. Passages can be "
            "topically related without actually answering the question: if they do not directly answer "
            "it, say what is missing instead of stretching them."
        )
        if confidence == "medium":
            guidance = "Evidence is moderate; state any uncertainty. " + guidance
        return {
            "status": "ok",
            "confidence": confidence,
            "query": query,
            "results": results,
            "guidance": guidance,
        }

    async def get_document(doc_id: str, tool_context: ToolContext) -> dict[str, Any]:
        """Fetch the full text of one knowledge-base document, section by section.

        Use this when search snippets are not enough, for example to read an entire SOP template,
        a whole product fact sheet, or the full compliance policy before writing copy.

        Args:
            doc_id: The document id, e.g. "product-glow-serum" or "compliance-claims-policy"
                (see `doc_id` / `chunk_id` prefix in search results, or call list_knowledge_sources).

        Returns:
            A dict with `status`, document metadata and `sections`, each with a citable `chunk_id`.
        """
        try:
            chunks = await asyncio.to_thread(store.get_document_chunks, doc_id.strip())
        except Exception:  # noqa: BLE001
            logger.exception("get_document failed")
            return {"status": "error", "error_code": "SEARCH_UNAVAILABLE", "message": "Knowledge base unavailable."}
        if not chunks:
            return {
                "status": "not_found",
                "message": f"No document with doc_id '{doc_id}'. Call list_knowledge_sources to see valid ids.",
            }
        sections: list[dict[str, Any]] = []
        used = 0
        truncated = False
        for c in chunks:
            if used + len(c.text) > MAX_DOCUMENT_CHARS:
                truncated = True
                break
            used += len(c.text)
            sections.append(
                {
                    "chunk_id": c.chunk_id,
                    "document": c.title,
                    "section": c.heading_path,
                    "status": c.status,
                    "owner": c.owner,
                    "text": c.text,
                }
            )
        _remember(tool_context, sections)
        tool_context.state["temp:kb_searches"] = int(tool_context.state.get("temp:kb_searches", 0)) + 1
        tool_context.state["temp:best_confidence"] = "high"
        head = chunks[0]
        return {
            "status": "ok",
            "doc_id": head.doc_id,
            "title": head.title,
            "doc_type": head.doc_type,
            "doc_status": head.status,
            "last_updated": head.updated,
            "owner": head.owner,
            "truncated": truncated,
            "sections": sections,
        }

    async def list_knowledge_sources(doc_type: str | None = None) -> dict[str, Any]:
        """List the documents available in the knowledge base (the catalogue).

        Use this when the user asks what information exists, or when you need a valid `doc_id`.

        Args:
            doc_type: Optional filter: product, brand, compliance, sop, research or learnings.

        Returns:
            A dict with `documents`: doc_id, title, doc_type, status (active/deprecated), updated, owner.
        """
        try:
            docs = await asyncio.to_thread(store.list_documents)
        except Exception:  # noqa: BLE001
            logger.exception("list_knowledge_sources failed")
            return {"status": "error", "error_code": "SEARCH_UNAVAILABLE", "message": "Knowledge base unavailable."}
        if doc_type:
            docs = [d for d in docs if d["doc_type"] == doc_type]
        return {"status": "ok", "count": len(docs), "documents": docs}

    async def check_compliance(copy_text: str) -> dict[str, Any]:
        """Run Lumera's deterministic advertising-compliance scan on customer-facing copy.

        ALWAYS call this on every piece of marketing copy you drafted (headlines, primary text,
        hooks, scripts, captions, CTAs) BEFORE showing it to the user. If `passed` is false, revise
        the copy using the `suggestion` of each violation (and add any missing disclaimer), then
        check again. Never present copy that did not pass.

        Args:
            copy_text: All the drafted copy concatenated, exactly as it will be published.

        Returns:
            `passed` (bool), `violations` (rule_id, severity block|warn, matched text, suggestion,
            policy_section) and `missing_disclaimers`.
        """
        if not copy_text.strip():
            return {"status": "error", "error_code": "EMPTY_COPY", "message": "copy_text is empty."}
        report = engine.scan(copy_text[:MAX_BRIEF_CHARS])
        return {
            "status": "ok",
            "passed": report.passed,
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "severity": v.severity,
                    "matched_text": v.match,
                    "suggestion": v.suggestion,
                    "policy_section": v.policy_section,
                }
                for v in report.violations
            ],
            "missing_disclaimers": [
                {"id": d.id, "required_any_of": d.required_any_of, "policy_section": d.policy_section}
                for d in report.missing_disclaimers
            ],
            "next_step": "Revise and re-check." if not report.passed else "Copy is compliant.",
        }

    async def save_creative_brief(
        title: str, brief_markdown: str, tool_context: ToolContext
    ) -> dict[str, Any]:
        """Validate and save a finished creative brief so the team can download it.

        Call this once, after you have written the complete brief and its copy passed
        `check_compliance`. The tool re-validates everything and REFUSES to save if the brief is
        incomplete, cites sources you never retrieved, or its copy blocks violate compliance rules.
        If it refuses, fix exactly what the error lists and call it again (at most twice).

        Args:
            title: Short brief title, e.g. "Glow Serum — problem-first UGC test".
            brief_markdown: The full brief in markdown following the SOP template, with every proof
                point cited as [chunk_id] and all customer-facing copy inside ```copy fences.

        Returns:
            On success: `status` "saved", `brief_id` and `download_path`. On refusal: `status`
            "rejected" with `problems` describing what to fix.
        """
        problems: list[str] = []
        title = (title or "").strip()
        body = (brief_markdown or "").strip()
        if not (3 <= len(title) <= 120):
            problems.append("title must be 3-120 characters.")
        if len(body) < 400:
            problems.append("brief is too short to be a usable creative brief.")
        if len(body) > MAX_BRIEF_CHARS:
            problems.append(f"brief exceeds {MAX_BRIEF_CHARS} characters.")

        missing = [name for name, pat in REQUIRED_BRIEF_SECTIONS.items() if not pat.search(body)]
        if missing:
            problems.append(f"missing required sections from the Creative Brief SOP: {missing}.")

        cited = extract_citations(body)
        seen = tool_context.state.get(SEEN_CHUNKS_KEY) or {}
        unverified = [c for c in cited if c not in seen]
        if not cited:
            problems.append("no citations: every proof point must cite its source as [chunk_id].")
        if unverified:
            problems.append(f"citations not retrieved in this session (possible fabrication): {unverified}.")

        copy_blocks = extract_copy_blocks(body)
        if not copy_blocks:
            problems.append("no ```copy blocks found: put all customer-facing copy in ```copy fences.")
        else:
            report = engine.scan("\n\n".join(b[2] for b in copy_blocks))
            for v in report.blocking:
                problems.append(
                    f"compliance rule '{v.rule_id}' violated by \"{v.match}\" ({v.policy_section}). {v.suggestion}"
                )
            for d in report.missing_disclaimers:
                problems.append(f"copy must include one of {d.required_any_of} ({d.description})")

        if problems:
            return {"status": "rejected", "problems": problems}

        brief_id = f"{datetime.now(UTC):%Y%m%d}-{slugify(title)[:48]}-{uuid.uuid4().hex[:6]}"
        out_dir = settings.output_path
        out_dir.mkdir(parents=True, exist_ok=True)
        header = (
            f"---\ntitle: {title}\nbrief_id: {brief_id}\n"
            f"created: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
            f"compliance_rules_version: {engine.version}\nsources: {cited}\n---\n\n"
        )
        await asyncio.to_thread((out_dir / f"{brief_id}.md").write_text, header + body, "utf-8")
        logger.info("saved creative brief %s (%d chars, %d sources)", brief_id, len(body), len(cited))
        return {
            "status": "saved",
            "brief_id": brief_id,
            "download_path": f"/api/briefs/{brief_id}",
            "sources_cited": len(cited),
        }

    return [
        search_knowledge_base,
        get_document,
        list_knowledge_sources,
        check_compliance,
        save_creative_brief,
    ]
