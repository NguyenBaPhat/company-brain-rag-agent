"""A deterministic, offline stand-in for Gemini (``LLM_MODE=mock``).

It exists so the whole stack (ADK agent loop, tools, guardrails, WebSocket streaming and the React
UI) can be exercised end-to-end in CI and on a laptop without an API key. It is NOT intelligent:
it performs extractive question answering over whatever the real tools return and labels every
answer as mock output. Two scripted demo flows (``/demo violation`` and ``/demo brief``) drive the
guardrail and the brief-saving paths.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

_VI_GLOSSARY = {
    "cách dùng": "how to use",
    "cách sử dụng": "how to use",
    "giá": "price",
    "thành phần": "ingredients",
    "chính sách": "policy",
    "hoàn trả": "returns",
    "vận chuyển": "shipping",
    "khách hàng": "customer",
    "quảng cáo": "ads",
    "kem": "cream",
}

_TEXTS = {
    "en": {
        "prefix": "_(mock LLM — offline test double)_ ",
        "found": "Here is what the knowledge base says:",
        "none": "I could not find this in the knowledge base, so I cannot answer it reliably.",
        "closest": "Closest documents:",
        "saved": "Brief saved. Brief id:",
    },
    "vi": {
        "prefix": "_(mock LLM — bản giả lập offline)_ ",
        "found": "Đây là những gì knowledge base cho biết:",
        "none": "Tôi không tìm thấy thông tin này trong knowledge base nên không thể trả lời chắc chắn.",
        "closest": "Tài liệu gần nhất:",
        "saved": "Đã lưu brief. Mã brief:",
    },
}


def _to_english(text: str) -> str:
    low = text.lower()
    for vi, en in _VI_GLOSSARY.items():
        low = low.replace(vi, en)
    low = low.replace("đ", "d")
    return unicodedata.normalize("NFKD", low).encode("ascii", "ignore").decode()[:280].strip()


def _call(name: str, args: dict[str, Any]) -> types.Content:
    return types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])


def _text(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


def _first_sentence(text: str, limit: int = 220) -> str:
    flat = re.sub(r"\s+", " ", re.sub(r"[#*`>]", "", text)).strip()
    m = re.match(r"(.+?[.!?])(\s|$)", flat)
    out = m.group(1) if m else flat
    return out[:limit]


class MockLlm(BaseLlm):
    model: str = "mock-extractive"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        system = str(getattr(llm_request.config, "system_instruction", "") or "")
        lang = "vi" if "Vietnamese" in system else "en"
        contents = llm_request.contents or []
        last_parts = contents[-1].parts if contents else []
        responses = [p.function_response for p in (last_parts or []) if p.function_response]
        user_text = next(
            (
                " ".join(p.text for p in c.parts if p.text)
                for c in reversed(contents)
                if c.role == "user" and c.parts and any(p.text for p in c.parts)
            ),
            "",
        )

        reply = self._decide(user_text, responses, lang)
        text = "".join(p.text or "" for p in (reply.parts or []))
        if stream and text:
            words = text.split(" ")
            for i in range(0, len(words), 3):
                await asyncio.sleep(0.015)
                chunk = " ".join(words[i : i + 3]) + (" " if i + 3 < len(words) else "")
                yield LlmResponse(content=_text(chunk), partial=True)
        yield LlmResponse(content=reply, partial=False, turn_complete=True)

    # ------------------------------------------------------------------ policy
    def _decide(self, user_text: str, responses: list[Any], lang: str) -> types.Content:
        t = _TEXTS[lang]
        demo = user_text.strip().lower()
        if not responses:  # fresh user turn -> retrieve first
            query = "Glow Serum approved benefit claims" if demo.startswith("/demo") else user_text[:290]
            return _call("search_knowledge_base", {"query": query})

        resp = dict(responses[0].response or {})
        name = responses[0].name
        if name == "search_knowledge_base":
            if resp.get("error_code") == "QUERY_NOT_ENGLISH":
                return _call("search_knowledge_base", {"query": _to_english(user_text)})
            if demo.startswith("/demo violation"):
                return _text(
                    "Here is a draft headline:\n\n```copy\nClinically proven to cure acne in 7 days.\n```\n\n"
                    "This is a scripted guardrail demonstration."
                )
            if demo.startswith("/demo brief") and resp.get("status") == "ok":
                return _call("save_creative_brief", self._demo_brief(resp["results"]))
            if resp.get("status") != "ok":
                near = ", ".join(resp.get("closest_documents", [])) or "—"
                return _text(f"{t['prefix']}{t['none']} {t['closest']} {near}")
            lines = [
                f"- {_first_sentence(r['text'])} [{r['chunk_id']}]" for r in resp["results"][:3]
            ]
            return _text(f"{t['prefix']}{t['found']}\n\n" + "\n".join(lines))
        if name == "save_creative_brief":
            if resp.get("status") == "saved":
                return _text(f"{t['prefix']}{t['saved']} `{resp['brief_id']}` — {resp['download_path']}")
            return _text(f"{t['prefix']}Brief rejected: {resp.get('problems')}")
        return _text(f"{t['prefix']}{t['none']}")

    @staticmethod
    def _demo_brief(results: list[dict[str, Any]]) -> dict[str, str]:
        cites = " ".join(f"[{r['chunk_id']}]" for r in results[:2])
        body = f"""## Objective
Lower CAC for Glow Serum cold audiences below $35 {cites}.

## Audience
Sensitive-Skin Sara, 28-38, cold audience.

## Key message
Glow, without the stinging.

## Proof points
- Lightweight, fragrance-free gel serum {cites}.

## Claims, disclaimers and prohibited claims
Approved: "helps visibly brighten the look of dull skin". Prohibited: disease-treatment claims, endorsements.

## Hook ideas
```copy
Hook 1: My skin barrier was wrecked, so I simplified.
Hook 2: Glow, without the stinging.
Hook 3: Three steps. Zero drama.
```

## Format and placement
15-second vertical UGC video for Reels and Stories.

## Call to action
```copy
Try it for 30 days.
```

## Success metrics and test plan
Primary KPI: CAC. One variable (hook), minimum $1,500 per variant, kill when frequency exceeds 3.5.
"""
        return {"title": "Glow Serum — mock demo brief", "brief_markdown": body}
