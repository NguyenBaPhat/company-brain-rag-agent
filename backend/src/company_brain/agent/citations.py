"""Citation helpers. A citation is a chunk id in square brackets: ``[product-glow-serum#how-to-use]``."""

from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9_-]*#[a-z0-9][a-z0-9_-]*)\]")

SEEN_CHUNKS_KEY = "kb_seen_chunks"  # session state: chunk_id -> {document, section, status}


def extract_citations(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in CITATION_RE.finditer(text):
        seen.setdefault(m.group(1))
    return list(seen)


def replace_unverified(text: str, unverified: set[str]) -> str:
    """Neutralise citations that do not correspond to any chunk the agent actually retrieved."""
    if not unverified:
        return text
    return CITATION_RE.sub(lambda m: "[unverified]" if m.group(1) in unverified else m.group(0), text)
