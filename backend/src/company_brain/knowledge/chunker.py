"""Heading-aware markdown chunking.

Each H2/H3 section becomes one chunk (the natural unit of meaning in our SOPs, fact sheets and
research notes). Oversized sections are split on paragraph / bullet boundaries. Chunk ids are
stable, human-readable and double as citation keys: ``<doc_id>#<section-slug>``.
"""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document

MAX_WORDS = 220

_HEADING = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")


def slugify(text: str) -> str:
    text = re.sub(r"[`*_]", "", text.lower())
    text = re.sub(r"^\d+[.)]\s*", "", text)  # "1. Prohibited" -> "prohibited"
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "section"


def _split_long(text: str, max_words: int) -> list[str]:
    """Split on blank lines / bullet lines so no part exceeds ``max_words`` (greedy)."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    pieces: list[str] = []
    for block in blocks:
        if len(block.split()) <= max_words:
            pieces.append(block)
            continue
        # A long block (e.g. a big bullet list): fall back to line-level splitting.
        pieces.extend(line for line in block.splitlines() if line.strip())

    parts: list[str] = []
    current: list[str] = []
    words = 0
    for piece in pieces:
        n = len(piece.split())
        if current and words + n > max_words:
            parts.append("\n\n".join(current))
            current, words = [], 0
        current.append(piece)
        words += n
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_document(doc: Document, max_words: int = MAX_WORDS) -> list[Chunk]:
    sections: list[tuple[str, str, list[str]]] = []  # (heading_path, leaf_heading, lines)
    h2 = ""
    current_path, current_leaf = "Introduction", "Introduction"
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_path, current_leaf, [body]))

    for line in doc.body.splitlines():
        m = _HEADING.match(line)
        if not m:
            current_lines.append(line)
            continue
        level, heading = len(m.group(1)), m.group(2).strip()
        if level == 1:  # document title; content below it (before the first H2) is the intro
            flush()
            current_lines = []
            current_path, current_leaf = "Introduction", "Introduction"
            continue
        flush()
        current_lines = []
        if level == 2:
            h2 = heading
            current_path, current_leaf = heading, heading
        else:
            current_path, current_leaf = (f"{h2} > {heading}" if h2 else heading), heading
    flush()

    chunks: list[Chunk] = []
    used: dict[str, int] = {}
    order = 0
    for heading_path, leaf, body_list in sections:
        parts = _split_long(body_list[0], max_words)
        for idx, part in enumerate(parts):
            base = f"{doc.doc_id}#{slugify(leaf)}"
            suffix = f"-p{idx + 1}" if len(parts) > 1 and idx > 0 else ""
            chunk_id = base + suffix
            if chunk_id in used:  # same heading repeated within one document
                used[chunk_id] += 1
                chunk_id = f"{chunk_id}-{used[chunk_id]}"
            else:
                used[chunk_id] = 1
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                title=doc.title,
                doc_type=doc.doc_type,
                status=doc.status,
                updated=doc.updated.isoformat(),
                owner=doc.owner,
                section=leaf,
                heading_path=heading_path,
                order=order,
                text=part.strip(),
                tags=doc.tags,
            )
            chunk.content_hash = hashlib.sha256(
                f"{chunk.chunk_id}|{chunk.status}|{chunk.updated}|{chunk.embed_text}".encode()
            ).hexdigest()[:16]
            chunks.append(chunk)
            order += 1
    return chunks


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    chunks = [c for d in docs for c in chunk_document(d)]
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be globally unique"
    return chunks
