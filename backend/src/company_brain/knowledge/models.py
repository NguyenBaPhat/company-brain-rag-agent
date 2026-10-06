"""Domain models for the knowledge base."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["product", "brand", "compliance", "sop", "research", "learnings"]
DocStatus = Literal["active", "deprecated"]

DOC_TYPES: tuple[str, ...] = ("product", "brand", "compliance", "sop", "research", "learnings")


class Document(BaseModel):
    """A source document with validated front matter."""

    doc_id: str
    title: str
    doc_type: DocType
    status: DocStatus = "active"
    updated: date
    owner: str = "unknown"
    tags: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    supersedes: str | None = None
    path: str
    body: str


class Chunk(BaseModel):
    """A retrievable passage. `chunk_id` doubles as the human-readable citation key."""

    chunk_id: str
    doc_id: str
    title: str
    doc_type: DocType
    status: DocStatus
    updated: str
    owner: str = "unknown"
    section: str
    heading_path: str
    order: int
    text: str
    tags: list[str] = Field(default_factory=list)
    content_hash: str = ""

    @property
    def embed_text(self) -> str:
        """Contextualised text that is embedded (doc title + heading path + body).

        Prefixing the title/heading is a cheap form of "contextual retrieval": a chunk that
        only says "Apply 2 pumps every morning" still knows it belongs to Glow Serum.
        """
        return f"{self.title} — {self.heading_path}\n{self.text}"


class SearchHit(BaseModel):
    chunk: Chunk
    relevance: float = Field(description="Dense cosine similarity between query and chunk, 0-1.")
    fusion_score: float = Field(description="Hybrid (dense+BM25) reciprocal-rank-fusion score.")
    rank: int
