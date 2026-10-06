"""Load and validate markdown documents (YAML front matter + body) from the KB directory."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import Document

logger = logging.getLogger(__name__)

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base content is invalid (fail fast at ingest time)."""


def parse_document(path: Path, root: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise KnowledgeBaseError(f"{path}: missing YAML front matter")
    try:
        meta = yaml.safe_load(match.group(1)) or {}
        if not isinstance(meta, dict):
            raise KnowledgeBaseError(f"{path}: front matter must be a mapping")
        return Document(path=str(path.relative_to(root)), body=match.group(2).strip(), **meta)
    except (yaml.YAMLError, ValidationError, TypeError) as exc:
        raise KnowledgeBaseError(f"{path}: invalid front matter: {exc}") from exc


def load_documents(kb_dir: Path) -> list[Document]:
    """Load every `*.md` file under ``kb_dir``. Raises on invalid or duplicated documents."""
    if not kb_dir.is_dir():
        raise KnowledgeBaseError(f"Knowledge base directory not found: {kb_dir}")

    documents: list[Document] = []
    seen: dict[str, str] = {}
    for path in sorted(kb_dir.rglob("*.md")):
        doc = parse_document(path, kb_dir)
        if doc.doc_id in seen:
            raise KnowledgeBaseError(f"Duplicate doc_id '{doc.doc_id}' in {path} and {seen[doc.doc_id]}")
        seen[doc.doc_id] = str(path)
        documents.append(doc)

    ids = set(seen)
    for doc in documents:
        for ref in (doc.superseded_by, doc.supersedes):
            if ref and ref not in ids:
                logger.warning("doc %s references unknown doc_id %s", doc.doc_id, ref)

    logger.info("Loaded %d documents from %s", len(documents), kb_dir)
    return documents
