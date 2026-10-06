from datetime import date

import pytest

from company_brain.knowledge.chunker import chunk_document, chunk_documents, slugify
from company_brain.knowledge.loader import KnowledgeBaseError, load_documents, parse_document
from company_brain.knowledge.models import Document


def _doc(body: str, doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, title="T", doc_type="sop", updated=date(2026, 1, 1), path="x.md", body=body)


def test_slugify_strips_numbering_and_symbols():
    assert slugify("1. Prohibited Disease & Treatment Claims") == "prohibited-disease-treatment-claims"
    assert slugify("???") == "section"


def test_sections_become_chunks_with_stable_citation_ids():
    chunks = chunk_document(_doc("# T\n\n## Overview\nHello.\n\n## How To Use\nApply it."))
    assert [c.chunk_id for c in chunks] == ["d1#overview", "d1#how-to-use"]
    assert chunks[1].heading_path == "How To Use"
    assert "T — How To Use" in chunks[1].embed_text  # contextual prefix


def test_h3_nested_under_h2_and_intro_kept():
    body = "# T\n\nIntro text here.\n\n## A\nalpha\n\n### Sub\nbeta"
    ids = {c.chunk_id: c for c in chunk_document(_doc(body))}
    assert "d1#introduction" in ids
    assert ids["d1#sub"].heading_path == "A > Sub"


def test_duplicate_headings_get_unique_ids():
    chunks = chunk_document(_doc("## Notes\none\n\n## Notes\ntwo"))
    assert len({c.chunk_id for c in chunks}) == 2


def test_long_sections_are_split_without_losing_text():
    para = " ".join(["word"] * 150)
    chunks = chunk_document(_doc(f"## Big\n{para}\n\n{para}\n\n{para}"))
    assert len(chunks) >= 2
    assert sum(len(c.text.split()) for c in chunks) == 450
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_content_hash_changes_when_text_changes():
    a = chunk_document(_doc("## A\none"))[0]
    b = chunk_document(_doc("## A\ntwo"))[0]
    assert a.chunk_id == b.chunk_id and a.content_hash != b.content_hash


def test_real_kb_loads_and_chunk_ids_are_globally_unique(settings):
    docs = load_documents(settings.kb_path)
    chunks = chunk_documents(docs)
    assert len(docs) >= 15 and len(chunks) >= 60
    assert {d.status for d in docs} == {"active", "deprecated"}


def test_missing_front_matter_raises(tmp_path):
    (tmp_path / "bad.md").write_text("# no front matter")
    with pytest.raises(KnowledgeBaseError, match="front matter"):
        parse_document(tmp_path / "bad.md", tmp_path)


def test_invalid_doc_type_raises(tmp_path):
    (tmp_path / "bad.md").write_text("---\ndoc_id: x\ntitle: X\ndoc_type: nonsense\nupdated: 2026-01-01\n---\nbody")
    with pytest.raises(KnowledgeBaseError, match="invalid front matter"):
        parse_document(tmp_path / "bad.md", tmp_path)


def test_duplicate_doc_id_raises(tmp_path):
    fm = "---\ndoc_id: same\ntitle: X\ndoc_type: sop\nupdated: 2026-01-01\n---\nbody"
    (tmp_path / "a.md").write_text(fm)
    (tmp_path / "b.md").write_text(fm)
    with pytest.raises(KnowledgeBaseError, match="Duplicate"):
        load_documents(tmp_path)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="not found"):
        load_documents(tmp_path / "nope")
