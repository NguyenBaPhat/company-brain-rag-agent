import pytest
from qdrant_client import QdrantClient

from company_brain.knowledge.chunker import chunk_documents
from company_brain.knowledge.ingest import build_embedder
from company_brain.knowledge.loader import load_documents
from company_brain.knowledge.store import KnowledgeStore, KnowledgeStoreError

pytestmark = pytest.mark.slow


def test_search_finds_expected_document(store):
    hits = store.search("Glow Serum usage instructions how to apply", top_k=3)
    assert hits[0].chunk.chunk_id == "product-glow-serum#how-to-use"
    assert hits[0].relevance > 0.5


def test_off_domain_query_has_low_relevance(store, settings):
    hits = store.search("what is the capital of France", top_k=5)
    assert max(h.relevance for h in hits) < settings.relevance_min


def test_deprecated_documents_hidden_by_default(store):
    default = store.search("Glow Serum price dermatologist recommended 2024", top_k=10)
    assert all(h.chunk.status == "active" for h in default)
    with_old = store.search("Glow Serum price 2024 archived", top_k=10, include_deprecated=True)
    assert any(h.chunk.status == "deprecated" for h in with_old)


def test_doc_type_filter(store):
    hits = store.search("what are we not allowed to claim", top_k=5, doc_types=["compliance"])
    assert hits and {h.chunk.doc_type for h in hits} == {"compliance"}


def test_get_document_chunks_ordered_and_get_chunk(store):
    chunks = store.get_document_chunks("product-glow-serum")
    assert [c.order for c in chunks] == sorted(c.order for c in chunks)
    assert store.get_chunk("product-glow-serum#overview").doc_id == "product-glow-serum"
    assert store.get_chunk("nope#nope") is None


def test_list_documents_catalogue(store):
    docs = {d["doc_id"]: d for d in store.list_documents()}
    assert docs["archive-learnings-2024"]["status"] == "deprecated"
    assert docs["product-glow-serum"]["owner"] == "Product Marketing"


def test_sync_is_idempotent_and_removes_stale_chunks(settings):
    kb = KnowledgeStore(QdrantClient(":memory:"), build_embedder(settings), "idem")
    chunks = chunk_documents(load_documents(settings.kb_path))
    first = kb.sync_chunks(chunks)
    assert first.added == len(chunks) and first.unchanged == 0
    second = kb.sync_chunks(chunks)
    assert second.added == second.updated == second.deleted == 0 and second.unchanged == len(chunks)

    changed = [c.model_copy(update={"content_hash": "changed"}) if i == 0 else c for i, c in enumerate(chunks)]
    third = kb.sync_chunks(changed[:-3])
    assert third.updated == 1 and third.deleted == 3
    assert kb.count() == len(chunks) - 3


def test_model_mismatch_is_detected(settings):
    client = QdrantClient(":memory:")
    kb = KnowledgeStore(client, build_embedder(settings), "fp")
    kb.ensure_collection()
    other_settings = settings.model_copy(update={"dense_model": "BAAI/bge-small-en-v1.5"})
    other = KnowledgeStore(client, build_embedder(other_settings), "fp")
    with pytest.raises(KnowledgeStoreError, match="--recreate"):
        other.ensure_collection()
