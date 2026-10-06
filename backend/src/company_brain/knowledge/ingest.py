"""Ingestion pipeline: markdown KB -> chunks -> embeddings -> Qdrant.  Run with `cb-ingest`."""

from __future__ import annotations

import argparse
import logging
import time

from ..config import Settings, get_settings
from ..logging_config import setup_logging
from .chunker import chunk_documents
from .embeddings import Embedder
from .loader import load_documents
from .store import IngestStats, KnowledgeStore, create_qdrant_client

logger = logging.getLogger(__name__)


def build_embedder(settings: Settings) -> Embedder:
    return Embedder(
        settings.dense_model,
        settings.sparse_model,
        cache_dir=str(settings.embedding_cache_path),
    )


def build_store(settings: Settings, embedder: Embedder | None = None) -> KnowledgeStore:
    return KnowledgeStore(
        create_qdrant_client(settings),
        embedder or build_embedder(settings),
        settings.qdrant_collection,
    )


def ingest(settings: Settings, store: KnowledgeStore | None = None, recreate: bool = False) -> IngestStats:
    store = store or build_store(settings)
    started = time.perf_counter()
    docs = load_documents(settings.kb_path)
    chunks = chunk_documents(docs)
    logger.info("Chunked %d documents into %d chunks", len(docs), len(chunks))
    store.ensure_collection(recreate=recreate)
    stats = store.sync_chunks(chunks)
    logger.info("Ingest done in %.1fs: %s (collection now holds %d)",
                time.perf_counter() - started, stats, store.count())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the knowledge base into Qdrant.")
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the collection.")
    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    stats = ingest(settings, recreate=args.recreate)
    print(f"Ingest complete: {stats}")


if __name__ == "__main__":
    main()
