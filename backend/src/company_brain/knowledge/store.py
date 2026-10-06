"""Qdrant-backed hybrid (dense + BM25) knowledge store.

Retrieval strategy
------------------
1. Run a dense (semantic) query and a sparse BM25 (lexical) query in parallel inside Qdrant.
2. Fuse the two ranked lists with Reciprocal Rank Fusion (server-side, one round trip).
3. Re-score every fused hit with the *dense cosine similarity* between the query and the chunk.
   RRF scores are rank-based and therefore not comparable across queries; cosine is, which makes
   it usable as an absolute "is this relevant at all?" signal for the not-enough-information gate.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from ..config import Settings
from .embeddings import Embedder, cosine
from .models import Chunk, SearchHit

logger = logging.getLogger(__name__)

_ID_NAMESPACE = uuid.UUID("6f1b6c3e-52f0-4d0e-9a53-0c0b7a4c6d11")
DENSE = "dense"
SPARSE = "bm25"


class KnowledgeStoreError(RuntimeError):
    pass


@dataclass
class IngestStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0

    def __str__(self) -> str:
        return (
            f"added={self.added} updated={self.updated} "
            f"unchanged={self.unchanged} deleted={self.deleted}"
        )


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def create_qdrant_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, timeout=30)
    if settings.qdrant_path:
        path = settings.resolve(settings.qdrant_path)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    return QdrantClient(":memory:")


class KnowledgeStore:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection

    # ------------------------------------------------------------------ admin
    def _fingerprint(self) -> dict[str, str]:
        return {
            "dense_model": self.embedder.dense_model_name,
            "sparse_model": self.embedder.sparse_model_name,
        }

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection)
        if self.collection_exists():
            meta = (self.client.get_collection(self.collection).config.metadata) or {}
            if meta and meta != self._fingerprint():
                raise KnowledgeStoreError(
                    f"Collection '{self.collection}' was built with {meta}, but the configured "
                    f"models are {self._fingerprint()}. Re-run ingestion with --recreate."
                )
            return
        logger.info("Creating collection %s (dim=%d)", self.collection, self.embedder.dim)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: qm.VectorParams(size=self.embedder.dim, distance=qm.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            metadata=self._fingerprint(),
        )
        with warnings.catch_warnings():  # embedded mode ignores payload indexes (server mode uses them)
            warnings.simplefilter("ignore")
            for field in ("doc_id", "doc_type", "status"):
                self.client.create_payload_index(
                    self.collection, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD
                )

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - health probe must never raise
            return False

    # ----------------------------------------------------------------- ingest
    def _existing_hashes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["chunk_id", "content_hash"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                out[payload.get("chunk_id", "")] = payload.get("content_hash", "")
            if offset is None:
                return out

    def sync_chunks(self, chunks: list[Chunk], *, batch_size: int = 32) -> IngestStats:
        """Idempotent sync: embed + upsert only new/changed chunks, delete chunks that vanished."""
        self.ensure_collection()
        existing = self._existing_hashes()
        stats = IngestStats()

        todo: list[Chunk] = []
        for c in chunks:
            if c.chunk_id not in existing:
                stats.added += 1
                todo.append(c)
            elif existing[c.chunk_id] != c.content_hash:
                stats.updated += 1
                todo.append(c)
            else:
                stats.unchanged += 1

        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            texts = [c.embed_text for c in batch]
            dense = self.embedder.dense_documents(texts)
            sparse = self.embedder.sparse_documents(texts)
            points = [
                qm.PointStruct(
                    id=point_id(c.chunk_id),
                    vector={
                        DENSE: d,
                        SPARSE: qm.SparseVector(indices=s.indices, values=s.values),
                    },
                    payload=c.model_dump(),
                )
                for c, d, s in zip(batch, dense, sparse, strict=True)
            ]
            self.client.upsert(self.collection, points=points, wait=True)

        stale = sorted(set(existing) - {c.chunk_id for c in chunks})
        if stale:
            self.client.delete(
                self.collection,
                points_selector=qm.PointIdsList(points=[point_id(cid) for cid in stale]),
                wait=True,
            )
            stats.deleted = len(stale)
        return stats

    # ----------------------------------------------------------------- search
    @staticmethod
    def _filter(doc_types: list[str] | None, include_deprecated: bool) -> qm.Filter | None:
        must: list[qm.Condition] = []
        if not include_deprecated:
            must.append(qm.FieldCondition(key="status", match=qm.MatchValue(value="active")))
        if doc_types:
            must.append(qm.FieldCondition(key="doc_type", match=qm.MatchAny(any=doc_types)))
        return qm.Filter(must=must) if must else None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidates: int = 20,
        doc_types: list[str] | None = None,
        include_deprecated: bool = False,
    ) -> list[SearchHit]:
        if not query.strip():
            return []
        flt = self._filter(doc_types, include_deprecated)
        dense_q = self.embedder.dense_query(query)
        sparse_q = self.embedder.sparse_query(query)
        result = self.client.query_points(
            self.collection,
            prefetch=[
                qm.Prefetch(query=dense_q, using=DENSE, limit=candidates, filter=flt),
                qm.Prefetch(
                    query=qm.SparseVector(indices=sparse_q.indices, values=sparse_q.values),
                    using=SPARSE,
                    limit=candidates,
                    filter=flt,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=[DENSE],
        )
        hits: list[SearchHit] = []
        for rank, p in enumerate(result.points, start=1):
            vec = p.vector[DENSE] if isinstance(p.vector, dict) else p.vector
            hits.append(
                SearchHit(
                    chunk=Chunk(**(p.payload or {})),
                    relevance=max(0.0, cosine(dense_q, vec)),
                    fusion_score=float(p.score),
                    rank=rank,
                )
            )
        return hits

    # ------------------------------------------------------------ doc access
    def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
        points, _ = self.client.scroll(
            self.collection, scroll_filter=flt, limit=500, with_payload=True, with_vectors=False
        )
        return sorted((Chunk(**(p.payload or {})) for p in points), key=lambda c: c.order)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        pts = self.client.retrieve(
            self.collection, ids=[point_id(chunk_id)], with_payload=True, with_vectors=False
        )
        return Chunk(**(pts[0].payload or {})) if pts else None

    def list_documents(self) -> list[dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["doc_id", "title", "doc_type", "status", "updated", "tags", "owner"],
                with_vectors=False,
            )
            for p in points:
                pl = p.payload or {}
                d = docs.setdefault(
                    pl["doc_id"],
                    {k: pl.get(k) for k in ("doc_id", "title", "doc_type", "status", "updated", "tags", "owner")}
                    | {"chunks": 0},
                )
                d["chunks"] += 1
            if offset is None:
                break
        return sorted(docs.values(), key=lambda d: (d["doc_type"], d["doc_id"]))
