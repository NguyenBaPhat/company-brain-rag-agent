"""Local embedding models (no API key, no network at query time once cached).

* Dense: multilingual sentence embeddings — lets a Vietnamese question match English documents.
* Sparse: BM25 term weights — keeps exact-match power for product names, numbers and policy terms.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class Embedder:
    """Lazy, thread-safe wrapper around fastembed dense + sparse models."""

    def __init__(self, dense_model: str, sparse_model: str, cache_dir: str | None = None) -> None:
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self._cache_dir = cache_dir
        self._dense: TextEmbedding | None = None
        self._sparse: SparseTextEmbedding | None = None
        self._lock = threading.Lock()
        self._dim: int | None = None

    # -- lazy loading ------------------------------------------------------
    def _get_dense(self) -> TextEmbedding:
        if self._dense is None:
            with self._lock:
                if self._dense is None:
                    logger.info("Loading dense model %s", self.dense_model_name)
                    self._dense = TextEmbedding(self.dense_model_name, cache_dir=self._cache_dir)
        return self._dense

    def _get_sparse(self) -> SparseTextEmbedding:
        if self._sparse is None:
            with self._lock:
                if self._sparse is None:
                    logger.info("Loading sparse model %s", self.sparse_model_name)
                    self._sparse = SparseTextEmbedding(
                        self.sparse_model_name, cache_dir=self._cache_dir
                    )
        return self._sparse

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(next(iter(self._get_dense().embed(["dimension probe"]))))
        return self._dim

    # -- dense -------------------------------------------------------------
    def dense_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._get_dense().passage_embed(list(texts))]

    def dense_query(self, text: str) -> list[float]:
        return next(iter(self._get_dense().query_embed(text))).tolist()

    # -- sparse ------------------------------------------------------------
    def sparse_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        return [
            SparseVec(v.indices.tolist(), v.values.tolist())
            for v in self._get_sparse().embed(list(texts))
        ]

    def sparse_query(self, text: str) -> SparseVec:
        v = next(iter(self._get_sparse().query_embed(text)))
        return SparseVec(v.indices.tolist(), v.values.tolist())


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / denom) if denom else 0.0
