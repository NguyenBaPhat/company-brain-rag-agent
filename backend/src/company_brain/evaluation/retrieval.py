"""Offline retrieval evaluation (no LLM, no API key).

Measures, on a labelled query set:
* hit@k / MRR — does the right document show up near the top?
* the *relevance gate* — how well does the dense-cosine threshold separate answerable queries from
  off-domain queries and from "gaps" (on-topic questions the KB cannot answer)?

Use it to choose an embedding model and to calibrate ``RELEVANCE_MIN`` with data instead of guesses::

    cb-eval-retrieval --models BAAI/bge-base-en-v1.5,sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient

from ..config import Settings, get_settings
from ..knowledge.chunker import chunk_documents
from ..knowledge.embeddings import Embedder
from ..knowledge.loader import load_documents
from ..knowledge.store import KnowledgeStore

CASES_PATH = "data/eval/retrieval_cases.jsonl"


@dataclass
class Case:
    id: str
    query: str
    expect_docs: list[str]
    kind: str  # answerable | off_domain | gap


@dataclass
class CaseResult:
    case: Case
    ranked_docs: list[str]
    best_relevance: float
    first_hit_rank: int | None


@dataclass
class Report:
    model: str
    hit_at: dict[int, float]
    mrr: float
    results: list[CaseResult] = field(default_factory=list)

    def gate_sweep(self, thresholds: list[float]) -> list[dict[str, float]]:
        ans = [r.best_relevance for r in self.results if r.case.kind == "answerable"]
        off = [r.best_relevance for r in self.results if r.case.kind == "off_domain"]
        gap = [r.best_relevance for r in self.results if r.case.kind == "gap"]
        rows = []
        for t in thresholds:
            rows.append(
                {
                    "threshold": t,
                    "answerable_pass": sum(x >= t for x in ans) / max(len(ans), 1),
                    "off_domain_reject": sum(x < t for x in off) / max(len(off), 1),
                    "gap_reject": sum(x < t for x in gap) / max(len(gap), 1),
                }
            )
        return rows

    def recommended_threshold(self) -> float:
        """Highest threshold that still passes >=95% of answerable queries (max off-domain rejection)."""
        grid = [round(0.05 + 0.01 * i, 2) for i in range(0, 90)]
        ok = [r for r in self.gate_sweep(grid) if r["answerable_pass"] >= 0.95]
        return max((r["threshold"] for r in ok), default=0.0)


def load_cases(path: Path) -> list[Case]:
    cases = [Case(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError(f"no eval cases in {path}")
    return cases


def evaluate(store: KnowledgeStore, cases: list[Case], settings: Settings, model: str = "") -> Report:
    results: list[CaseResult] = []
    for case in cases:
        hits = store.search(case.query, top_k=settings.retrieval_top_k, candidates=settings.retrieval_candidates)
        ranked: list[str] = []
        for h in hits:
            if h.chunk.doc_id not in ranked:
                ranked.append(h.chunk.doc_id)
        first = next((i + 1 for i, d in enumerate(ranked) if d in case.expect_docs), None)
        results.append(
            CaseResult(case, ranked, max((h.relevance for h in hits), default=0.0), first)
        )
    answerable = [r for r in results if r.case.kind == "answerable"]
    n = max(len(answerable), 1)
    hit_at = {k: sum(1 for r in answerable if r.first_hit_rank and r.first_hit_rank <= k) / n for k in (1, 3, 5)}
    mrr = sum(1 / r.first_hit_rank for r in answerable if r.first_hit_rank) / n
    return Report(model=model, hit_at=hit_at, mrr=mrr, results=results)


def build_eval_store(settings: Settings, dense_model: str) -> KnowledgeStore:
    embedder = Embedder(dense_model, settings.sparse_model, str(settings.embedding_cache_path))
    store = KnowledgeStore(QdrantClient(":memory:"), embedder, "eval")
    store.sync_chunks(chunk_documents(load_documents(settings.kb_path)))
    return store


def print_report(report: Report, verbose: bool) -> None:
    print(f"\n=== {report.model}")
    print(
        f"hit@1={report.hit_at[1]:.2f}  hit@3={report.hit_at[3]:.2f}  "
        f"hit@5={report.hit_at[5]:.2f}  MRR={report.mrr:.3f}"
    )
    misses = [r for r in report.results if r.case.kind == "answerable" and not r.first_hit_rank]
    for r in misses:
        print(f"  MISS {r.case.id}: {r.case.query!r} expected={r.case.expect_docs} got={r.ranked_docs[:3]}")
    for kind in ("answerable", "off_domain", "gap"):
        vals = sorted(r.best_relevance for r in report.results if r.case.kind == kind)
        if vals:
            median = vals[len(vals) // 2]
            print(f"  best-relevance [{kind:10s}] min={vals[0]:.2f} median={median:.2f} max={vals[-1]:.2f}")
    print(f"  recommended RELEVANCE_MIN (keeps >=95% answerable): {report.recommended_threshold():.2f}")
    if verbose:
        for row in report.gate_sweep([round(0.1 * i, 2) for i in range(2, 9)]):
            print(
                f"    t={row['threshold']:.2f}  answerable_pass={row['answerable_pass']:.2f}  "
                f"off_domain_reject={row['off_domain_reject']:.2f}  gap_reject={row['gap_reject']:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and calibrate the gate.")
    parser.add_argument("--models", default="", help="Comma-separated dense models to compare.")
    parser.add_argument("--cases", default=CASES_PATH)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level="WARNING")

    settings = get_settings()
    cases = load_cases(settings.resolve(args.cases))
    models = [m.strip() for m in args.models.split(",") if m.strip()] or [settings.dense_model]
    for model in models:
        store = build_eval_store(settings, model)
        print_report(evaluate(store, cases, settings, model), args.verbose)


if __name__ == "__main__":
    main()
