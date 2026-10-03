"""Applies the trained ranker to a retrieved candidate pool.

This is the inference half of Phase 4, and it is deliberately a module rather than
inline code in a script: the same three steps - build a pool, featurize it, reorder
it - are needed by the evaluation harness, by the ablation, and eventually by the
agent's retrieval tool. Duplicating them would mean the number in the report and the
behaviour in the product could silently diverge.

The pool comes from the existing pipeline (dense retrieval, cross-encoder reranked)
rather than from the vector store directly, because `rerank_score` is a feature: the
model's job is not to replace the cross-encoder but to decide when to override it
using citation evidence.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

from app.config import BASE_DIR
from app.ingestion import lexical, retrieval
from app.ranking.features import Features, extract
from app.ranking.model import TrainedRanker
from app.storage import db, vector_store

logger = logging.getLogger(__name__)

# 60 candidates deep. The measured Recall@20 of 1.0000 on specific-paper queries says
# the right answer is already retrieved; a pool this size guarantees it is present to
# be promoted, while staying small enough that featurizing it costs nothing.
POOL_SIZE = 60

# BM25 is consulted deeper than the pool so a candidate the lexical index ranks 55th
# still gets its real score rather than a 0.0 that the model would read as "no
# lexical match at all".
BM25_DEPTH = 200


@dataclass
class RankContext:
    """Corpus-wide lookups the featurizer needs. Loaded once - these are read for
    every candidate of every query, and the tables only change when the corpus is
    rebuilt."""

    metrics: dict[str, dict]
    titles: dict[str, str]
    bm25: lexical.BM25Index


@lru_cache(maxsize=1)
def context() -> RankContext:
    with db.connect() as conn:
        metrics = {
            row["paper_id"]: dict(row)
            for row in conn.execute(
                "SELECT paper_id, citation_count, influential_citation_count, cnci, "
                "pagerank, in_degree, cohort_year, venue FROM paper_metrics"
            )
        }
        titles = {
            row["paper_id"]: row["title"]
            for row in conn.execute("SELECT paper_id, title FROM papers")
        }
    logger.info("rank context: %d papers with metrics, %d titles", len(metrics), len(titles))
    return RankContext(metrics=metrics, titles=titles, bm25=lexical.get_index())


def candidate_pool(
    query: str,
    pool_size: int = POOL_SIZE,
    collection: str = vector_store.COLLECTION_NAME,
) -> list[dict]:
    """Retrieves and annotates the candidates the ranker will reorder.

    Title and BM25 score are attached here rather than in `extract`, so the featurizer
    stays a pure function of what it is handed and the pool can be cached to disk with
    everything the features need already in it.
    """
    hits = retrieval.search(
        query,
        top_k=pool_size,
        fetch_k=pool_size * 2,
        fusion="none",
        collection=collection,
    )
    if not hits:
        return []

    ctx = context()
    bm25_scores = {h.paper_id: h.score for h in ctx.bm25.search(query, top_k=BM25_DEPTH)}
    for hit in hits:
        paper_id = hit["paper_id"]
        hit["title"] = ctx.titles.get(paper_id, "")
        hit["bm25_score"] = bm25_scores.get(paper_id, 0.0)
    return hits


def feature_rows(query: str, pool: list[dict]) -> list[list[float]]:
    """One feature row per candidate, in `Features.names()` order."""
    ctx = context()
    return [
        extract(
            query,
            hit,
            ctx.metrics.get(hit["paper_id"]),
            bm25_score=hit.get("bm25_score", 0.0),
        ).to_list()
        for hit in pool
    ]


def project(rows: list[list[float]], kept_names: list[str]) -> list[list[float]]:
    """Selects the columns a given model was trained on.

    Required by the ablation: a variant trained without the citation group expects 11
    columns, not 15, and LightGBM would either error or - worse, if the counts
    happened to line up - read the wrong feature under the right name.
    """
    all_names = Features.names()
    missing = [name for name in kept_names if name not in all_names]
    if missing:
        raise ValueError(f"model expects features this featurizer does not produce: {missing}")
    idx = [all_names.index(name) for name in kept_names]
    return [[row[i] for i in idx] for row in rows]


def rank(
    ranker: TrainedRanker,
    query: str,
    pool: list[dict],
    rows: list[list[float]] | None = None,
) -> list[dict]:
    """Reorders a candidate pool by the model's score, best first.

    `rows` lets a caller pass features computed earlier - the ablation scores six
    models over the same 40 pools, and recomputing identical features six times would
    be waste that also risks the variants seeing subtly different inputs.
    """
    if not pool:
        return []
    rows = rows if rows is not None else feature_rows(query, pool)
    scores = ranker.score(project(rows, ranker.feature_names))
    for hit, score in zip(pool, scores):
        hit["ranker_score"] = float(score)
    return sorted(pool, key=lambda h: h["ranker_score"], reverse=True)


MODEL_PATH = Path(BASE_DIR) / "data" / "ranker.pkl"


@lru_cache(maxsize=1)
def _shipped_ranker() -> TrainedRanker | None:
    """The trained model, or None when it has not been built yet.

    None rather than an exception: a fresh clone has no `data/ranker.pkl` until
    `scripts.train_ranker` runs, and the app should still serve search in that state
    rather than refusing to start.
    """
    if not MODEL_PATH.exists():
        logger.warning("no trained ranker at %s - falling back to fixed-weight blend", MODEL_PATH)
        return None
    return TrainedRanker.load(MODEL_PATH)


def search_ranked(
    query: str,
    top_k: int = 5,
    collection: str = vector_store.COLLECTION_NAME,
    fallback_citation_weight: float = 1.0,
) -> list[dict]:
    """Retrieval as the product ships it: pool, featurize, reorder by the learned model.

    This replaces a fixed citation weight, and the reason is the project's central
    finding reproduced inside the product rather than on the benchmark. Searching for
    the panel's own Springer paper by title, and for the original transformer paper,
    under the two fixed weights:

        weight 0.0   panel paper rank 1,  transformer absent from the top 5
        weight 1.0   panel paper absent,  transformer rank 1

    No single weight serves both. A weight of 1.0 gives citation centrality the same
    say as text relevance, so any paper with measured centrality outranks a paper
    without one whatever the text says - and a paper discovered live has no centrality
    until a crawl reaches it. A weight of 0.0 throws away the signal that rescues the
    transformer paper from the cross-encoder's blind spot.

    The learned ranker places all four regression papers at rank 1, because it can
    condition - trust the text score when a query names a paper precisely, reach for
    centrality when the query describes one instead. That conditional behaviour is what
    Phase 4 was built to provide and what the eval set measured (+0.1303 nDCG@10 over
    the cross-encoder, outside the 0.0828 seed-variance range).

    Falls back to the fixed blend when no model is present, so behaviour degrades to
    the previous shipped configuration rather than to nothing.
    """
    ranker = _shipped_ranker()
    if ranker is None:
        return retrieval.search(
            query,
            top_k=top_k,
            fetch_k=POOL_SIZE,
            collection=collection,
            citation_weight=fallback_citation_weight,
        )

    pool = candidate_pool(query, collection=collection)
    if not pool:
        return []
    return rank(ranker, query, pool)[:top_k]
