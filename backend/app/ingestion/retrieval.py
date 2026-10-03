"""Retrieval: hybrid candidate generation, then cross-encoder rerank.

Embedding similarity alone is a coarse filter - fast but imprecise. Over-fetching
candidates and reranking with a cross-encoder (which scores query and chunk
together, rather than comparing independent embeddings) gets meaningfully better
precision. That matters twice over here: better top-k chunks means better agent
answers, and fewer-but-better chunks means fewer tokens in the synthesis prompt.

Candidates come from two retrievers fused with Reciprocal Rank Fusion - dense for
meaning, BM25 for exact wording. See lexical.py for why both are needed.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import RERANK_MODEL
from app.ingestion import embed, lexical
from app.storage import db, vector_store

# The constant from Cormack et al.'s RRF paper (SIGIR 2009). It damps the influence
# of the very top ranks so one retriever's confident-but-wrong first place can't
# dominate the fused order; 60 is their empirically chosen default.
RRF_K = 60


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Fuses ranked id lists into one score per id: sum of 1/(k + rank).

    RRF uses only POSITIONS, never the underlying scores, which is exactly why it
    suits this pair: a BM25 score of 25.2 and a cosine distance of 0.31 live on
    incomparable scales, and any attempt to normalize them into a weighted sum needs
    calibration that shifts with every corpus and query. Ranks sidestep that entirely.

    An id found by both retrievers accumulates from both, so agreement is rewarded
    without either being able to veto the other.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def search(
    query: str,
    top_k: int = 5,
    fetch_k: int = 20,
    paper_id: str | None = None,
    fusion: str = "none",
    collection: str = vector_store.COLLECTION_NAME,
    model_name: str | None = None,
    citation_weight: float = 0.0,
) -> list[dict]:
    """Retrieves chunks for a query, best first.

    `fusion` selects WHERE BM25 joins the pipeline, which turned out to matter more
    than whether it joins at all:

      "none"  dense retrieval, cross-encoder reranked. 0.5339 nDCG@10.
      "pre"   fuse BM25 into the candidate pool BEFORE reranking. 0.5213 - no better
              than dense, because the rerank that follows re-scores every candidate on
              its own terms and discards the fused order, so BM25 never reaches the
              output. Widening the pool changed nothing (0.5210), confirming it is the
              rerank erasing the signal rather than candidates being crowded out.
      "post"  fuse BM25 with the RERANKED order, so the lexical opinion survives into
              the final ranking. Splits hard by intent: specific-paper +0.053 nDCG and
              comparison +0.077, but topical -0.133, since a question phrased
              differently from the paper is exactly BM25's weakness.

    Default is "none": no single global setting wins everywhere, and "post" buys
    specific-paper accuracy with topical accuracy. The agent applies "post"
    selectively, as a corrective retry once retrieval has been graded a miss - which
    is the closest thing to intent-gating available before the learned ranker exists.

    `collection` lets evaluation run against the frozen eval collection instead of the
    live one, so measurements aren't perturbed by papers the app ingests meanwhile.
    `model_name` selects the embedding model, so a bake-off can compare models without
    each needing its own code path.
    """
    dense_hits = vector_store.query(
        embed.embed_query(query, model_name=model_name),
        top_k=fetch_k,
        paper_id=paper_id,
        collection=collection,
    )
    if not dense_hits:
        return []

    # BM25 has no per-paper filter, so paper-scoped retrieval (the single-paper chat
    # path) stays dense-only rather than fusing against an unfiltered list.
    lexical_hits = (
        [] if (paper_id or fusion == "none") else lexical.get_index().search(query, top_k=fetch_k)
    )

    if fusion == "pre":
        candidates = _fuse(dense_hits, lexical_hits, limit=fetch_k)
        return _rerank(query, candidates)[:top_k]

    reranked = _rerank(query, dense_hits)

    if fusion == "post" and lexical_hits:
        reranked = _fuse(reranked, lexical_hits, limit=fetch_k)

    if citation_weight > 0:
        reranked = _blend_citation_evidence(reranked, citation_weight)

    return reranked[:top_k]


@lru_cache(maxsize=1)
def _pagerank_by_paper() -> dict[str, float]:
    """Centrality scores, loaded once. Cached because this is read on every query and
    the table only changes when the corpus is rebuilt."""
    with db.connect() as conn:
        # NULL is preserved as None, not coerced to 0.0 - the caller distinguishes
        # "no centrality computed" from "centrality computed as zero", and collapsing
        # them here would make that distinction unavailable.
        return {
            row["paper_id"]: row["pagerank"]
            for row in conn.execute(
                "SELECT paper_id, pagerank FROM paper_metrics WHERE pagerank IS NOT NULL"
            )
        }


def _blend_citation_evidence(reranked: list[dict], weight: float) -> list[dict]:
    """Fuses the cross-encoder ordering with a citation-centrality ordering.

    This exists because the cross-encoder has a specific, reproducible blind spot: it
    is trained for question-answering relevance, so for a query like "give me the
    original transformer paper" it rewards a survey whose abstract literally says
    "the 2017 paper Attention Is All You Need introduced the Transformer" over the
    paper itself, whose abstract never contains its own title. Measured on that query,
    dense retrieval ranks the correct paper 1st, the cross-encoder pushes it to 11th.

    The reranker optimizes for what a paper is ABOUT. Citation centrality captures
    what a paper IS, and fusing by rank position lets the second signal correct the
    first where it is confidently wrong.

    Rank-based fusion again, for the same reason as _fuse: a cross-encoder logit and a
    PageRank value are on incomparable scales.
    """
    pagerank = _pagerank_by_paper()

    # Only papers with a COMPUTED centrality take part in the citation ordering.
    #
    # The earlier version defaulted a missing score to 0.0, which ranks "we never
    # measured this paper" below every paper we did measure - an active claim of
    # irrelevance derived from an absence of data. That is the same mistake the
    # scientometric layer refuses to make elsewhere: `normalized_impact` returns None
    # rather than inventing a denominator, and the UI prints "not computable" rather
    # than 0x. The ranking has to honour it too.
    #
    # It is not a theoretical concern. Every paper discovered live sits outside the
    # citation graph until a crawl covers it, and discovering papers live is the
    # system's main use. Measured on the panel's own Springer paper, searched by its
    # title: rank 1 at weight 0.0, rank 4 at 0.3, and absent from the top 5 at the
    # shipped weight of 1.0. The one paper the user most wanted found was the one this
    # default most reliably buried.
    #
    # Excluded rather than imputed, because RRF sums per-ranking contributions: a paper
    # absent from this ordering keeps its text-relevance score and nothing else, so it
    # competes on text alone. A paper with real centrality still gains from both
    # orderings and so still outranks it when their text scores are close - which is
    # the behaviour that fixes "the original transformer paper". Absence costs a paper
    # the bonus; it no longer imposes a penalty.
    with_centrality = [hit for hit in reranked if pagerank.get(hit["paper_id"]) is not None]
    cite_order = sorted(
        with_centrality, key=lambda h: pagerank[h["paper_id"]], reverse=True
    )

    scores: dict[str, float] = {}
    for ordering, w in ((reranked, 1.0), (cite_order, weight)):
        for rank, hit in enumerate(ordering, start=1):
            scores[hit["chunk_id"]] = scores.get(hit["chunk_id"], 0.0) + w / (RRF_K + rank)

    by_id = {hit["chunk_id"]: hit for hit in reranked}
    return [by_id[cid] for cid in sorted(scores, key=scores.get, reverse=True)]


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Scores candidates with the cross-encoder and sorts by that score."""
    if not candidates:
        return []
    scores = _reranker().predict([(query, c["text"]) for c in candidates])
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates


def _fuse(dense_hits: list[dict], lexical_hits: list, limit: int) -> list[dict]:
    """Merges dense and lexical hits by RRF, returning chunk dicts in fused order."""
    if not lexical_hits:
        return dense_hits

    by_id: dict[str, dict] = {hit["chunk_id"]: hit for hit in dense_hits}
    for hit in lexical_hits:
        by_id.setdefault(
            hit.chunk_id,
            {
                "chunk_id": hit.chunk_id,
                "text": hit.text,
                "paper_id": hit.paper_id,
                # Distance is dense-retrieval's own metric and is genuinely unknown for
                # a lexical-only hit. None rather than a filler value, so nothing
                # downstream mistakes a placeholder for a real measurement.
                "distance": None,
                "start_page": None,
                "end_page": None,
            },
        )

    fused = reciprocal_rank_fusion(
        [[hit["chunk_id"] for hit in dense_hits], [hit.chunk_id for hit in lexical_hits]]
    )
    ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)

    results = []
    for chunk_id, score in ordered[:limit]:
        chunk = by_id[chunk_id]
        chunk["rrf_score"] = score
        results.append(chunk)
    return results
