"""Scores a retrieval configuration against the evaluation set.

    python -m scripts.evaluate                  # score the current system
    python -m scripts.evaluate --validate       # check the eval set is well-formed first
    python -m scripts.evaluate --per-query      # show every query, not just the worst

Each named system is one row of the results table. As new configurations land
(hybrid retrieval, then the learned ranker) they get registered in SYSTEMS below
and scored by exactly the same code, so the comparison stays honest.
"""

import argparse
import sys
from functools import lru_cache
from pathlib import Path

from app.config import BASE_DIR
from app.evaluation import harness
from app.ingestion import retrieval
from app.llm import grading
from app.ranking import rerank
from app.ranking.model import TrainedRanker
from app.storage import db, vector_store

EVAL_SET_PATH = Path(BASE_DIR) / "data" / "eval_set.json"
RANKER_PATH = Path(BASE_DIR) / "data" / "ranker.pkl"


def _to_paper_ranking(hits: list[dict]) -> list[str]:
    """Collapses chunk hits to a paper ranking, keeping each paper's best position.

    Retrieval works on chunks but the eval set judges papers, so several chunks of
    one paper must not occupy several slots in the ranking.
    """
    ranked: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit["paper_id"] not in seen:
            seen.add(hit["paper_id"])
            ranked.append(hit["paper_id"])
    return ranked


# Every system measures against the FROZEN eval collection, never the live one the
# running app writes to - otherwise a paper ingested by an ordinary search between
# two measurements could move a score, and there'd be no way to tell that apart from
# the change actually under test.
EVAL_COLLECTION = vector_store.EVAL_COLLECTION_NAME


def dense_only(query: str, k: int) -> list[str]:
    """Dense vector search + cross-encoder rerank. The Phase 0 baseline.

    Over-fetching by 4x leaves room for same-paper chunks to collapse without
    starving the tail of the ranking.
    """
    return _to_paper_ranking(
        retrieval.search(query, top_k=k, fetch_k=k * 4, fusion="none", collection=EVAL_COLLECTION)
    )


def hybrid_post(query: str, k: int) -> list[str]:
    """Dense, reranked, then fused with BM25 so the lexical ranking survives."""
    return _to_paper_ranking(
        retrieval.search(query, top_k=k, fetch_k=k * 4, fusion="post", collection=EVAL_COLLECTION)
    )


def corrective(query: str, k: int) -> list[str]:
    """Dense retrieval, graded, with a lexical-fusion retry when the grade is a miss.

    This is the agent's Phase 2 corrective loop lifted to the retrieval level so the
    harness can measure it. The harness scores retrieval.search, while grading lives
    a layer above in agent.py - so without exposing the mechanism here, the shipped
    behaviour would be unmeasurable and its results-table row would just repeat the
    dense-only number.

    Costs one extra cheap-tier LLM call per query.
    """
    hits = retrieval.search(query, top_k=k, fetch_k=k * 4, fusion="none", collection=EVAL_COLLECTION)
    grade = grading.grade_retrieval(query, hits[:6])

    if grade.needs_retry:
        # Same reasoning as the agent: retry with a genuinely different signal rather
        # than re-running the retrieval that was just judged irrelevant.
        retry = retrieval.search(query, top_k=k, fetch_k=k * 4, fusion="post", collection=EVAL_COLLECTION)
        # Retry results go FIRST. Appending them after the originals leaves the
        # rejected ranking occupying every position the metrics actually look at,
        # so the correction would be invisible no matter how good the retry was.
        # The originals stay on as fallback rather than being discarded - "judged
        # irrelevant" is one cheap model's opinion, not grounds for throwing away
        # the only candidates we have.
        seen = {h["chunk_id"] for h in retry}
        hits = retry + [h for h in hits if h["chunk_id"] not in seen]

    return _to_paper_ranking(hits)


@lru_cache(maxsize=1)
def _ranker() -> TrainedRanker:
    if not RANKER_PATH.exists():
        sys.exit(f"No trained ranker at {RANKER_PATH} - run: python -m scripts.train_ranker")
    return TrainedRanker.load(RANKER_PATH)


def learned(query: str, k: int) -> list[str]:
    """Dense retrieval, cross-encoder reranked, then reordered by the LambdaMART model.

    The cross-encoder is not replaced - its score is the model's strongest feature.
    What the model adds is the decision of WHEN to override it with citation evidence,
    which the measurements showed a fixed weight cannot express.

    This is the Phase 4 row of the results table, and it is deliberately registered
    here rather than computed in a one-off script: a headline number that only some
    vanished script can reproduce is not a result.
    """
    pool = rerank.candidate_pool(query, collection=EVAL_COLLECTION)
    return _to_paper_ranking(rerank.rank(_ranker(), query, pool))[:k]


SYSTEMS: dict[str, harness.RetrievalFn] = {
    "dense-only": dense_only,
    "hybrid-post": hybrid_post,
    "corrective": corrective,
    "learned": learned,
}


def validate(queries: list[harness.EvalQuery]) -> bool:
    """Checks every judged paper actually exists in the corpus.

    A judgement pointing at a paper we never ingested is unanswerable: the system
    is marked wrong for failing to return something it could not possibly return,
    which quietly drags the baseline down and makes every later comparison look
    better than it is.
    """
    ok = True
    for query in queries:
        if not query.judgements:
            print(f"  [{query.id}] no relevance judgements")
            ok = False
        for paper_id, grade in query.judgements.items():
            if not 0 <= grade <= 3:
                print(f"  [{query.id}] grade {grade} for {paper_id} outside 0-3")
                ok = False
            if _find_paper(paper_id) is None:
                print(f"  [{query.id}] judged paper NOT IN CORPUS: {paper_id}")
                ok = False
    return ok


def _find_paper(canonical_id: str) -> dict | None:
    """Looks up by canonical id, so a bare id in the eval set matches a versioned
    id in the corpus."""
    with db.connect() as conn:
        row = conn.execute(
            "SELECT paper_id, title FROM papers WHERE canonical_id = ? LIMIT 1", (canonical_id,)
        ).fetchone()
        return dict(row) if row else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="*", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument("--validate", action="store_true", help="check the eval set, then exit")
    parser.add_argument("--per-query", action="store_true", help="print every query's score")
    args = parser.parse_args()

    if not EVAL_SET_PATH.exists():
        sys.exit(f"No eval set at {EVAL_SET_PATH}")

    queries = harness.load_eval_set(EVAL_SET_PATH)
    eval_vectors = vector_store.count(EVAL_COLLECTION)
    live_vectors = vector_store.count(vector_store.COLLECTION_NAME)
    print(f"Loaded {len(queries)} queries from {EVAL_SET_PATH.name}")
    print(f"Eval collection: {eval_vectors} vectors (live collection has {live_vectors})")
    if eval_vectors == 0:
        sys.exit("Eval collection is empty - run: python -m scripts.build_corpus --stage snapshot")
    print()

    if args.validate:
        print("Validating...")
        sys.exit(0 if validate(queries) else "Eval set has problems (above).")

    for name in args.systems:
        report = harness.evaluate(SYSTEMS[name], queries)
        print(harness.format_report(report, name))
        if args.per_query:
            print("\n  All queries:")
            for score in sorted(report.per_query, key=lambda s: s.ndcg, reverse=True):
                print(f"    {score.ndcg:.3f}  {score.query_id:<30} {score.top_hit or '(nothing)'}")
        print()


if __name__ == "__main__":
    main()
