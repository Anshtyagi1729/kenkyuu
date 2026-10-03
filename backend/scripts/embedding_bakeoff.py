"""Compares embedding models on the evaluation set.

Each model gets its own Chroma collection - not an optimization but a necessity,
since models output different dimensionalities and a collection is fixed-width. The
chunk TEXT is shared (it lives in SQLite), so every model indexes byte-identical
input and the only variable is the model itself.

Each model is also used with its own query/passage convention from embed.SPECS.
Applying BGE's instruction prefix to SPECTER, or none to E5, degrades a model
silently rather than erroring - which would quietly turn a model comparison into a
prompt-handling comparison.

    python -m scripts.embedding_bakeoff                    # all candidates
    python -m scripts.embedding_bakeoff --models allenai/specter
    python -m scripts.embedding_bakeoff --skip-index       # re-score existing collections

Indexing dominates the runtime (a BERT-base model over ~8.7k abstracts on CPU takes
a while), so this is built to run unattended and is safe to re-run: --skip-index
re-scores collections already built.
"""

import argparse
import logging
import re
import time
from pathlib import Path

from app.config import BASE_DIR, EMBEDDING_MODEL
from app.evaluation import harness
from app.ingestion import pipeline, retrieval
from app.storage import db, vector_store

logger = logging.getLogger("bakeoff")

CANDIDATES = [
    EMBEDDING_MODEL,            # BAAI/bge-small-en-v1.5 - the incumbent
    "BAAI/bge-base-en-v1.5",    # same family, 3x the parameters
    "allenai/specter",          # trained on citation-linked paper pairs
]

EVAL_SET_PATH = Path(BASE_DIR) / "data" / "eval_set.json"


def collection_for(model_name: str) -> str:
    """Chroma collection names allow a limited character set, so the model path is
    flattened rather than passed through."""
    return "bakeoff_" + re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def all_paper_ids_with_abstracts() -> list[str]:
    """Every paper eligible for indexing.

    Deliberately not pipeline.unindexed_paper_ids(): that asks "which papers lack a
    chunk row in SQLite", and chunk TEXT is shared across models, so after the first
    model it would return nothing and every later model would score against an empty
    collection.
    """
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT paper_id FROM papers WHERE abstract IS NOT NULL AND abstract != '' ORDER BY paper_id"
        )
        return [row["paper_id"] for row in rows]


def build(model_name: str, paper_ids: list[str]) -> tuple[str, float]:
    collection = collection_for(model_name)
    logger.info("[%s] indexing %d papers into %s", model_name, len(paper_ids), collection)

    start = time.time()
    indexed = pipeline.index_abstracts(paper_ids, collection=collection, model_name=model_name)
    elapsed = time.time() - start

    logger.info("[%s] indexed %d in %.0fs (%.1f/sec)", model_name, indexed, elapsed, indexed / max(elapsed, 1))
    return collection, elapsed


def score(model_name: str, collection: str, queries: list[harness.EvalQuery]) -> harness.EvalReport:
    def system(query: str, k: int) -> list[str]:
        hits = retrieval.search(
            query, top_k=k, fetch_k=k * 4, fusion="none", collection=collection, model_name=model_name
        )
        ranked, seen = [], set()
        for hit in hits:
            if hit["paper_id"] not in seen:
                seen.add(hit["paper_id"])
                ranked.append(hit["paper_id"])
        return ranked

    return harness.evaluate(system, queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=CANDIDATES)
    parser.add_argument("--skip-index", action="store_true", help="score collections already built")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    queries = harness.load_eval_set(EVAL_SET_PATH)
    paper_ids = all_paper_ids_with_abstracts()
    logger.info("%d queries, %d papers, %d models", len(queries), len(paper_ids), len(args.models))

    reports: dict[str, harness.EvalReport] = {}
    for model_name in args.models:
        collection = collection_for(model_name)

        if not args.skip_index:
            build(model_name, paper_ids)
        vectors = vector_store.count(collection)
        if vectors == 0:
            logger.warning("[%s] collection empty - skipping", model_name)
            continue

        logger.info("[%s] scoring against %d vectors", model_name, vectors)
        reports[model_name] = score(model_name, collection, queries)
        logger.info("[%s] %s", model_name, harness.format_report(reports[model_name], model_name))

    print("\n" + "=" * 78)
    print(f"{'model':<30}{'nDCG@10':>10}{'Recall@20':>12}{'MRR':>10}{'specific':>12}")
    print("-" * 78)
    for model_name, report in sorted(reports.items(), key=lambda kv: kv[1].ndcg, reverse=True):
        specific = report.by_intent.get("specific_paper", {}).get("ndcg", 0.0)
        print(
            f"{model_name:<30}{report.ndcg:>10.4f}{report.recall:>12.4f}"
            f"{report.mrr:>10.4f}{specific:>12.4f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
