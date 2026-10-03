"""Assembles labeled (query, candidate, features, label) rows for the ranker.

For each training query: retrieve candidates, compute features for each, label the
known-correct paper 1 and the rest 0. That is the standard learning-to-rank setup -
the model learns to order a candidate list, not to classify papers in isolation.

Train and test share the document collection but never share QUERIES. That is correct
for retrieval (the corpus is the same corpus at train and test time); what would
invalidate the result is reusing an evaluation query, and the eval queries are
hand-written while these are generated from abstracts.

    python -m scripts.build_training_set --queries 1200
"""

import argparse
import logging
import pickle
import time
from pathlib import Path

from app.config import BASE_DIR
from app.ingestion import lexical, retrieval
from app.ranking import training_data as td
from app.ranking.features import Features, extract
from app.storage import db, vector_store

logger = logging.getLogger("build_training_set")

OUT_PATH = Path(BASE_DIR) / "data" / "training_rows.pkl"
CANDIDATES_PER_QUERY = 30


def load_metrics() -> dict[str, dict]:
    with db.connect() as conn:
        return {
            r["paper_id"]: dict(r)
            for r in conn.execute(
                "SELECT m.paper_id, m.citation_count, m.influential_citation_count, m.cnci, "
                "m.pagerank, m.in_degree, m.cohort_year, m.venue FROM paper_metrics m"
            )
        }


def load_titles() -> dict[str, str]:
    with db.connect() as conn:
        return {r["paper_id"]: r["title"] for r in conn.execute("SELECT paper_id, title FROM papers")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=1200)
    parser.add_argument("--citation-per-bucket", type=int, default=150)
    parser.add_argument("--title-queries", type=int, default=600)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # Synthetic is the primary source (unbiased by fame); rebalanced citation pairs
    # add real lookup signal without reintroducing the skew.
    examples = td.build_synthetic(limit=args.queries)
    examples += td.build_citation_balanced(per_bucket=args.citation_per_bucket)
    # Title lookups are a THIRD query shape, not more of the same. Without them the
    # model never sees a query that IS a paper's title, and it measurably mishandles
    # them - see build_title_lookups for the failures that motivated this.
    examples += td.build_title_lookups(limit=args.title_queries)
    logger.info("Total training queries: %d", len(examples))

    metrics, titles = load_metrics(), load_titles()
    bm25 = lexical.get_index()

    rows, skipped, start = [], 0, time.time()
    for i, example in enumerate(examples, 1):
        hits = retrieval.search(
            example.query, top_k=CANDIDATES_PER_QUERY, fetch_k=CANDIDATES_PER_QUERY * 2,
            fusion="none", collection=vector_store.EVAL_COLLECTION_NAME,
        )
        if not hits:
            skipped += 1
            continue

        # A query whose correct paper was never retrieved teaches nothing about
        # ranking - there is no right answer in the list to learn to raise.
        candidate_ids = {h["paper_id"] for h in hits}
        if example.positive_paper_id not in candidate_ids:
            skipped += 1
            continue

        bm25_scores = {h.paper_id: h.score for h in bm25.search(example.query, top_k=50)}

        group = []
        for hit in hits:
            pid = hit["paper_id"]
            hit["title"] = titles.get(pid, "")
            feats = extract(example.query, hit, metrics.get(pid), bm25_score=bm25_scores.get(pid, 0.0))
            group.append((feats.to_list(), int(pid == example.positive_paper_id)))
        rows.append({"query": example.query, "source": example.source, "group": group})

        if i % 100 == 0:
            rate = i / (time.time() - start)
            logger.info("%d/%d queries (%.1f/s, %d usable, %d skipped)", i, len(examples), rate, len(rows), skipped)

    OUT_PATH.write_bytes(pickle.dumps({"rows": rows, "feature_names": Features.names()}))
    positives = sum(sum(label for _, label in r["group"]) for r in rows)
    logger.info(
        "Wrote %d query groups (%d candidate rows, %d positives) to %s",
        len(rows), sum(len(r["group"]) for r in rows), positives, OUT_PATH.name,
    )


if __name__ == "__main__":
    main()
