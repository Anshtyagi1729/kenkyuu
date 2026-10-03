"""Milestone 0 verification: confirms the two-tier ingestion pipeline.

Tier 1 (search) should be fast and cheap - metadata + abstract embeddings
only, no PDF downloads. Tier 2 (full-text ingest) is lazy and cached - only
runs for papers we explicitly ask for, and is a no-op on repeat calls.

Run with:
    python -m scripts.ingest_sample
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import embed, pipeline
from app.storage import db, vector_store

QUERY = "retrieval augmented generation"


def main() -> None:
    db.init_db()

    print(f"Tier 1 - searching arXiv for: {QUERY!r} (metadata + abstracts only, no PDFs)")
    start = time.monotonic()
    papers = pipeline.search_and_record(QUERY, max_results=5)
    print(f"  Found + recorded {len(papers)} papers in {time.monotonic() - start:.1f}s:")
    for p in papers:
        print(f"    - {p.arxiv_id}: {p.title}")

    print(f"\nVector store holds {vector_store.count()} chunks (abstracts only so far).")

    print("\nTier 2 - lazily ingesting full text for just the first result...")
    target = papers[0].arxiv_id
    start = time.monotonic()
    chunk_count = pipeline.ingest_full_text(target)
    print(f"  Ingested {chunk_count} body chunks for {target} in {time.monotonic() - start:.1f}s")

    print("  Re-requesting the same paper (should be instant, a no-op)...")
    start = time.monotonic()
    repeat_count = pipeline.ingest_full_text(target)
    print(f"  Returned {repeat_count} new chunks in {time.monotonic() - start:.2f}s (0 expected)")

    print(f"\nVector store now holds {vector_store.count()} chunks total.")

    test_query = "how does retrieval improve factual accuracy in language models"
    print(f"\nTest retrieval query: {test_query!r}")
    hits = vector_store.query(embed.embed_query(test_query), top_k=3)
    for hit in hits:
        paper = db.get_paper(hit["paper_id"])
        snippet = hit["text"][:160].replace("\n", " ")
        print(f"  [{hit['distance']:.3f}] {paper['title']}\n      {snippet}...")


if __name__ == "__main__":
    main()
