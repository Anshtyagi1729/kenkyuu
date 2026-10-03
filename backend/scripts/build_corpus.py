"""Builds the evaluation corpus: arXiv metadata, then Semantic Scholar metrics.

Run this FIRST and let it run. Everything in the scientometric phase depends on
having a corpus with real cohort depth, and the crawl is mostly wall-clock time
(rate limits), not compute - so it should be working in the background while
other phases are built.

Five stages, each independently resumable and safe to re-run:

  seed      Resolves the landmark paper titles and stores them. Cohort sampling
            catches famous papers only by luck, so these are fetched explicitly.

  arxiv     Walks every (category, year) cohort and stores paper metadata.
            Progress is checkpointed per cohort, so an interrupted run picks up
            at the next unfetched page rather than re-walking from the start.

  metrics   Fills in citation counts and venue from Semantic Scholar, 500 papers
            per request. The work queue is "papers with no metrics row", so this
            is resumable for free and safe to re-run after adding more papers.

  index     Embeds title+abstract for every paper into the live vector collection.
            Local model, no API, so nothing here is rate-limited.

  snapshot  Copies the live collection to the frozen eval collection. Run this
            before measuring, so every configuration is scored against the same
            corpus even as the running app keeps ingesting new papers.

Usage:
    python -m scripts.build_corpus                       # every stage, defaults
    python -m scripts.build_corpus --per-cohort 400      # bigger corpus
    python -m scripts.build_corpus --stage metrics       # just enrich what exists
    python -m scripts.build_corpus --stage snapshot      # re-freeze before measuring
    python -m scripts.build_corpus --years 2020 2026     # narrower year range

Safe to interrupt with Ctrl+C at any point; re-running continues where it left off.

Do NOT run this against arXiv while the app is also serving live searches - the
rate limiter is per-process and cannot coordinate across two, which trips a 429
that locks out both for several minutes.
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.config import CORPUS_CATEGORIES
from app.evaluation.landmarks import LANDMARK_TITLES
from app.ingestion import arxiv_client, pipeline, semantic_scholar_client
from app.scientometrics import graph, impact
from app.storage import db, vector_store

logger = logging.getLogger("build_corpus")

PAGE_SIZE = 100  # arXiv results per request; their API is unhappy above a few hundred
DEFAULT_PER_COHORT = 200
DEFAULT_YEAR_RANGE = (2017, 2026)  # 2017 = "Attention Is All You Need", a sane floor for DL
TITLE_MATCH_THRESHOLD = 0.85  # similarity below this is treated as "not the paper we asked for"
CONSECUTIVE_FAILURE_LIMIT = 3
COOL_OFF_SECONDS = 120

_interrupted = False


def _handle_interrupt(signum, frame) -> None:
    """Finish the page in flight, then stop cleanly - killing mid-write could leave
    a cohort checkpoint pointing past papers that were never actually stored."""
    global _interrupted
    if _interrupted:  # second Ctrl+C means they really mean it
        sys.exit(130)
    _interrupted = True
    logger.warning("Interrupt received - finishing current page, then stopping. Ctrl+C again to force.")


@dataclass
class CrawlStats:
    fetched: int = 0
    stored: int = 0
    skipped: int = 0

    def __str__(self) -> str:
        return f"{self.stored} stored, {self.skipped} already known, {self.fetched} fetched"


def _cohort_key(category: str, year: int) -> str:
    return f"arxiv_offset:{category}:{year}"


def crawl_arxiv(categories: tuple[str, ...], years: range, per_cohort: int) -> CrawlStats:
    """Walks each (category, year) cohort, storing paper metadata as it goes."""
    stats = CrawlStats()
    total_cohorts = len(categories) * len(years)
    cohort_num = 0
    consecutive_failures = 0

    for category in categories:
        for year in years:
            cohort_num += 1
            if _interrupted:
                return stats

            key = _cohort_key(category, year)
            offset = int(db.get_crawl_state(key, "0"))

            if offset >= per_cohort:
                logger.info("[%d/%d] %s %d - complete (%d papers)", cohort_num, total_cohorts, category, year, offset)
                continue

            logger.info(
                "[%d/%d] %s %d - resuming at offset %d, target %d",
                cohort_num, total_cohorts, category, year, offset, per_cohort,
            )

            while offset < per_cohort and not _interrupted:
                page_size = min(PAGE_SIZE, per_cohort - offset)
                try:
                    papers = arxiv_client.list_cohort(category, year, start=offset, page_size=page_size)
                    consecutive_failures = 0
                except Exception as exc:  # noqa: BLE001 - one bad page shouldn't end the crawl
                    consecutive_failures += 1
                    logger.warning("  %s %d @ %d failed (%s) - skipping to next cohort", category, year, offset, exc)

                    # Failures this close together mean the API is refusing us, not that
                    # this particular cohort is bad - without a pause the crawl sprints
                    # through every remaining cohort in seconds, "completing" with most
                    # of the corpus silently missing. Cohort offsets are checkpointed, so
                    # a later re-run resumes them; the cool-off just stops us burning
                    # through the whole plan while the door is shut.
                    if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                        logger.error(
                            "%d consecutive failures - pausing %ds to let the API recover",
                            consecutive_failures, COOL_OFF_SECONDS,
                        )
                        time.sleep(COOL_OFF_SECONDS)
                        consecutive_failures = 0
                    break

                if not papers:  # cohort exhausted before hitting the target
                    logger.info("  %s %d - only %d papers available", category, year, offset)
                    break

                for paper in papers:
                    stats.fetched += 1
                    if db.get_paper(paper.arxiv_id) is not None:
                        stats.skipped += 1
                        continue
                    _store(paper, category)
                    stats.stored += 1

                offset += len(papers)
                db.set_crawl_state(key, str(offset))

    return stats


def _store(paper: arxiv_client.ArxivPaper, category: str) -> None:
    """`category` is the crawl bucket this paper was found under; it is only a
    fallback. The paper's OWN primary category is what matters, because that is the
    cohort its citation impact gets normalized against - and a cross-listed paper
    found under cs.AI may really be a cs.LG paper. Landmarks in particular arrive
    with no crawl bucket at all, so without this they would be filed under a bucket
    name that matches no real cohort and could never be normalized."""
    db.upsert_paper(
        paper_id=paper.arxiv_id,
        source="arxiv",
        title=paper.title,
        abstract=paper.abstract,
        authors=paper.authors,
        published=paper.published,
        external_ids={"ArXiv": paper.arxiv_id, "primary_category": paper.primary_category or category},
    )


def seed_landmarks() -> tuple[int, list[str]]:
    """Resolves each landmark title to a real paper and stores it.

    Returns (stored, unresolved_titles). Unresolved titles are reported rather than
    silently dropped - a landmark missing from the corpus quietly breaks any eval
    query that depends on it, and we'd rather see the failure now than discover it
    as an inexplicably bad score later.
    """
    stored = 0
    unresolved: list[str] = []

    for i, title in enumerate(LANDMARK_TITLES, start=1):
        if _interrupted:
            break
        try:
            # Category scoping is deliberately disabled here. Several landmarks predate
            # or sit outside the five DL categories (Adam is cs.LG but U-Net is cs.CV,
            # and some older papers are filed under math/stat only) - scoping would
            # drop exactly the foundational papers this list exists to guarantee.
            matches = arxiv_client.search(title, max_results=5, categories=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  [%d/%d] %s - lookup failed (%s)", i, len(LANDMARK_TITLES), title[:50], exc)
            unresolved.append(title)
            continue

        best = _best_title_match(title, matches)
        if best is None:
            logger.warning("  [%d/%d] %s - NO CONFIDENT MATCH", i, len(LANDMARK_TITLES), title[:50])
            unresolved.append(title)
            continue

        _store(best, category="landmark")
        stored += 1
        logger.info("  [%d/%d] %-14s %s", i, len(LANDMARK_TITLES), best.arxiv_id, best.title[:52])

    return stored, unresolved


def _normalize_title(title: str) -> str:
    return "".join(c for c in title.lower() if c.isalnum() or c.isspace()).strip()


def _best_title_match(wanted: str, candidates: list[arxiv_client.ArxivPaper]) -> arxiv_client.ArxivPaper | None:
    """Closest candidate by title similarity, or None if nothing is close enough.

    The threshold guards against arXiv's search confidently returning a same-topic
    paper when the one we asked for isn't indexed - seeding "Mamba" with a paper
    merely *about* Mamba would poison an eval set built on top of it.
    """
    target = _normalize_title(wanted)
    best, best_score = None, 0.0

    for candidate in candidates:
        score = SequenceMatcher(None, target, _normalize_title(candidate.title)).ratio()
        if score > best_score:
            best, best_score = candidate, score

    return best if best_score >= TITLE_MATCH_THRESHOLD else None


def crawl_citation_graph() -> tuple[int, int]:
    """Fetches reference lists in batches and stores the edges between corpus papers.

    Returns (edges_stored, papers_with_references).

    Only edges whose BOTH endpoints are in the corpus are kept - the induced subgraph.
    A typical paper cites 40+ works, most outside a deep-learning sample, and keeping
    those would add hundreds of thousands of leaf nodes that can never be ranked or
    returned while diluting the centrality of everything real. The resulting claim is
    narrower and honest: central within deep learning as sampled here.
    """
    s2_lookup = graph.s2_to_corpus_id()
    paper_ids = [pid for pid in _all_paper_ids() if pid]
    batch_size = semantic_scholar_client.BATCH_LIMIT
    total_batches = (len(paper_ids) + batch_size - 1) // batch_size

    logger.info("Fetching references for %d papers in %d batches", len(paper_ids), total_batches)

    all_edges: list[tuple[str, str]] = []
    with_refs = 0

    for batch_num, start in enumerate(range(0, len(paper_ids), batch_size), start=1):
        if _interrupted:
            break
        batch = paper_ids[start : start + batch_size]

        try:
            references = semantic_scholar_client.get_references_batch(batch)
        except Exception as exc:  # noqa: BLE001 - keep edges gathered so far
            logger.warning("  batch %d/%d failed (%s) - continuing", batch_num, total_batches, exc)
            continue

        batch_edges = [
            (citing_id, s2_lookup[cited_s2])
            for citing_id, cited_s2_ids in references.items()
            for cited_s2 in cited_s2_ids
            if cited_s2 in s2_lookup
        ]
        all_edges.extend(batch_edges)
        with_refs += len(references)

        logger.info(
            "  batch %d/%d - %d papers had references, %d internal edges (%d total)",
            batch_num, total_batches, len(references), len(batch_edges), len(all_edges),
        )

    if all_edges:
        graph.store_edges(all_edges)
    return len(all_edges), with_refs


def _all_paper_ids() -> list[str]:
    with db.connect() as conn:
        return [row["paper_id"] for row in conn.execute("SELECT paper_id FROM papers ORDER BY paper_id")]


def crawl_metrics() -> int:
    """Fills citation counts and venue for every paper still missing them."""
    pending = db.paper_ids_missing_metrics()
    if not pending:
        logger.info("Metrics already complete for every paper.")
        return 0

    batch_size = semantic_scholar_client.BATCH_LIMIT
    total_batches = (len(pending) + batch_size - 1) // batch_size
    logger.info("Enriching %d papers in %d batches of up to %d", len(pending), total_batches, batch_size)

    enriched = 0
    for batch_num, start in enumerate(range(0, len(pending), batch_size), start=1):
        if _interrupted:
            break
        batch = pending[start : start + batch_size]

        try:
            metrics = semantic_scholar_client.get_metrics_batch(batch)
        except Exception as exc:  # noqa: BLE001 - keep whatever the earlier batches stored
            logger.warning("  batch %d/%d failed (%s) - continuing", batch_num, total_batches, exc)
            continue

        for paper_id in batch:
            found = metrics.get(paper_id)
            paper = db.get_paper(paper_id)
            # Cohort year comes from the arXiv SUBMISSION date, never the id prefix:
            # a paper submitted 2023-12-31 is announced in January and gets a 2401.x
            # id, which would file it under the wrong cohort and skew that year's median.
            cohort_year = int(paper["published"][:4]) if paper and paper.get("published") else None
            category = _primary_category(paper)

            db.upsert_paper_metrics(
                paper_id=paper_id,
                s2_paper_id=found.s2_paper_id if found else None,
                citation_count=found.citation_count if found else None,
                influential_citation_count=found.influential_citation_count if found else None,
                reference_count=found.reference_count if found else None,
                venue=found.venue if found else None,
                cohort_year=cohort_year,
                primary_category=category,
            )
            if found:
                enriched += 1

        logger.info(
            "  batch %d/%d - %d/%d matched in S2 (%d total enriched)",
            batch_num, total_batches, len(metrics), len(batch), enriched,
        )

    return enriched


def _primary_category(paper: dict | None) -> str | None:
    if not paper:
        return None
    return json.loads(paper.get("external_ids") or "{}").get("primary_category")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=["seed", "arxiv", "metrics", "graph", "cohorts", "index", "snapshot", "all"],
        default="all",
    )
    parser.add_argument(
        "--per-cohort", type=int, default=DEFAULT_PER_COHORT,
        help=f"papers per (category, year) pair; default {DEFAULT_PER_COHORT}",
    )
    parser.add_argument(
        "--years", nargs=2, type=int, metavar=("START", "END"), default=DEFAULT_YEAR_RANGE,
        help=f"inclusive year range; default {DEFAULT_YEAR_RANGE[0]} {DEFAULT_YEAR_RANGE[1]}",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    signal.signal(signal.SIGINT, _handle_interrupt)

    db.init_db()
    years = range(args.years[0], args.years[1] + 1)

    target = len(CORPUS_CATEGORIES) * len(years) * args.per_cohort
    logger.info(
        "Corpus target: %d categories x %d years x %d papers = up to %d (overlap between "
        "categories means the real total will be lower)",
        len(CORPUS_CATEGORIES), len(years), args.per_cohort, target,
    )
    logger.info("Starting from %d papers, %d with metrics", db.count_papers(), db.count_metrics())

    if args.stage in ("seed", "all"):
        logger.info("--- Stage 1: landmark papers ---")
        stored, unresolved = seed_landmarks()
        logger.info("Done: %d/%d landmarks resolved", stored, len(LANDMARK_TITLES))
        if unresolved:
            logger.warning("UNRESOLVED landmarks (check these by hand): %s", "; ".join(unresolved))

    if args.stage in ("arxiv", "all") and not _interrupted:
        logger.info("--- Stage 2: cohort crawl ---")
        logger.info("Done: %s", crawl_arxiv(CORPUS_CATEGORIES, years, args.per_cohort))

    if args.stage in ("metrics", "all") and not _interrupted:
        logger.info("--- Stage 3: Semantic Scholar metrics ---")
        logger.info("Done: %d papers enriched", crawl_metrics())

    if args.stage in ("graph", "all") and not _interrupted:
        logger.info("--- Stage 3b: citation graph ---")
        edges, with_refs = crawl_citation_graph()
        logger.info("Done: %d internal edges from %d papers with references", edges, with_refs)
        logger.info("Graph: %s", graph.graph_stats(graph.load_graph()))

    if args.stage in ("cohorts", "all") and not _interrupted:
        logger.info("--- Stage 3c: cohort baselines ---")
        n = impact.store_cohorts(impact.compute_cohorts())
        n += impact.store_cohorts(impact.compute_pooled_cohorts())
        logger.info("Done: %d cohort baselines (per-category and pooled-by-year)", n)

        logger.info("Computing derived scores...")
        g = graph.load_graph()
        stored = graph.store_scores(graph.compute_pagerank(g), dict(g.in_degree()))
        normalized = impact.compute_and_store_cnci()
        logger.info("Done: %d centrality scores, %d papers with a CNCI", stored, normalized)

    if args.stage in ("index", "all") and not _interrupted:
        logger.info("--- Stage 4: abstract embeddings ---")
        pending = pipeline.unindexed_paper_ids()
        logger.info("Embedding %d abstracts (local model, no API involved)", len(pending))
        logger.info("Done: %d indexed", pipeline.index_abstracts(pending))

    if args.stage in ("snapshot", "all") and not _interrupted:
        logger.info("--- Stage 5: freeze eval snapshot ---")
        logger.info("Done: %d vectors copied to the eval collection", vector_store.snapshot_to_eval())

    logger.info("Corpus now: %d papers, %d with metrics", db.count_papers(), db.count_metrics())
    if _interrupted:
        logger.info("Stopped early - re-run the same command to continue.")


if __name__ == "__main__":
    main()
