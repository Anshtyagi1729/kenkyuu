"""A paper's scientometric scorecard, in numbers a human can read.

Phase 3 computed the metrics; this assembles them into the claim the product
actually makes. The requirement it serves is concreteness: a recommendation that
says "this paper is relevant" is an opinion, while one that says "cited 193,374
times, 6,446x the median 2017 paper in its field, and 99.9th percentile by citation
network centrality" is a measurement the reader can check.

Percentiles are computed against this corpus, not against science as a whole, and
the wording says so. A raw CNCI of 6,446x is hard to place without knowing that
almost every paper sits near 1x; the percentile supplies that context.

Missing values stay missing. A paper whose CNCI could not be computed - because its
cohort was too small to trust a median - reports that rather than a 0.0 the reader
would take for "no impact". Half the point of the Phase 3 work was refusing to
invent denominators; that refusal has to survive into the UI or it was pointless.
"""

import logging
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from functools import lru_cache

from app.storage import db

logger = logging.getLogger(__name__)

_METRIC_COLUMNS = (
    "citation_count, influential_citation_count, reference_count, venue, "
    "cohort_year, primary_category, cnci, pagerank, in_degree"
)


@dataclass
class Scorecard:
    paper_id: str
    title: str

    citation_count: int | None
    influential_citation_count: int | None
    influential_ratio: float | None

    cnci: float | None
    cnci_percentile: float | None

    pagerank: float | None
    pagerank_percentile: float | None
    in_degree: int | None

    venue: str | None
    cohort_year: int | None
    primary_category: str | None

    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _distributions() -> dict[str, list[float]]:
    """Sorted value lists for percentile lookup.

    Loaded once and held: this is read for every paper in every result list, and the
    tables only change when the corpus is rebuilt. Sorted lists plus bisect rather
    than a percentile library - it is a few thousand floats and an O(log n) lookup.

    NULLs are excluded rather than treated as zero. A paper with no computed CNCI is
    not a paper with the lowest CNCI, and folding the two together would push every
    real value' percentile up by however many were simply never measured.
    """
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT citation_count, cnci, pagerank FROM paper_metrics"
        ).fetchall()

    dists = {
        "citation_count": sorted(r["citation_count"] for r in rows if r["citation_count"] is not None),
        "cnci": sorted(r["cnci"] for r in rows if r["cnci"] is not None),
        "pagerank": sorted(r["pagerank"] for r in rows if r["pagerank"] is not None),
    }
    logger.info(
        "scorecard distributions: %s",
        ", ".join(f"{k}={len(v)}" for k, v in dists.items()),
    )
    return dists


def percentile_of(value: float | None, metric: str) -> float | None:
    """Where `value` falls in the corpus distribution of `metric`, 0-100.

    Ties are resolved to the MIDPOINT of the tied block, not its lower edge. This is
    not a refinement - it is required for correctness here, because these
    distributions are dominated by ties. 7,670 of 8,770 papers share the identical
    floor PageRank of (1-d)/N, the value the algorithm assigns to a node with no
    incoming edges. Taking the lower edge reports every one of those as the 0th
    percentile, i.e. as the single least central paper in the corpus, when in truth
    they are indistinguishable from each other and sit below only the ~1,100 papers
    the graph actually connects.
    """
    if value is None:
        return None
    values = _distributions().get(metric) or []
    if not values:
        return None
    lo, hi = bisect_left(values, value), bisect_right(values, value)
    return round(100.0 * (lo + hi) / 2 / len(values), 1)


def _ordinal_clause(percentile: float | None) -> str:
    """Percentiles read better as "top N%" once they are extreme, which is exactly
    where scientometric values live - "99.9th percentile" and "top 0.1%" are the same
    fact, and the second is the one a reader feels."""
    if percentile is None:
        return ""
    top = 100.0 - percentile
    if top < 1.0:
        return f"top {top:.1f}% of the corpus"
    return f"{percentile:.0f}th percentile"


def _summarize(card: "Scorecard") -> str:
    """One sentence a professor can read without a legend.

    Deliberately built from whatever is present rather than requiring a full set: a
    2026 preprint has citations of 0 and no CNCI, and the honest sentence for it is
    short, not padded with placeholders.
    """
    parts: list[str] = []

    if card.citation_count is not None:
        cites = f"{card.citation_count:,} citation{'s' if card.citation_count != 1 else ''}"
        if card.influential_ratio is not None and card.influential_citation_count:
            cites += f" ({card.influential_citation_count:,} judged substantive)"
        parts.append(cites)

    if card.cnci is not None and card.cohort_year is not None:
        field = card.primary_category or "all fields"
        parts.append(f"{card.cnci:,.1f}x the {card.cohort_year} median for {field}")
    elif card.cohort_year is not None:
        parts.append(f"{card.cohort_year} paper, too few cohort peers to normalize")

    if card.pagerank is not None:
        centrality = "central in the citation network"
        rank_clause = _ordinal_clause(card.pagerank_percentile)
        if rank_clause:
            centrality += f" ({rank_clause})"
        if card.in_degree:
            centrality += f", cited by {card.in_degree:,} papers in this corpus"
        parts.append(centrality)

    if card.venue:
        parts.append(f"published at {card.venue}")

    return "; ".join(parts) if parts else "No citation metrics available for this paper yet."


def _build(paper_id: str, title: str, row: dict | None) -> Scorecard:
    row = row or {}
    citations = row.get("citation_count")
    influential = row.get("influential_citation_count")
    cnci = row.get("cnci")
    pagerank = row.get("pagerank")

    card = Scorecard(
        paper_id=paper_id,
        title=title,
        citation_count=citations,
        influential_citation_count=influential,
        # Guarded against the 0-citation case rather than left to raise: brand-new
        # preprints are the common case in a corpus that crawls the current year.
        influential_ratio=(influential / citations) if citations and influential is not None else None,
        cnci=cnci,
        cnci_percentile=percentile_of(cnci, "cnci"),
        pagerank=pagerank,
        pagerank_percentile=percentile_of(pagerank, "pagerank"),
        in_degree=row.get("in_degree"),
        venue=row.get("venue"),
        cohort_year=row.get("cohort_year"),
        primary_category=row.get("primary_category"),
        summary="",
    )
    card.summary = _summarize(card)
    return card


def get(paper_id: str) -> Scorecard | None:
    """Scorecard for one paper, or None if the paper itself is unknown.

    A paper present in the corpus but with no metrics row still returns a card - the
    summary then says so. That distinction matters to the caller: "we have never
    heard of this paper" and "we have it but Semantic Scholar has not indexed it" are
    different answers, and collapsing both to None would make the second unreportable.

    The returned card carries the id the CALLER asked for, matching `get_many`. The
    corpus may store the paper under a different version string, and handing that back
    instead would mean a caller who asked about `1706.03762` receives a card labelled
    `1706.03762v7` - which reads as a different paper, and which the agent would then
    quote as the citation id in its answer.
    """
    with db.connect() as conn:
        paper = conn.execute(
            "SELECT paper_id, title FROM papers WHERE paper_id = ? OR canonical_id = ? LIMIT 1",
            (paper_id, db.canonical_paper_id(paper_id)),
        ).fetchone()
        if paper is None:
            return None
        metrics = conn.execute(
            f"SELECT {_METRIC_COLUMNS} FROM paper_metrics WHERE paper_id = ?", (paper["paper_id"],)
        ).fetchone()

    return _build(paper_id, paper["title"], dict(metrics) if metrics else None)


def get_many(paper_ids: list[str]) -> dict[str, Scorecard]:
    """Scorecards for a result list, in one pass rather than one query per paper.

    Result lists are the main consumer - every search returns 5-10 papers, and a
    per-paper round trip would make the metrics display cost more than the retrieval
    that produced it.

    Keyed by the id the CALLER asked for, even when the corpus stores that paper
    under a different version string. Callers hold ids that came from a chunk, a
    citation edge or a URL, and those disagree about versions freely - `1706.03762`
    from one path and `1706.03762v7` from another. Returning the corpus's spelling
    would make the result unlookupable by the caller and silently render the card as
    missing.
    """
    if not paper_ids:
        return {}

    # Matched on canonical id, so a bare request finds a versioned row and vice versa.
    wanted = {paper_id: db.canonical_paper_id(paper_id) for paper_id in paper_ids}
    placeholders = ",".join("?" * len(wanted))
    canonicals = list(wanted.values())

    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT p.paper_id, p.title, COALESCE(p.canonical_id, p.paper_id) AS cid, "
            f"       {', '.join('m.' + c.strip() for c in _METRIC_COLUMNS.split(','))} "
            f"FROM papers p LEFT JOIN paper_metrics m ON m.paper_id = p.paper_id "
            f"WHERE COALESCE(p.canonical_id, p.paper_id) IN ({placeholders})",
            canonicals,
        ).fetchall()

    # Several corpus rows can share a canonical id (a bare and a versioned copy of the
    # same paper). Centrality is reconciled across them by graph.store_scores, so any
    # row will do; first wins, deterministically ordered by the query above.
    by_canonical: dict[str, dict] = {}
    for row in rows:
        by_canonical.setdefault(row["cid"], dict(row))

    cards: dict[str, Scorecard] = {}
    for requested, canonical in wanted.items():
        row = by_canonical.get(canonical)
        if row is None:
            continue
        cards[requested] = _build(requested, row["title"], row)
    return cards
