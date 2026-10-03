"""Finding related papers through citation structure rather than text similarity.

This answers a question the vector index structurally cannot. Dense retrieval finds
papers that TALK like the one you have; these find papers that sit in the same place
in the literature. Those differ constantly - a paper can share none of another's
vocabulary and still be its closest intellectual neighbour, because the same
community cites them together.

Two classical measures, both already implemented in graph.py and reached from here:

  co-citation (Small, 1973)        papers frequently cited ALONGSIDE this one.
                                   Forward-looking: it reflects how the field has
                                   come to group this work, so it sharpens as the
                                   paper accumulates citations - and says nothing at
                                   all about a paper nobody has cited yet.

  bibliographic coupling (Kessler, 1963)
                                   papers that SHARE REFERENCES with this one.
                                   Backward-looking, fixed at publication, and
                                   therefore the one that works for a paper from
                                   last week.

The pair is complementary by construction, which is why both are exposed rather than
one being picked as the default: which is usable depends on the age of the paper
being asked about, and that is knowable from the data rather than from the user.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

import networkx as nx

from app.scientometrics import graph
from app.storage import db

logger = logging.getLogger(__name__)

CO_CITATION = "co_citation"
COUPLING = "bibliographic_coupling"
METHODS = (CO_CITATION, COUPLING)


@dataclass
class RelatedPaper:
    paper_id: str
    title: str
    # Shared citing papers (co-citation) or shared references (coupling). Reported
    # rather than normalized into a 0-1 score: "cited alongside this one by 14 papers"
    # is checkable, and a similarity score between two papers is not.
    strength: int
    method: str


@lru_cache(maxsize=1)
def _graph() -> nx.DiGraph:
    """The citation graph, loaded once.

    Cached deliberately: both measures walk the graph twice per call, and rebuilding
    8,770 nodes from SQLite on every request would make this the slowest tool the
    agent has. The graph only changes when the corpus is rebuilt.
    """
    g = graph.load_graph()
    logger.info("citation graph loaded: %s", graph.graph_stats(g))
    return g


@lru_cache(maxsize=1)
def _titles() -> dict[str, str]:
    """Canonical paper id -> title. Keyed canonically to match the graph's nodes."""
    with db.connect() as conn:
        return {
            row["cid"]: row["title"]
            for row in conn.execute(
                "SELECT COALESCE(canonical_id, paper_id) AS cid, title FROM papers"
            )
        }


def find(paper_id: str, method: str = CO_CITATION, limit: int = 10) -> list[RelatedPaper]:
    """Papers related to `paper_id` by citation structure, strongest first.

    Returns an empty list when the paper is isolated in the graph - which is the
    common case, not an error: 5,784 of 8,770 corpus papers have no internal citation
    edges at all, because a sample of one field mostly does not cite itself. Callers
    must treat empty as "no citation evidence for this paper", never as "no related
    work exists".
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    canonical = db.canonical_paper_id(paper_id)
    g = _graph()
    fn = graph.co_citation if method == CO_CITATION else graph.bibliographic_coupling
    pairs = fn(g, canonical, limit=limit)

    titles = _titles()
    return [
        RelatedPaper(paper_id=pid, title=titles.get(pid, "(unknown title)"), strength=n, method=method)
        for pid, n in pairs
    ]


def find_best_effort(paper_id: str, limit: int = 10) -> tuple[list[RelatedPaper], str]:
    """Tries co-citation, falling back to bibliographic coupling.

    The fallback is the whole point rather than a safety net. Co-citation gives the
    better answer when it is available, but it is empty by definition for a paper
    nothing has cited yet - which describes every recent preprint, precisely the
    papers a professor is most likely to be asking about. Coupling needs only the
    paper's own reference list, which exists from the day it is published.

    Returns the results and which method produced them, so the caller can say which
    kind of relatedness it is reporting instead of presenting two different
    relationships as one undifferentiated "related" list.
    """
    results = find(paper_id, CO_CITATION, limit)
    if results:
        return results, CO_CITATION
    return find(paper_id, COUPLING, limit), COUPLING
