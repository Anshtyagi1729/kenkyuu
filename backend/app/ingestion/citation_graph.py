"""A paper's direct citation neighborhood: what it cites (references) and what
cites it (citations), resolved to our own paper_id scheme.

Node identity: prefer the arXiv id when a neighbor has one, since that's what
the rest of the app (paper detail pages, chat, folders) keys off - falling back
to Semantic Scholar's own id only for neighbors with no arXiv version. Neighbor
metadata gets upserted into our papers table as it's discovered, so clicking
into a graph node that was never explicitly searched for still resolves on the
paper detail page instead of 404ing.

No separate SQLite cache for the resolved graph - semantic_scholar_client's
own @cached responses already make a repeat view of the same paper's graph a
local, free lookup (same caching philosophy as the rest of the app).
"""

import re

from app.ingestion import semantic_scholar_client
from app.storage import db

MAX_NEIGHBORS_PER_DIRECTION = 20  # keeps the graph readable and the S2 request small


def _s2_lookup_id(paper: dict) -> str:
    """Semantic Scholar's API accepts a prefixed external id (ARXIV:xxx) as a
    paper_id directly, so an arXiv-sourced paper never needs its own S2 internal
    id resolved first."""
    if paper["source"] == "arxiv":
        bare_id = re.sub(r"v\d+$", "", paper["paper_id"])
        return f"ARXIV:{bare_id}"
    return paper["paper_id"]


def _node_id(neighbor: semantic_scholar_client.S2Paper) -> str:
    return neighbor.external_ids.get("ArXiv") or neighbor.paper_id


def _upsert_neighbor(neighbor: semantic_scholar_client.S2Paper) -> str:
    """Records a graph neighbour so its node can be opened, filling gaps only.

    Must never overwrite. This used the replacing upsert, so merely viewing a citation
    graph rewrote every corpus neighbour with Semantic Scholar's copy: 102 crawl papers
    had their arXiv submission dates replaced by a bare year (BERT, submitted 2018-10-11,
    showed "2019") and their arXiv abstracts by S2's. Returns the id of the existing row
    when there is one, so a node opens the paper the corpus already holds.
    """
    node_id = _node_id(neighbor)
    existing = db.get_paper(node_id)
    if existing is not None:
        node_id = existing["paper_id"]
    db.merge_external_paper(
        paper_id=node_id,
        source="arxiv" if neighbor.external_ids.get("ArXiv") else "semantic_scholar",
        title=neighbor.title,
        abstract=neighbor.abstract,
        authors=neighbor.authors,
        published=str(neighbor.year) if neighbor.year else None,
        external_ids=neighbor.external_ids,
    )
    return node_id


def get_graph(paper_id: str) -> dict:
    """Returns {"nodes": [{"paper_id", "title", "year"}, ...],
    "edges": [{"source", "target"}, ...]} for paper_id's direct references and
    citations. Edge direction is citing -> cited (an edge into paper_id means
    "cites paper_id"; an edge out of it means "paper_id cites this")."""
    paper = db.get_paper(paper_id)
    if paper is None:
        raise ValueError(f"unknown paper_id: {paper_id}")
    s2_id = _s2_lookup_id(paper)

    year = (paper["published"] or "")[:4]
    nodes = {paper_id: {"paper_id": paper_id, "title": paper["title"], "year": int(year) if year.isdigit() else None}}
    edges = []

    try:
        references = semantic_scholar_client.get_references(s2_id, limit=MAX_NEIGHBORS_PER_DIRECTION)
    except Exception:  # noqa: BLE001 - a citation-graph glitch shouldn't break the page, just show fewer edges
        references = []
    for ref in references:
        # Some Semantic Scholar records are stubs - a title with no stable id at all
        # (not even their own paperId) - unusable as a graph node either way.
        if not ref.title or not ref.paper_id:
            continue
        ref_id = _upsert_neighbor(ref)
        nodes.setdefault(ref_id, {"paper_id": ref_id, "title": ref.title, "year": ref.year})
        edges.append({"source": paper_id, "target": ref_id})

    try:
        citations = semantic_scholar_client.get_citations(s2_id, limit=MAX_NEIGHBORS_PER_DIRECTION)
    except Exception:  # noqa: BLE001
        citations = []
    for cit in citations:
        if not cit.title or not cit.paper_id:
            continue
        cit_id = _upsert_neighbor(cit)
        nodes.setdefault(cit_id, {"paper_id": cit_id, "title": cit.title, "year": cit.year})
        edges.append({"source": cit_id, "target": paper_id})

    return {"nodes": list(nodes.values()), "edges": edges}
