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

import json
import logging
import re

from app.ingestion import openalex_client, pipeline, semantic_scholar_client
from app.ranking import identity
from app.storage import db

MAX_NEIGHBORS_PER_DIRECTION = 20  # keeps the graph readable and the S2 request small

logger = logging.getLogger(__name__)


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


def _graph_from_s2(paper_id: str) -> dict:
    """Fallback graph from Semantic Scholar. Returns {"nodes": [{"paper_id", "title", "year"}, ...],
    "edges": [{"source", "target"}, ...]} for paper_id's direct references and
    citations. Edge direction is citing -> cited (an edge into paper_id means
    "cites paper_id"; an edge out of it means "paper_id cites this")."""
    paper = db.get_paper(paper_id)
    if paper is None:
        raise ValueError(f"unknown paper_id: {paper_id}")
    s2_id = _s2_lookup_id(paper)

    year = (paper["published"] or "")[:4]
    nodes = {paper_id: {"paper_id": paper_id, "title": paper["title"], "year": int(year) if year.isdigit() else None,
                        "relation": "center"}}
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
        nodes.setdefault(ref_id, {"paper_id": ref_id, "title": ref.title, "year": ref.year,
                                  "citation_count": ref.citation_count, "relation": "reference"})
        edges.append({"source": paper_id, "target": ref_id})

    try:
        citations = semantic_scholar_client.get_citations(s2_id, limit=MAX_NEIGHBORS_PER_DIRECTION)
    except Exception:  # noqa: BLE001
        citations = []
    for cit in citations:
        if not cit.title or not cit.paper_id:
            continue
        cit_id = _upsert_neighbor(cit)
        nodes.setdefault(cit_id, {"paper_id": cit_id, "title": cit.title, "year": cit.year,
                                  "citation_count": cit.citation_count, "relation": "citation"})
        edges.append({"source": cit_id, "target": paper_id})

    return {"nodes": list(nodes.values()), "edges": edges, "source": "semantic_scholar"}


def get_graph(paper_id: str) -> dict:
    """A paper's citation neighbourhood: up to 20 papers it cites and 20 that cite it.

    OpenAlex first, Semantic Scholar as the fallback. OpenAlex needs no key and answers in
    a few seconds; Semantic Scholar's per-paper endpoints answer 429 to unauthenticated
    callers, which made this page take around 25 seconds and come back nearly empty for
    most papers outside the corpus. Measured on the panel's own Springer paper: OpenAlex
    lists 42 references and 2 citing works, where the old path found one neighbour.
    """
    paper = db.get_paper(paper_id)
    if paper is None:
        raise ValueError(f"unknown paper_id: {paper_id}")
    try:
        graph = _graph_from_openalex(paper)
        if graph is not None:
            return graph
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail the page
        logger.info("OpenAlex graph unavailable for %s: %s", paper_id, str(exc)[:160])
    return _graph_from_s2(paper["paper_id"])


def _resolve_openalex_id(paper: dict) -> str | None:
    """This paper's OpenAlex id, by the most reliable route available.

    Stored id, then DOI, then the DataCite DOI arXiv assigns every preprint
    (10.48550/arXiv.<id>), then a title search accepted only on an exact title match -
    a near-miss here would draw the graph of a different paper under this one's name.
    """
    ids = json.loads(paper.get("external_ids") or "{}")
    if ids.get("OpenAlex"):
        return ids["OpenAlex"]
    if ids.get("DOI"):
        work = openalex_client.get_by_doi(ids["DOI"])
        if work:
            return work.openalex_id
    arxiv_id = ids.get("ArXiv") or (paper["paper_id"] if paper.get("source") == "arxiv" else None)
    if arxiv_id:
        work = openalex_client.get_by_doi(f"10.48550/arxiv.{db.canonical_paper_id(arxiv_id)}")
        if work:
            return work.openalex_id
    for work in openalex_client.search(paper["title"], limit=5):
        if identity.title_sequence(work.title) == identity.title_sequence(paper["title"]):
            return work.openalex_id
    return None


def _neighbour_id(work: openalex_client.OpenAlexWork) -> str | None:
    """The corpus id for a neighbouring work, recording it (without embedding) if new.

    An existing corpus row is reused - by arXiv id, then DOI - so a node opens the paper
    the corpus already holds, with its full metrics, rather than a thin duplicate.
    """
    if work.arxiv_id:
        existing = db.get_paper(work.arxiv_id)
        if existing:
            return existing["paper_id"]
    if work.doi:
        existing_id = db.find_paper_by_doi(work.doi)
        if existing_id:
            return existing_id
    return pipeline.record_openalex_work(work, index=False)


def _dedupe_works(works: list[openalex_client.OpenAlexWork]) -> list[openalex_client.OpenAlexWork]:
    """Drops repeat records of one paper. OpenAlex sometimes holds a preprint and its
    published version as two works, and BERT appeared twice among the Transformer's
    citing papers. Matched on title words in order, keeping the first (most-cited)."""
    seen: set[tuple[str, ...]] = set()
    unique = []
    for work in works:
        key = identity.title_sequence(work.title)
        if key and key not in seen:
            seen.add(key)
            unique.append(work)
    return unique


def _graph_from_openalex(paper: dict) -> dict | None:
    oa_id = _resolve_openalex_id(paper)
    if oa_id is None:
        return None
    work = openalex_client.get_work(oa_id)
    if work is None:
        return None

    center_id = paper["paper_id"]
    center_key = identity.title_sequence(paper["title"])
    references = sorted(
        openalex_client.works_by_ids(work.referenced_works) if work.referenced_works else [],
        key=lambda w: w.citation_count or 0,
        reverse=True,
    )
    citing = openalex_client.citing_works(oa_id, MAX_NEIGHBORS_PER_DIRECTION + 5)

    nodes = {center_id: {
        "paper_id": center_id,
        "title": paper["title"],
        "year": work.year,
        "citation_count": work.citation_count,
        "relation": "center",
    }}
    edges = []
    for relation, works in (("reference", references), ("citation", citing)):
        kept = 0
        for neighbour in _dedupe_works(works):
            if kept >= MAX_NEIGHBORS_PER_DIRECTION:
                break
            if identity.title_sequence(neighbour.title) == center_key:
                continue  # another record of the centre paper itself
            node_id = _neighbour_id(neighbour)
            if not node_id or node_id == center_id or node_id in nodes:
                continue
            nodes[node_id] = {
                "paper_id": node_id,
                "title": neighbour.title,
                "year": neighbour.year,
                "citation_count": neighbour.citation_count,
                "relation": relation,
            }
            edges.append(
                {"source": center_id, "target": node_id} if relation == "reference"
                else {"source": node_id, "target": center_id}
            )
            kept += 1

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "source": "openalex",
        "total_references": len(work.referenced_works),
        "total_citations": work.citation_count,
    }
