"""Citation-graph structure: PageRank centrality, co-citation, bibliographic coupling.

Citation counts measure how OFTEN a paper is cited. These measure something citation
counts cannot see: WHERE a paper sits in the network, and which papers the field
treats as belonging together.

PageRank (Page et al., 1999) is the same algorithm Google used on web links, applied
to citations: a paper is important if important papers cite it, recursively. This is
genuinely different from a raw count - twenty citations from foundational papers
outweigh a hundred from forgettable ones, and no amount of counting reveals that.

Co-citation (Small, 1973) and bibliographic coupling (Kessler, 1963) both measure
relatedness, in opposite temporal directions:

  co-citation   A and B are cited TOGETHER by later papers. The field treats them as
                related. Looks forward, so it takes years to accumulate - a paper
                published last month has none.
  coupling      A and B cite the SAME earlier papers. They share intellectual
                ancestry. Looks backward, and a paper's reference list exists on the
                day it is published, so this works immediately for brand-new work.

They complement each other precisely where the other is blind, which is why both are
here rather than just the more familiar one.

The graph is the INDUCED SUBGRAPH over corpus papers: an edge is kept only when both
endpoints are papers we hold. References out to the wider literature are dropped, so
centrality means "central within deep learning as sampled here" - a well-defined,
honest claim. Keeping external endpoints would add hundreds of thousands of leaf
nodes that can never be ranked or returned, diluting the scores of everything real.
"""

import logging
from dataclasses import dataclass

import networkx as nx

from app.storage import db

logger = logging.getLogger(__name__)

# PageRank's damping factor, the standard 0.85 from the original paper: the
# probability of following a citation rather than jumping to a random paper.
DAMPING = 0.85


@dataclass
class GraphStats:
    nodes: int
    edges: int
    isolated: int

    def __str__(self) -> str:
        connected = self.nodes - self.isolated
        return f"{self.nodes} nodes, {self.edges} edges, {connected} connected ({self.isolated} isolated)"


def s2_to_corpus_id() -> dict[str, str]:
    """Maps Semantic Scholar paperIds to CANONICAL corpus paper ids.

    Needed because references come back as S2 ids while the corpus is keyed by arXiv
    id. Papers without an s2_paper_id simply cannot be graph endpoints.

    Canonical rather than raw, because `1810.04805` and `1810.04805v2` are the same
    paper and share one S2 id. Mapping to raw ids made this dict's construction
    order decide which of the two received every citation edge, leaving the other a
    zero-degree node - see the note on `load_graph`.
    """
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT COALESCE(p.canonical_id, m.paper_id) AS cid, m.s2_paper_id "
            "FROM paper_metrics m JOIN papers p ON p.paper_id = m.paper_id "
            "WHERE m.s2_paper_id IS NOT NULL"
        ).fetchall()
    return {row["s2_paper_id"]: row["cid"] for row in rows}


def store_edges(edges: list[tuple[str, str]]) -> int:
    """Persists citing -> cited edges, ignoring duplicates."""
    with db.connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO citations (citing_paper_id, cited_paper_id) VALUES (?, ?)", edges
        )
    return len(edges)


def load_graph() -> nx.DiGraph:
    """Builds the directed citation graph from stored edges.

    Edge direction is citing -> cited, which is what PageRank needs: importance should
    flow backwards along citations, from the paper doing the citing to the paper being
    cited. Reversing this would rank papers by how many references they have, which
    measures bibliography length rather than influence.
    """
    graph = nx.DiGraph()
    with db.connect() as conn:
        graph.add_nodes_from(
            row["cid"]
            for row in conn.execute(
                "SELECT DISTINCT COALESCE(canonical_id, paper_id) AS cid FROM papers"
            )
        )
        graph.add_edges_from(
            (row["citing"], row["cited"])
            for row in conn.execute(
                """
                SELECT COALESCE(pc.canonical_id, c.citing_paper_id) AS citing,
                       COALESCE(pd.canonical_id, c.cited_paper_id)  AS cited
                FROM citations c
                LEFT JOIN papers pc ON pc.paper_id = c.citing_paper_id
                LEFT JOIN papers pd ON pd.paper_id = c.cited_paper_id
                """
            )
        )
    # A paper cannot cite itself, but two VERSIONS of it can appear on either end of
    # an edge, and collapsing to canonical ids turns that into a self-loop. PageRank
    # would treat it as a node endorsing itself, inflating exactly the well-known
    # papers most likely to have multiple indexed versions.
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def graph_stats(graph: nx.DiGraph) -> GraphStats:
    isolated = sum(1 for _, degree in graph.degree() if degree == 0)
    return GraphStats(graph.number_of_nodes(), graph.number_of_edges(), isolated)


def compute_pagerank(graph: nx.DiGraph) -> dict[str, float]:
    """PageRank over the citation graph, keyed by paper_id."""
    if graph.number_of_edges() == 0:
        logger.warning("Citation graph has no edges - PageRank would be uniform, skipping")
        return {}
    return nx.pagerank(graph, alpha=DAMPING)


def co_citation(graph: nx.DiGraph, paper_id: str, limit: int = 20) -> list[tuple[str, int]]:
    """Papers most often cited alongside `paper_id`, with the count.

    Walks backward to everything citing this paper, then forward to what else those
    papers cited. Each shared citing paper is one unit of co-citation strength.
    """
    if paper_id not in graph:
        return []

    strength: dict[str, int] = {}
    for citing in graph.predecessors(paper_id):
        for also_cited in graph.successors(citing):
            if also_cited != paper_id:
                strength[also_cited] = strength.get(also_cited, 0) + 1

    return sorted(strength.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def bibliographic_coupling(graph: nx.DiGraph, paper_id: str, limit: int = 20) -> list[tuple[str, int]]:
    """Papers sharing the most references with `paper_id`, with the overlap count.

    The mirror image of co-citation: forward to this paper's references, then backward
    to who else cites them. Works for papers too new to have been cited at all.
    """
    if paper_id not in graph:
        return []

    own_refs = set(graph.successors(paper_id))
    if not own_refs:
        return []

    overlap: dict[str, int] = {}
    for ref in own_refs:
        for other in graph.predecessors(ref):
            if other != paper_id:
                overlap[other] = overlap.get(other, 0) + 1

    return sorted(overlap.items(), key=lambda kv: kv[1], reverse=True)[:limit]


def store_scores(pagerank: dict[str, float], in_degree: dict[str, int]) -> int:
    """Persists centrality scores onto paper_metrics.

    Scores are computed per canonical paper but written to EVERY row sharing that
    canonical id, so a versioned and a bare row of the same paper report identical
    centrality. Without this, which score a query sees would depend on which version
    of the paper retrieval happened to return - and since the ranker reads centrality
    as a feature, that made citation evidence silently invisible for whichever twin
    the graph had not attached its edges to.
    """
    scores = {
        cid: (pagerank.get(cid), in_degree.get(cid, 0))
        for cid in set(pagerank) | set(in_degree)
    }
    with db.connect() as conn:
        rows = [
            (*scores[row["cid"]], row["paper_id"])
            for row in conn.execute(
                "SELECT paper_id, COALESCE(canonical_id, paper_id) AS cid FROM papers"
            )
            if row["cid"] in scores
        ]
        conn.executemany(
            "UPDATE paper_metrics SET pagerank = ?, in_degree = ? WHERE paper_id = ?", rows
        )
    return len(rows)
