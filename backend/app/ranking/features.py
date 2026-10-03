"""Feature extraction for the learned ranker.

One row per (query, candidate paper). The model's job is to learn how to weight
these, and specifically WHEN to trust the citation features - the evaluation showed
that applying them uniformly trades accuracy on famous papers against accuracy on
everything else, so a useful model must condition rather than apply a fixed weight.

Conditioning needs a signal for what kind of query this is. An LLM intent classifier
exists (llm/grading.py) and is accurate, but calling it once per training example
means thousands of API calls against a free tier. So intent is approximated here by
cheap QUERY-SHAPE features - length, recency words, comparison words, and how much
the query overlaps the candidate's title. A gradient-boosted tree can split on those
to recover most of the conditioning, and unlike an LLM call they cost nothing at
inference time either.

Citation features are stored as log1p. Raw counts span six orders of magnitude, and
while trees are scale-invariant in principle, log-scaling makes the split points
meaningful (the gap between 10 and 100 citations matters more than between 10,000 and
10,090) and keeps the same feature usable by a linear model for comparison.
"""

import math
import re
from dataclasses import asdict, dataclass

RECENCY_WORDS = ("recent", "latest", "new", "newest", "current", "emerging", "this year", "2024", "2025", "2026")
COMPARISON_WORDS = ("compare", "versus", " vs ", "difference between", "contrast")
FOUNDATIONAL_WORDS = ("original", "introduced", "first", "seminal", "foundational", "proposed by")

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Features:
    # --- text relevance ---
    rerank_score: float          # cross-encoder, the strongest single signal
    bm25_score: float            # lexical overlap
    title_overlap: float         # fraction of query words appearing in the title

    # --- citation evidence ---
    log_citations: float
    log_cnci: float              # field-normalized; 0.0 when not computable
    log_pagerank: float
    log_in_degree: float
    has_cnci: int                # explicit missingness flag, see note below
    influential_ratio: float     # substantive citations / all citations

    # --- paper properties ---
    age_years: float
    has_venue: int

    # --- query shape (stands in for intent) ---
    query_words: int
    is_recency_query: int
    is_comparison_query: int
    is_foundational_query: int

    def to_list(self) -> list[float]:
        return [float(v) for v in asdict(self).values()]

    @staticmethod
    def names() -> list[str]:
        return list(Features.__dataclass_fields__)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _log1p(value) -> float:
    """log1p, treating missing as 0. Paired with an explicit has_* flag wherever
    missingness is itself informative - otherwise the model cannot distinguish
    "no citations" from "we never computed it", which are different claims."""
    return math.log1p(value) if value else 0.0


def extract(
    query: str,
    candidate: dict,
    metrics: dict | None,
    current_year: int = 2026,
    bm25_score: float = 0.0,
) -> Features:
    metrics = metrics or {}
    query_tokens = _tokens(query)
    title_tokens = _tokens(candidate.get("title", ""))
    overlap = len(query_tokens & title_tokens) / len(query_tokens) if query_tokens else 0.0

    citations = metrics.get("citation_count") or 0
    influential = metrics.get("influential_citation_count") or 0
    cnci = metrics.get("cnci")
    year = metrics.get("cohort_year")
    lowered = f" {query.lower()} "

    return Features(
        rerank_score=float(candidate.get("rerank_score") or 0.0),
        bm25_score=float(bm25_score),
        title_overlap=overlap,
        log_citations=_log1p(citations),
        log_cnci=_log1p(cnci),
        log_pagerank=_log1p((metrics.get("pagerank") or 0.0) * 1e4),  # scale up: raw values are ~1e-5
        log_in_degree=_log1p(metrics.get("in_degree") or 0),
        has_cnci=int(cnci is not None),
        influential_ratio=(influential / citations) if citations else 0.0,
        age_years=float(current_year - year) if year else 0.0,
        has_venue=int(bool(metrics.get("venue"))),
        query_words=len(query.split()),
        is_recency_query=int(any(w in lowered for w in RECENCY_WORDS)),
        is_comparison_query=int(any(w in lowered for w in COMPARISON_WORDS)),
        is_foundational_query=int(any(w in lowered for w in FOUNDATIONAL_WORDS)),
    )
