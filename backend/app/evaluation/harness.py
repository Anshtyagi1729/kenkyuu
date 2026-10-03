"""Runs a retrieval system against the evaluation set and scores it.

The central abstraction is `RetrievalFn`: any callable taking (query, k) and
returning ranked paper ids. Every configuration we compare - dense-only today,
hybrid next week, the learned ranker after that - plugs in through that one seam,
so the rows of the results table are produced by identical scoring code and are
genuinely comparable rather than each being measured slightly differently.
"""

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from app.storage.db import canonical_paper_id

# (query, k) -> ranked paper ids, best first.
RetrievalFn = Callable[[str, int], list[str]]

NDCG_K = 10
RECALL_K = 20
# Retrieve deeper than any metric needs so Recall@20 isn't silently capped by a
# system that was only asked for 10 results.
RETRIEVE_K = 50


@dataclass
class EvalQuery:
    id: str
    query: str
    intent: str
    judgements: dict[str, int]
    note: str = ""


@dataclass
class QueryScore:
    query_id: str
    intent: str
    ndcg: float
    recall: float
    rr: float
    top_hit: str | None


@dataclass
class EvalReport:
    ndcg: float
    recall: float
    mrr: float
    per_query: list[QueryScore] = field(default_factory=list)

    @property
    def by_intent(self) -> dict[str, dict[str, float]]:
        """Scores broken out by query intent.

        Worth reporting separately because the intents fail differently: text-only
        retrieval is weakest on find-this-specific-paper queries, so an aggregate
        average can hide both a large win and a large regression at once.
        """
        grouped: dict[str, list[QueryScore]] = defaultdict(list)
        for score in self.per_query:
            grouped[score.intent].append(score)

        return {
            intent: {
                "ndcg": _mean(s.ndcg for s in scores),
                "recall": _mean(s.recall for s in scores),
                "mrr": _mean(s.rr for s in scores),
                "n": len(scores),
            }
            for intent, scores in sorted(grouped.items())
        }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _dedupe(paper_ids) -> list[str]:
    """Keeps each paper's best (first) position, dropping later repeats."""
    seen, ranked = set(), []
    for paper_id in paper_ids:
        if paper_id not in seen:
            seen.add(paper_id)
            ranked.append(paper_id)
    return ranked


def load_eval_set(path: str | Path) -> list[EvalQuery]:
    """Loads queries, canonicalizing judgement ids so a versioned id in the corpus
    still matches a bare id written in the eval file."""
    data = json.loads(Path(path).read_text())
    return [
        EvalQuery(
            id=entry["id"],
            query=entry["query"],
            intent=entry["intent"],
            judgements={canonical_paper_id(pid): grade for pid, grade in entry["relevant"].items()},
            note=entry.get("note", ""),
        )
        for entry in data["queries"]
    ]


def evaluate(system: RetrievalFn, queries: list[EvalQuery], retrieve_k: int = RETRIEVE_K) -> EvalReport:
    scores: list[QueryScore] = []

    for query in queries:
        # Canonicalize what the system returns too: the corpus stores versioned arXiv
        # ids, the eval set is written with bare ones, and without this every judgement
        # would silently miss and every system would score a flat zero.
        #
        # Then DEDUPE, because canonicalizing can create duplicates that were not there
        # before: the corpus holds `1810.04805` and `1810.04805v2` as separate rows,
        # both indexed, so a system can legitimately return both as distinct results
        # and only this mapping reveals them to be one paper. Left in, the same paper
        # occupies two ranks and its relevance grade is counted twice, which pushes DCG
        # above the ideal DCG - nDCG@10 of 1.0323 was observed on the specific_paper
        # intent, which is impossible by construction and is what exposed this.
        #
        # It inflated exactly the wrong thing: the duplicated papers are the landmarks,
        # which are the queries the citation features are supposed to win, so the bug
        # flattered every system that ranked them well.
        ranked = _dedupe(canonical_paper_id(pid) for pid in system(query.query, retrieve_k))

        scores.append(
            QueryScore(
                query_id=query.id,
                intent=query.intent,
                ndcg=ndcg_at_k(ranked, query.judgements, NDCG_K),
                recall=recall_at_k(ranked, query.judgements, RECALL_K),
                rr=reciprocal_rank(ranked, query.judgements),
                top_hit=ranked[0] if ranked else None,
            )
        )

    return EvalReport(
        ndcg=_mean(s.ndcg for s in scores),
        recall=_mean(s.recall for s in scores),
        mrr=_mean(s.rr for s in scores),
        per_query=scores,
    )


def format_report(report: EvalReport, label: str) -> str:
    """Human-readable summary, including the worst queries - an aggregate score tells
    you whether something improved, but the failures tell you what to fix next."""
    lines = [
        f"{label}",
        f"  {f'nDCG@{NDCG_K}':<10}{report.ndcg:.4f}",
        f"  {f'Recall@{RECALL_K}':<10}{report.recall:.4f}",
        f"  {'MRR':<10}{report.mrr:.4f}",
        f"  {'queries':<10}{len(report.per_query)}",
        "",
        "  By intent:",
    ]
    for intent, stats in report.by_intent.items():
        lines.append(
            f"    {intent:<16} n={stats['n']:<3} nDCG={stats['ndcg']:.4f}  "
            f"Recall={stats['recall']:.4f}  MRR={stats['mrr']:.4f}"
        )

    worst = sorted(report.per_query, key=lambda s: s.ndcg)[:5]
    lines += ["", "  Weakest queries:"]
    lines += [f"    {s.ndcg:.3f}  {s.query_id:<28} top hit: {s.top_hit or '(nothing)'}" for s in worst]

    return "\n".join(lines)
