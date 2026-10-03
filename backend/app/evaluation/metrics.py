"""Ranking metrics: nDCG, Recall@k, MRR.

Pure functions over (ranked ids, relevance judgements). No retrieval, no I/O, no
database - which keeps them trivially testable and means the same code scores every
system configuration we compare, so the numbers in the results table are genuinely
comparable rather than each system being scored by its own slightly different code.

Relevance is graded, not binary:
    3  this is the paper being asked for
    2  highly relevant
    1  marginally relevant
    0  irrelevant (the default for anything unjudged)

Graded judgements matter for a research-paper corpus: "returned a good survey of the
topic" and "returned the exact paper requested" are both successes, but not equal
ones, and a binary metric is blind to the difference.
"""

import math

Judgements = dict[str, int]


def dcg(gains: list[int]) -> float:
    """Discounted Cumulative Gain over an already-ordered list of relevance grades.

    Each result's gain is discounted by log2 of its position, so a relevant paper at
    rank 1 counts for more than the same paper at rank 10. Position is 1-indexed:
    rank 1 divides by log2(2) = 1 (no discount), rank 2 by log2(3), and so on.
    """
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: list[str], judgements: Judgements, k: int = 10) -> float:
    """Normalized DCG - the system's DCG divided by the best DCG achievable.

    Normalizing is what makes scores comparable across queries: a query with six
    relevant papers can reach a far higher raw DCG than one with a single relevant
    paper, and without normalization the easy queries would dominate the average.

    Returns 0.0 when a query has no relevant papers at all, since there is nothing
    to be right about and a perfect score would be meaningless.
    """
    gains = [judgements.get(paper_id, 0) for paper_id in ranked_ids[:k]]
    ideal_gains = sorted(judgements.values(), reverse=True)[:k]

    ideal = dcg(ideal_gains)
    return dcg(gains) / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_ids: list[str], judgements: Judgements, k: int = 20) -> float:
    """Fraction of all relevant papers that appear anywhere in the top k.

    Position-blind on purpose: this answers "did we find it at all", which is a
    different question from nDCG's "did we rank it well". A system can have good
    recall and bad nDCG (it finds the right papers but buries them), and that
    distinction tells you whether to fix retrieval or fix ranking.
    """
    relevant = {pid for pid, grade in judgements.items() if grade > 0}
    if not relevant:
        return 0.0
    found = relevant.intersection(ranked_ids[:k])
    return len(found) / len(relevant)


def reciprocal_rank(ranked_ids: list[str], judgements: Judgements) -> float:
    """1 / (rank of the first relevant result), or 0.0 if none is relevant.

    Only the first hit counts, which makes this the right metric for
    find-me-this-specific-paper queries: for those there is exactly one correct
    answer and burying it at rank 8 is barely better than not finding it.
    """
    for rank, paper_id in enumerate(ranked_ids, start=1):
        if judgements.get(paper_id, 0) > 0:
            return 1.0 / rank
    return 0.0
