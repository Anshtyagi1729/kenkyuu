"""Training pairs for the learned ranker, built without popularity bias.

The obvious source - citation edges, where a citing paper's context is the query and
the cited paper the answer - turns out to be unusable here, and the reason is worth
recording because it is not obvious.

Our citation graph is the INDUCED subgraph over the corpus: an edge survives only if
both endpoints are papers we hold. A randomly sampled paper's obscure references are
almost never also in a 8,770-paper sample, but its famous references (Attention,
BERT, Adam) almost always are, because those were seeded deliberately. The result is
that 72.8% of edges point at top-1% cited papers and only 1.0% at below-median ones.

Training on that distribution teaches precisely the shortcut the evaluation showed to
be harmful - "rank by fame" - and an ablation would then certify citation count as
the most valuable feature. The bias would survive every check that did not
specifically look for it.

So the primary signal here is SYNTHETIC QUERIES instead: a sentence from a paper's
abstract becomes a query whose correct answer is that paper. Two properties matter:

  1. No fame bias by construction. Every paper is equally likely to be a target,
     because every paper has an abstract. The distribution of target fame matches the
     corpus itself rather than the citation graph's skew.
  2. Unlimited and free. No API calls, no rate limits, ~8,700 papers available.

The trade-off is honest: a sentence lifted from an abstract is a less natural query
than a real citation context. It buys unbiasedness at some cost in realism, which is
the right trade when the bias is the thing that invalidates the result.
"""

import logging
import random
import re
from dataclasses import dataclass, field

from app.storage import db

logger = logging.getLogger(__name__)

# Sentences shorter than this are fragments ("We evaluate on three datasets.") that
# identify nothing; much longer ones are usually two clauses and read as a summary
# rather than a question someone would type.
MIN_QUERY_CHARS = 45
MAX_QUERY_CHARS = 220

# Sentences that describe the paper's contribution make the best queries. Openers
# about background ("Recent work has shown...") describe the field, not this paper,
# and would be correct answers for hundreds of papers at once.
CONTRIBUTION_MARKERS = (
    "we propose", "we introduce", "we present", "this paper", "we develop",
    "we show", "our method", "our approach", "we describe", "in this work",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TrainingExample:
    """One query with its known-correct paper."""

    query: str
    positive_paper_id: str
    source: str  # "synthetic" | "citation"
    negatives: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


def _query_from_abstract(abstract: str) -> str | None:
    """Picks the sentence that best identifies this specific paper.

    Prefers a contribution sentence; falls back to any sentence of workable length.
    Returns None when nothing usable is found, rather than emitting a fragment that
    would train the model on noise.
    """
    candidates = [s for s in _sentences(abstract) if MIN_QUERY_CHARS <= len(s) <= MAX_QUERY_CHARS]
    if not candidates:
        return None

    for sentence in candidates:
        lowered = sentence.lower()
        if any(marker in lowered for marker in CONTRIBUTION_MARKERS):
            return sentence

    # No contribution sentence - the second sentence is usually more specific than
    # the first, which tends to be scene-setting.
    return candidates[1] if len(candidates) > 1 else candidates[0]


def build_synthetic(limit: int | None = None, seed: int = 0) -> list[TrainingExample]:
    """Generates (query, correct paper) pairs from abstracts across the whole corpus."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT paper_id, abstract FROM papers "
            "WHERE abstract IS NOT NULL AND LENGTH(abstract) > 200 ORDER BY paper_id"
        ).fetchall()

    examples = []
    for row in rows:
        query = _query_from_abstract(row["abstract"])
        if query:
            examples.append(TrainingExample(query, row["paper_id"], "synthetic"))

    random.Random(seed).shuffle(examples)
    if limit:
        examples = examples[:limit]

    logger.info("Built %d synthetic training examples from %d papers", len(examples), len(rows))
    return examples


def build_citation_balanced(per_bucket: int = 200, seed: int = 0) -> list[TrainingExample]:
    """Citation-derived pairs, resampled so target fame is spread rather than skewed.

    Real citation edges carry signal the synthetic queries cannot - they reflect what
    researchers actually look up. Including them REQUIRES rebalancing, because the raw
    distribution is 72.8% top-1% papers. Sampling evenly across fame buckets keeps the
    signal while removing the skew that makes it dangerous.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT c.citing_paper_id, c.cited_paper_id, m.citation_count, p.abstract
            FROM citations c
            JOIN paper_metrics m ON m.paper_id = c.cited_paper_id
            JOIN papers p ON p.paper_id = c.citing_paper_id
            WHERE p.abstract IS NOT NULL AND m.citation_count IS NOT NULL
            """
        ).fetchall()

    buckets: dict[str, list] = {"low": [], "mid": [], "high": [], "top": []}
    for row in rows:
        count = row["citation_count"]
        key = "low" if count < 10 else "mid" if count < 100 else "high" if count < 1000 else "top"
        buckets[key].append(row)

    rng = random.Random(seed)
    examples = []
    for name, bucket in buckets.items():
        chosen = rng.sample(bucket, min(per_bucket, len(bucket)))
        logger.info("  citation bucket %-5s %d available -> %d sampled", name, len(bucket), len(chosen))
        for row in chosen:
            query = _query_from_abstract(row["abstract"])
            if query:
                examples.append(TrainingExample(query, row["cited_paper_id"], "citation"))

    rng.shuffle(examples)
    return examples


# How a person actually asks for a paper they know the name of. Each template is
# applied to a real title; the model needs to see that the framing words carry no
# information and the title does.
_TITLE_FRAMINGS = (
    "{title}",
    "the {title} paper",
    "find me {title}",
    "show me the paper {title}",
    "the original {title} paper",
)

# A recalled title is rarely exact, so some examples drop words. Dropping is the
# realistic corruption: people forget a subtitle or a qualifier, they do not usually
# insert words that were never there. The panel's own paper arrived as "error weighted
# drift ADOPTING ensemble framework" for "...drift-ADAPTIVE...", i.e. one wrong word
# inside an otherwise intact title.
_TITLE_DROP_FRACTIONS = (0.0, 0.0, 0.15, 0.3)


def build_title_lookups(limit: int | None = None, seed: int = 0) -> list[TrainingExample]:
    """Generates (title-as-query, that paper) pairs.

    This closes a gap that was costing the shipped ranker its most important query
    type. Both existing generators produce queries drawn from ABSTRACT prose - a
    sentence lifted from an abstract, or a citation context. Neither ever presents a
    TITLE as the query, so `title_overlap` near 1.0 was a feature value the model had
    almost no training signal for.

    The consequence was measurable and bad: asked "attention is all you need" or "LoRA
    low-rank adaptation of large language models" - the exact titles - the ranker
    demoted the correct paper to rank 3 and out of the top 5 respectively, from
    positions 2 and 1 in the dense pool. It had learned to lean on the cross-encoder,
    which for a title query prefers papers *about* the title. Descriptive queries
    ("the paper that introduced LoRA") worked, because those resemble the training
    distribution.

    Title lookup is the single most important query for the intended user: a professor
    searching for a specific paper, very often their own. It cannot be the one query
    shape absent from training.

    Deliberately NOT sampled toward famous papers. Every paper with a title is
    eligible, so the model learns "the title matches" as a signal in its own right
    rather than one entangled with fame - which is the same reason `build_synthetic`
    is preferred over raw citation contexts.
    """
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT paper_id, title FROM papers "
            "WHERE title IS NOT NULL AND LENGTH(title) > 20 ORDER BY paper_id"
        ).fetchall()

    rng = random.Random(seed)
    examples = []
    for row in rows:
        framing = rng.choice(_TITLE_FRAMINGS)
        drop = rng.choice(_TITLE_DROP_FRACTIONS)
        title = _drop_words(row["title"], drop, rng) if drop else row["title"]
        if not title:
            continue
        query = framing.format(title=title)
        # A title that already begins with an article would otherwise produce "the The
        # Drill-Down..." - a string no user types, which would only teach the model to
        # tolerate noise it will never see.
        query = re.sub(r"^(the|a|an)\s+(the|a|an)\s+", r"\2 ", query, flags=re.IGNORECASE)
        examples.append(TrainingExample(query, row["paper_id"], "title_lookup"))

    rng.shuffle(examples)
    if limit:
        examples = examples[:limit]

    logger.info("Built %d title-lookup training examples from %d papers", len(examples), len(rows))
    return examples


def _drop_words(title: str, fraction: float, rng: random.Random) -> str:
    """Removes a fraction of a title's words, preserving order.

    Order is preserved because a misremembered title is still roughly sequential; a
    shuffled bag of its words would train the model on an input shape no user
    produces. At least three words are always kept - below that the query stops
    identifying anything and the example would teach the model to guess.
    """
    words = title.split()
    keep_count = max(3, round(len(words) * (1 - fraction)))
    if keep_count >= len(words):
        return title
    keep_idx = sorted(rng.sample(range(len(words)), keep_count))
    return " ".join(words[i] for i in keep_idx)
