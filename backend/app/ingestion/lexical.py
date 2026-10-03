"""BM25 lexical retrieval over the chunk corpus.

Complements, rather than competes with, the dense retrieval in retrieval.py. The two
fail in opposite directions:

  dense    matches MEANING. Finds "how do I make attention use less memory" against a
           paper that never uses the word "memory". Blind to exact identity - a 2024
           paper discussing residual connections scores as well as the ResNet paper.

  BM25     matches WORDS. Nails "attention is all you need" because those exact tokens
           appear in exactly one title. Useless when the query and the paper use
           different vocabulary for the same idea.

The Phase 0 baseline showed precisely the failure BM25 addresses: correct papers
retrieved but ranked 11-14, outranked by recent papers on the same topic. Those
queries are near-verbatim titles, which is BM25's strongest case.

Index construction is in-memory and rebuilt on demand. At ~9k chunks that costs a
couple of seconds and a few tens of MB, which is far simpler than maintaining a
persistent inverted index - and this corpus grows in batches, not continuously.
"""

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.storage import db

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    Deliberately no stemming or stopword removal. Academic titles lean on short
    function words far more than general text does - "Attention Is All You Need" is
    almost entirely stopwords, and stripping them would leave "attention", destroying
    the exact-phrase signal that makes BM25 worth adding in the first place.
    """
    return _TOKEN.findall(text.lower())


@dataclass
class LexicalHit:
    chunk_id: str
    paper_id: str
    text: str
    score: float


class BM25Index:
    """In-memory BM25 over all stored chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._paper_ids: list[str] = []
        self._texts: list[str] = []

    def build(self) -> int:
        with db.connect() as conn:
            rows = conn.execute("SELECT chunk_id, paper_id, text FROM chunks ORDER BY chunk_id").fetchall()

        self._chunk_ids = [row["chunk_id"] for row in rows]
        self._paper_ids = [row["paper_id"] for row in rows]
        self._texts = [row["text"] for row in rows]
        self._bm25 = BM25Okapi([tokenize(text) for text in self._texts]) if rows else None

        logger.info("BM25 index built over %d chunks", len(rows))
        return len(rows)

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def search(self, query: str, top_k: int = 20) -> list[LexicalHit]:
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        # Take the top_k by score, then drop zeros: BM25 gives 0.0 to any document
        # sharing no query term, and padding the result with unrelated documents
        # would only add noise for the fusion step to sort back out.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            LexicalHit(
                chunk_id=self._chunk_ids[i],
                paper_id=self._paper_ids[i],
                text=self._texts[i],
                score=float(scores[i]),
            )
            for i in ranked
            if scores[i] > 0
        ]


_index: BM25Index | None = None


def get_index(rebuild: bool = False) -> BM25Index:
    """Process-wide index, built on first use.

    Cached because construction tokenizes every chunk in the corpus - doing that per
    query would dominate retrieval latency. Pass rebuild=True after ingesting new
    papers, since the index is a snapshot and won't otherwise see them.
    """
    global _index
    if _index is None or rebuild:
        _index = BM25Index()
        _index.build()
    return _index
