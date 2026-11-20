"""Title identity: deciding whether a candidate IS the paper a query names.

Shared by the agent (to label search results honestly) and by retrieval (to resolve
known-item queries). It lives here rather than in the agent because both need the
same definition of "this title matches", and two copies would drift.
"""

import re

# Words that carry no identifying information in a title lookup. Deliberately short:
# this is NOT general stopword removal (see lexical.tokenize for why that would be
# wrong here), only the framing words people wrap around a title when asking.
TITLE_NOISE = frozenset({
    "a", "an", "the", "of", "for", "in", "on", "and", "or", "to", "with", "via",
    "using", "based", "paper", "papers", "original", "find", "show", "me", "get",
    "about", "this", "that", "is", "are", "by", "from", "at", "as", "its",
})

# A known-item query must carry at least this many content words. "bert" alone is a
# name, not a title, and set-equality on one token would promote any paper whose
# title happens to be that single word.
MIN_KNOWN_ITEM_TOKENS = 3


def title_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in TITLE_NOISE}


def title_similarity(query: str, title: str) -> float:
    """Fraction of the query's content words that appear in the title.

    Asymmetric on purpose. A long title containing every word of a short query IS
    often the paper asked for, and dividing by the union (Jaccard) would punish it for
    having a subtitle. What matters is whether the query is accounted for.
    """
    q = title_tokens(query)
    if not q:
        return 0.0
    return len(q & title_tokens(title)) / len(q)


# Words people wrap AROUND a title when asking for it ("find me the paper ..."). Only
# stripped at the edges of the word sequence, never from the middle, where they belong
# to the title itself: the "is" in "Attention Is All You Need" is part of its identity.
EDGE_FRAMING = frozenset({
    "find", "me", "show", "get", "give", "the", "a", "an", "paper", "papers",
    "original", "please", "titled", "called", "article", "on", "about",
})


def title_sequence(text: str) -> tuple[str, ...]:
    """A title's words in order, lowercased, with framing words trimmed from both ends."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    start, end = 0, len(words)
    while start < end and words[start] in EDGE_FRAMING:
        start += 1
    while end > start and words[end - 1] in EDGE_FRAMING:
        end -= 1
    return tuple(words[start:end])


def is_known_item(query: str, title: str) -> bool:
    """True when the query is this paper's title, give or take framing words around it.

    Ordered sequence equality - not containment, and not set equality - and both of
    the weaker versions were tried and failed on the same famous paper:

    * Containment: "attention is all you need" is contained in "Not All Attention Is
      All You Need" and "Element-wise Attention Is All You Need", so every
      coverage-based measure scores all of them 1.0. That is why the learned ranker,
      whose title_overlap feature is exactly such a measure, put the real paper
      anywhere from rank 2 to rank 5 across eight retrains.
    * Set equality: "Is Attention All You Need?" has exactly the same words as
      "Attention Is All You Need" in a different order. As a result the federated
      search both labelled it an exact match and merged the two into one record - and
      which title survived the merge depended on which thread answered first.

    Word order is part of a title's identity. Framing words are trimmed from the ends
    of both strings identically, so "find me the paper attention is all you need"
    matches while nothing inside a title is ever discarded.
    """
    q = title_sequence(query)
    return len(q) >= MIN_KNOWN_ITEM_TOKENS and q == title_sequence(title)
