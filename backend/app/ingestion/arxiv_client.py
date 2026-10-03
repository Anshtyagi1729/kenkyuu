"""arXiv API client — search, metadata, PDF fetch.

arXiv's terms of use ask for at most one request per 3 seconds over a single
connection, so every call here goes through a shared rate limiter.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import feedparser
import requests

from app.cache import cached
from app.config import API_CACHE_TTL_SECONDS, CORPUS_CATEGORIES

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# arXiv rejects the default python-requests User-Agent with 406 Not Acceptable, and
# their API terms ask callers to identify themselves anyway. A contact URL is the
# convention so they can get in touch instead of silently blocking a misbehaving
# client. HTTPS directly, rather than relying on the http -> https redirect.
REQUEST_HEADERS = {
    "User-Agent": "paper-agent/0.1 (academic research project; +https://arxiv.org/help/api)",
}
MIN_REQUEST_INTERVAL = 3.0

# The throttle below is per-process, so it cannot coordinate with anything else
# talking to arXiv at the same time - a corpus crawl running alongside the live app
# is enough to trip a 429. Backing off and retrying keeps a long crawl from losing
# a whole cohort to one transient rejection.
#
# Delays are generous on purpose. arXiv doesn't rate-limit request-by-request; once
# you exceed their limits they refuse everything for a stretch of minutes. A budget
# measured in seconds just burns all its attempts inside that window and gives up
# while the door is still shut. 15s doubling over 5 attempts spans ~4 minutes, which
# actually outlasts a typical penalty.
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 15.0

_last_request_at = 0.0
_lock = Lock()


def _throttle() -> None:
    global _last_request_at
    with _lock:
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    pdf_url: str
    # arXiv's own primary classification, e.g. "cs.LG". Needed because field-normalized
    # impact compares a paper against its own category cohort - a paper stored without
    # its real category has no peer group and cannot be normalized at all.
    primary_category: str | None = None
    # True when this result came from the exact-phrase title search rather than the
    # broad OR-across-words search - a strong enough identity signal that callers
    # looking for one specific named paper can trust it over looser semantic ranking.
    is_title_match: bool = False


SORT_MODES = {
    # Best-matching text, regardless of age - the default, and the only mode
    # arXiv's own relevance ranking supports. Good for "the paper that defines X".
    "relevance": "relevance",
    # Newest first, ignoring match quality beyond the keyword filter - for
    # "recent/latest/new work on X". arXiv relevance-ranking has no concept of
    # recency at all, so without this a query for "recent papers on X" can easily
    # surface an older, heavily-matching paper over a newer, thinner-matching one.
    "recency": "submittedDate",
}


def _category_clause(categories: Sequence[str] | None) -> str:
    """arXiv search syntax for "in any of these categories", e.g.
    `(cat:cs.LG OR cat:cs.CL)`. Empty string when unscoped."""
    if not categories:
        return ""
    return "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"


def _scoped(query: str, categories: Sequence[str] | None) -> str:
    """Appends a category filter, restructuring the query so arXiv actually applies it.

    arXiv's parser does NOT treat an unquoted multi-word field value as a unit. Given
    `all:galaxy cluster mass function AND (cat:cs.LG OR ...)` it binds `all:` to the
    first word and then loses the trailing boolean clause entirely - the filter is
    silently dropped and the caller gets unfiltered results that look plausible.
    Measured: that query returned five papers whose ONLY category was astro-ph, one of
    them from 1994, every one of which the filter should have excluded.

    Wrapping each word in its own `all:` term and parenthesizing the group makes the
    structure explicit, and the filter then holds - the same query returns only papers
    genuinely cross-listed into cs.LG / cs.CV / stat.ML.

    ANDing every word is stricter than the unscoped form, which is why it is used only
    when a filter was actually asked for: a long unscoped query that requires every
    word to appear would return nothing.
    """
    clause = _category_clause(categories)
    if not clause:
        return query

    field, _, value = query.partition(":")
    words = value.split()
    # A quoted phrase (`ti:"attention is all you need"`) is already a single parser
    # token and must not be split apart - doing so would destroy the phrase match that
    # is the entire point of the title query.
    if len(words) <= 1 or value.startswith('"'):
        return f"{query} AND {clause}"

    grouped = " AND ".join(f"{field}:{word}" for word in words)
    return f"({grouped}) AND {clause}"


def _get_with_backoff(params: dict) -> requests.Response:
    """GET against the arXiv API, retrying rate limits with exponential backoff.

    Only 429 is retried - a malformed query or a server error will fail the same way
    on every attempt, so retrying those just burns the rate-limit budget.
    """
    delay = RETRY_BASE_DELAY
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        _throttle()
        resp = requests.get(ARXIV_API_URL, params=params, headers=REQUEST_HEADERS, timeout=30)

        if resp.status_code != 429:
            resp.raise_for_status()
            return resp

        if attempt == RETRY_ATTEMPTS:
            resp.raise_for_status()

        logger.info("arXiv rate limited (attempt %d/%d) - retrying in %.0fs", attempt, RETRY_ATTEMPTS, delay)
        time.sleep(delay)
        delay *= 2

    raise RuntimeError("unreachable: loop always returns or raises")


def _run_query(search_query: str, max_results: int, sort: str, start: int = 0) -> list[ArxivPaper]:
    resp = _get_with_backoff(
        {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": SORT_MODES.get(sort, "relevance"),
            "sortOrder": "descending",
        }
    )
    feed = feedparser.parse(resp.content)

    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        pdf_url = next(
            (link.href for link in entry.links if getattr(link, "title", "") == "pdf"),
            entry.id.replace("/abs/", "/pdf/"),
        )
        primary = getattr(entry, "arxiv_primary_category", None) or {}
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=" ".join(entry.title.split()),
                abstract=" ".join(entry.summary.split()),
                authors=[a.name for a in entry.authors],
                published=entry.published,
                pdf_url=pdf_url,
                primary_category=primary.get("term"),
            )
        )
    return papers


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def search(
    query: str,
    max_results: int = 10,
    sort: str = "relevance",
    categories: Sequence[str] | None = CORPUS_CATEGORIES,
) -> list[ArxivPaper]:
    """Searches arXiv, scoped to `categories` (the configured deep-learning set by
    default; pass None to search all of arXiv)."""
    # arXiv's `all:` search on an unquoted multi-word query effectively ORs across
    # every word - including stopwords like "is"/"all"/"need" - so a famous paper can
    # rank arbitrarily low against thinner matches on those common words alone. E.g.
    # `all:attention is all you need` doesn't even put "Attention Is All You Need"
    # (1706.03762) in the top 5; `ti:"attention is all you need"` does, as #1.
    #
    # A phrase-quoted title search only helps when the query IS (close to) an actual
    # title though - it's useless/wrong for a loosely-phrased topical query like "how
    # does retrieval improve factual accuracy". So run both and merge, title-phrase
    # hits first: this adds precision for "find this specific paper" lookups without
    # taking anything away from open-ended topic search.
    title_hits: list[ArxivPaper] = []
    if sort == "relevance" and len(query.split()) > 1:
        title_hits = _run_query(_scoped(f'ti:"{query}"', categories), max_results, sort)
        for p in title_hits:
            p.is_title_match = True

    broad_hits = _run_query(_scoped(f"all:{query}", categories), max_results, sort)

    seen_ids = {p.arxiv_id for p in title_hits}
    merged = title_hits + [p for p in broad_hits if p.arxiv_id not in seen_ids]
    return merged[:max_results]


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def search_by_author(
    author: str,
    max_results: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[ArxivPaper]:
    """Papers by an author, newest first, optionally restricted to a year range.

    A separate entry point rather than a flag on `search`, because the two have
    genuinely different shapes. `search` runs a title pass and a broad pass and merges
    them; an author query wants neither - the name is an exact field match, and
    ranking it by text relevance would score a prolific author's papers against each
    other on how often their own name appears.

    Sorted by recency, since "what has this person published" almost always means
    "lately". arXiv has no relevance signal for an author query anyway.

    The name is phrase-quoted so `au:"Yann LeCun"` matches the author rather than
    every paper containing both words anywhere - unquoted, arXiv's parser splits it
    and the results become noise.
    """
    clauses = [f'au:"{author}"']
    if year_from or year_to:
        # arXiv wants both ends of the range; open-ended requests get a wide but
        # harmless bound rather than being rejected.
        start_year = year_from or 1991  # arXiv's first year
        end_year = year_to or 2100
        clauses.append(f"submittedDate:[{start_year}01010000 TO {end_year}12312359]")

    return _run_query(" AND ".join(clauses), max_results, sort="recency")


def list_cohort(category: str, year: int, start: int = 0, page_size: int = 100) -> list[ArxivPaper]:
    """One page of papers from a single (category, year) cohort, newest first.

    This is the crawl primitive, and it's deliberately shaped around the cohort
    rather than just "latest in category": field-normalized impact compares a paper
    against others of its own category AND year, so the corpus has to be built with
    that grouping in mind. Crawling newest-first across a category would pile up
    recent papers and leave earlier years too thin to compute a stable median for.

    Not cached - during a crawl each offset is read exactly once, so caching would
    only burn disk.
    """
    window = f"submittedDate:[{year}01010000 TO {year}12312359]"
    return _run_query(f"cat:{category} AND {window}", page_size, sort="recency", start=start)


def pdf_url_for(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def fetch_pdf(pdf_url: str, dest_dir: str) -> str:
    _throttle()
    resp = requests.get(pdf_url, headers=REQUEST_HEADERS, timeout=60)
    resp.raise_for_status()

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    filename = pdf_url.rstrip("/").split("/")[-1]
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    dest_path = Path(dest_dir) / filename
    dest_path.write_bytes(resp.content)
    return str(dest_path)
