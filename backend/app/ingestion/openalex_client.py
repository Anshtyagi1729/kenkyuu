"""OpenAlex client: keyless scholarly search across journals, conferences and preprints.

Why this source exists. Live discovery previously rested on two APIs that each fail in
practice: arXiv allows roughly one request per three seconds and has refused all traffic
from this machine for days at a time, and Semantic Scholar now answers HTTP 429 to every
unauthenticated search. When both were down the agent silently fell back to its local
corpus, so a professor asking for a journal paper got the nearest preprint instead.

OpenAlex is the open successor to Microsoft Academic Graph: 250M+ works with abstracts,
authors, venues, citation counts and open-access links, under a generous published
limit (10 requests/second, 100k/day) with no key. It covers IEEE, Springer, Elsevier and
ACM - which arXiv does not - and indexes arXiv itself, so it is the one source that can
answer for both halves of the literature. Measured 2026-11-20 it returned the panel's
Springer paper as its first result for that paper's title.
"""

import datetime
import logging
import re
import time
from dataclasses import dataclass, field
from threading import Lock

import requests

from app.cache import cached
from app.config import API_CACHE_TTL_SECONDS, OPENALEX_MAILTO

logger = logging.getLogger(__name__)

API_BASE = "https://api.openalex.org"
USER_AGENT = "kenkyuu-research-agent/0.1"

# The documented ceiling is 10 requests/second; staying well under it costs nothing at
# interactive volumes and keeps a burst of parallel agent calls from tripping it.
MIN_REQUEST_INTERVAL = 0.2

# Interactive budget: fail fast and let the other sources answer, rather than holding a
# user's request open (the lesson learned from the S2 and arXiv retry loops).
RETRY_ATTEMPTS = 2
RETRY_BASE_DELAY = 1.0

# Only the fields we read. A full work record runs to tens of kilobytes, mostly
# per-author institution lists and reference ids we never use.
_SELECT = ",".join([
    "id", "display_name", "doi", "publication_year", "publication_date",
    "authorships", "primary_location", "best_oa_location", "locations",
    "cited_by_count", "abstract_inverted_index",
])

# Scholarly documents only. OpenAlex also indexes datasets, peer-review reports, errata
# and supplementary files as "works"; with a date sort they flood the results. Measured:
# a recency search for "large language model agents" returned a dataset deposit dated
# 2027. book-chapter stays in because many conference papers (Springer LNCS and similar
# proceedings) are registered as chapters.
_DOCUMENT_TYPES = "type:article|preprint|review|book-chapter"

# How far back "recent" reaches for a recency search.
RECENT_YEARS = 2

_ARXIV_ABS = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", re.IGNORECASE)

_last_request_at = 0.0
_lock = Lock()


@dataclass
class OpenAlexWork:
    openalex_id: str
    title: str
    abstract: str | None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    publication_date: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    # A direct open-access PDF, when one exists. This is what lets the agent read the
    # full text of a journal paper, not just arXiv preprints.
    pdf_url: str | None = None


@dataclass
class AuthorMatch:
    openalex_id: str
    name: str
    works_count: int
    institution: str | None


def _throttle() -> None:
    global _last_request_at
    with _lock:
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _get(path: str, params: dict) -> dict:
    """GET with a short retry on rate limits and server errors.

    5xx is retried as well as 429 here, unlike the S2 client: OpenAlex occasionally
    answers a heavy search with a transient 503 that succeeds a second later.
    """
    if OPENALEX_MAILTO:
        params = {**params, "mailto": OPENALEX_MAILTO}
    delay = RETRY_BASE_DELAY
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        _throttle()
        resp = requests.get(
            f"{API_BASE}{path}", params=params, headers={"User-Agent": USER_AGENT}, timeout=20
        )
        if resp.status_code not in (429, 500, 502, 503, 504):
            resp.raise_for_status()
            return resp.json()
        if attempt == RETRY_ATTEMPTS:
            resp.raise_for_status()
        logger.info("OpenAlex %d (attempt %d/%d) - retrying in %.1fs", resp.status_code, attempt, RETRY_ATTEMPTS, delay)
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable: loop always returns or raises")


def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Rebuilds abstract text from OpenAlex's inverted index ({word: [positions]}).

    OpenAlex does not ship abstracts as plain text - for licensing reasons it publishes
    each word with the positions it occurs at - so the text has to be reassembled by
    placing every word at each of its positions.
    """
    if not inverted_index:
        return None
    positioned = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    if not positioned:
        return None
    return " ".join(word for _, word in sorted(positioned))


def _base_filters() -> list[str]:
    """Document types only, and nothing dated after today - some records carry
    placeholder publication dates years in the future, which a date sort puts first."""
    return [_DOCUMENT_TYPES, f"to_publication_date:{datetime.date.today().isoformat()}"]


def _short_id(url: str | None) -> str:
    return (url or "").rsplit("/", 1)[-1]


def _arxiv_id(work: dict) -> str | None:
    """arXiv id from any location of the work. OpenAlex lists the arXiv copy as one of
    several locations rather than as an identifier, so it has to be read off a URL."""
    for loc in work.get("locations") or []:
        for key in ("landing_page_url", "pdf_url"):
            m = _ARXIV_ABS.search((loc or {}).get(key) or "")
            if m:
                return m.group(1)
    return None


def _parse(work: dict) -> OpenAlexWork:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    best_oa = work.get("best_oa_location") or {}
    doi = work.get("doi")
    return OpenAlexWork(
        openalex_id=_short_id(work.get("id")),
        title=work.get("display_name") or "",
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        authors=[
            (a.get("author") or {}).get("display_name", "")
            for a in work.get("authorships") or []
            if (a.get("author") or {}).get("display_name")
        ],
        year=work.get("publication_year"),
        publication_date=work.get("publication_date"),
        venue=source.get("display_name"),
        citation_count=work.get("cited_by_count"),
        doi=doi.removeprefix("https://doi.org/") if doi else None,
        arxiv_id=_arxiv_id(work),
        pdf_url=best_oa.get("pdf_url"),
    )


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def search(query: str, limit: int = 10, sort: str = "relevance") -> list[OpenAlexWork]:
    """Full-text search over titles, abstracts and fulltext, best match first.

    `sort="recency"` orders by publication date instead - relevance has no notion of
    time and will return a 2017 paper for "what's new in X".
    """
    filters = _base_filters()
    if sort == "recency":
        # Recent AND relevant: restrict to the last few years, keep relevance order.
        # Sorting the whole result set by date instead returns the newest thing that
        # matches any of the words, which is rarely what "what's new in X" means.
        filters.append(f"from_publication_date:{datetime.date.today().year - RECENT_YEARS}-01-01")
    params = {"search": query, "per-page": limit, "select": _SELECT, "filter": ",".join(filters)}
    data = _get("/works", params)
    return [_parse(w) for w in data.get("results") or [] if w.get("display_name")]


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def find_author(name: str) -> AuthorMatch | None:
    """The most prolific author matching `name`.

    Picking by works count is a deliberate heuristic for name collisions: "Seba Susan"
    may match several profiles, including split duplicates of the same person, and the
    largest is usually the real one. The match is returned with its institution so the
    agent can say WHO it looked up rather than silently guessing between namesakes.
    """
    data = _get("/authors", {"search": name, "per-page": 5})
    candidates = data.get("results") or []
    if not candidates:
        return None
    best = max(candidates, key=lambda a: a.get("works_count") or 0)
    institutions = best.get("last_known_institutions") or []
    return AuthorMatch(
        openalex_id=_short_id(best.get("id")),
        name=best.get("display_name") or name,
        works_count=best.get("works_count") or 0,
        institution=(institutions[0] or {}).get("display_name") if institutions else None,
    )


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def works_by_author(
    author_id: str,
    limit: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[OpenAlexWork]:
    """An author's works, newest first, optionally within a year range.

    This is what arXiv author search cannot do: it returns a researcher's journal and
    conference papers as well as their preprints. For many professors - including the
    panel's own author - most of their output never appears on arXiv at all.
    """
    filters = [f"author.id:{author_id}", *_base_filters()]
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    data = _get("/works", {
        "filter": ",".join(filters),
        "sort": "publication_date:desc",
        "per-page": limit,
        "select": _SELECT,
    })
    return [_parse(w) for w in data.get("results") or [] if w.get("display_name")]


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def get_by_doi(doi: str) -> OpenAlexWork | None:
    """The full OpenAlex record for a DOI, or None if OpenAlex does not hold it.

    The second half of title resolution: Crossref identifies the paper and gives its DOI
    but rarely an abstract; OpenAlex, asked by DOI, supplies the abstract, author list,
    citation count and any open-access PDF.
    """
    try:
        data = _get(f"/works/doi:{doi}", {"select": _SELECT})
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    return _parse(data) if data.get("display_name") else None
