"""Semantic Scholar Graph API client — search, metadata, citation graph.

Unauthenticated access is capped at 100 requests / 5 minutes; the wait is
sized to stay under that (3s/request) whether or not an API key is set.
Set S2_API_KEY in .env to raise the limit.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from threading import Lock

import requests

from app.cache import cached
from app.config import API_CACHE_TTL_SECONDS, S2_API_KEY

logger = logging.getLogger(__name__)

API_BASE = "https://api.semanticscholar.org/graph/v1"
MIN_REQUEST_INTERVAL = 3.0
BATCH_LIMIT = 500  # documented server-side cap on ids per /paper/batch request

# Unauthenticated requests draw on a pool shared with every other anonymous caller,
# so a 429 means "someone else is busy right now", not "you did something wrong" -
# it clears on its own. Backing off and retrying is the difference between an
# overnight crawl finishing and it quietly dropping batches.
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 5.0

# Interactive requests get a much shorter budget than the crawl. A background crawl
# should wait out a rate limit, because finishing slowly beats dropping batches. A
# user-facing search must not: the patient budget spends 75 seconds (5+10+20+40) inside
# a single request, which on a persistent 429 is 75 seconds of silence and then a
# failure anyway. Failing fast and reporting reduced coverage is strictly better than
# stalling the whole answer.
#
# Measured 2026-10-03, unauthenticated: every GET endpoint - /paper/search,
# /paper/search/match, /paper/search/bulk and even /paper/{id} - returns 429, while
# POST /paper/batch returns 200. Anonymous access is effectively batch-only now, so for
# live search the retry is a formality and the real fix is an API key.
INTERACTIVE_RETRY_ATTEMPTS = 2
INTERACTIVE_RETRY_DELAY = 1.5

_last_request_at = 0.0
_lock = Lock()


def _throttle() -> None:
    global _last_request_at
    with _lock:
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _headers() -> dict:
    return {"x-api-key": S2_API_KEY} if S2_API_KEY else {}


@dataclass
class S2Paper:
    paper_id: str
    title: str
    abstract: str | None
    year: int | None
    citation_count: int
    authors: list[str] = field(default_factory=list)
    external_ids: dict = field(default_factory=dict)


def _parse_paper(data: dict) -> S2Paper:
    return S2Paper(
        paper_id=data["paperId"],
        title=data.get("title") or "",
        abstract=data.get("abstract"),
        year=data.get("year"),
        citation_count=data.get("citationCount") or 0,
        authors=[a.get("name", "") for a in data.get("authors") or []],
        external_ids=data.get("externalIds") or {},
    )


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def search(query: str, limit: int = 10) -> list[S2Paper]:
    resp = _get_with_backoff(
        f"{API_BASE}/paper/search",
        params={
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,citationCount,authors,externalIds",
        },
    )
    return [_parse_paper(p) for p in resp.json().get("data", [])]


@cached(ttl_seconds=API_CACHE_TTL_SECONDS * 24)  # paper metadata barely changes
def get_paper(paper_id: str) -> S2Paper:
    resp = _get_with_backoff(
        f"{API_BASE}/paper/{paper_id}",
        params={"fields": "title,abstract,year,citationCount,authors,externalIds"},
    )
    return _parse_paper(resp.json())


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def get_citations(paper_id: str, limit: int = 50) -> list[S2Paper]:
    """Papers that cite this one."""
    resp = _get_with_backoff(
        f"{API_BASE}/paper/{paper_id}/citations",
        params={"limit": limit, "fields": "title,abstract,year,citationCount,authors,externalIds"},
    )
    return [_parse_paper(c["citingPaper"]) for c in resp.json().get("data", []) if c.get("citingPaper")]


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def get_references(paper_id: str, limit: int = 50) -> list[S2Paper]:
    """Papers this one cites."""
    resp = _get_with_backoff(
        f"{API_BASE}/paper/{paper_id}/references",
        params={"limit": limit, "fields": "title,abstract,year,citationCount,authors,externalIds"},
    )
    return [_parse_paper(r["citedPaper"]) for r in resp.json().get("data", []) if r.get("citedPaper")]


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def match_title(title: str) -> S2Paper | None:
    """The single paper whose title best matches `title`, or None.

    Uses /paper/search/match, which exists for exactly this question - "which paper is
    this title?" - rather than /paper/search, which ranks by relevance and will happily
    return ten papers on the subject when the user named one specific work.

    This is the right endpoint for finding a professor's own paper, including one
    published at IEEE, Springer, Elsevier or ACM that has no arXiv preprint. Semantic
    Scholar is the only source in this system that covers those at all.

    Returns None rather than raising when S2 reports no match (it answers 404 with an
    explanatory body for "title not found"), because "no paper has this title" is a
    normal, useful answer and not an error.

    NOTE: unauthenticated callers currently receive 429 on every GET endpoint,
    including this one, so in practice this requires S2_API_KEY to be set. That is the
    documented fix, not a workaround - the key is free.
    """
    if not title.strip():
        return None
    try:
        resp = _get_with_backoff(
            f"{API_BASE}/paper/search/match",
            params={
                "query": title,
                "fields": "title,abstract,year,citationCount,authors,externalIds,venue",
            },
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.info("S2 has no title match for %r", title[:80])
            return None
        raise

    data = resp.json().get("data") or []
    return _parse_paper(data[0]) if data else None


# --- Bulk metrics ----------------------------------------------------------


@dataclass
class S2Metrics:
    """The scientometric slice of a paper - what the corpus crawl needs, without
    dragging abstracts and author lists through a 500-paper response."""

    s2_paper_id: str
    citation_count: int | None
    influential_citation_count: int | None
    reference_count: int | None
    venue: str | None
    year: int | None


_VERSION_SUFFIX = re.compile(r"v\d+$")

_BATCH_FIELDS = "externalIds,venue,year,citationCount,influentialCitationCount,referenceCount"
_REFERENCE_FIELDS = "references.paperId"


def get_references_batch(paper_ids: list[str]) -> dict[str, list[str]]:
    """Reference lists (as S2 paperIds) for up to BATCH_LIMIT papers in ONE request.

    This is what makes a corpus-wide citation graph feasible at all. The per-paper
    /references endpoint under a 3s rate limit would take about seven hours for this
    corpus; batched, it is a couple of minutes.

    Returns {input_id: [cited_s2_id, ...]}. Papers S2 doesn't know, and references
    with no paperId (unparsed bibliography entries), are simply absent.
    """
    if not paper_ids:
        return {}
    if len(paper_ids) > BATCH_LIMIT:
        raise ValueError(f"batch of {len(paper_ids)} exceeds S2's limit of {BATCH_LIMIT}")

    try:
        resp = _post_with_backoff(
            f"{API_BASE}/paper/batch",
            params={"fields": _REFERENCE_FIELDS},
            body={"ids": [_batch_id(i) for i in paper_ids]},
        )
    except requests.HTTPError as exc:
        if _is_no_valid_ids(exc):
            logger.info("S2 has none of these %d papers indexed yet", len(paper_ids))
            return {}
        raise

    out: dict[str, list[str]] = {}
    for original_id, record in zip(paper_ids, resp.json()):
        if not record:
            continue
        refs = record.get("references") or []
        cited = [r["paperId"] for r in refs if r and r.get("paperId")]
        if cited:
            out[original_id] = cited
    return out


def _is_no_valid_ids(exc: requests.HTTPError) -> bool:
    """True for S2's "none of these ids are known to me" 400, as opposed to a genuinely
    malformed request that happens to share the status code."""
    resp = exc.response
    if resp is None or resp.status_code != 400:
        return False
    try:
        return "no valid paper ids" in resp.json().get("error", "").lower()
    except ValueError:  # body wasn't JSON, so this is some other 400
        return False


def _get_with_backoff(
    url: str,
    *,
    params: dict,
    timeout: int = 30,
    attempts: int = INTERACTIVE_RETRY_ATTEMPTS,
    base_delay: float = INTERACTIVE_RETRY_DELAY,
) -> requests.Response:
    """GET, retrying on rate limits with exponential backoff.

    The live endpoints need this as much as the batch ones, and for a sharper reason:
    unauthenticated requests draw on a pool shared with every other anonymous caller,
    so a 429 means "someone else is busy", not "you did something wrong". Without a
    retry, a single 429 makes Semantic Scholar unreachable for that request - and S2
    is the ONLY route to papers that are not on arXiv (IEEE, Springer, Elsevier,
    ACM). A professor searching for their own journal paper would be told it does not
    exist, because an unrelated stranger was mid-crawl.

    Only 429 is retried. A 404 means the paper genuinely is not indexed, and retrying
    it just burns the rate-limit budget on a request that will keep failing the same
    way.
    """
    delay = base_delay
    for attempt in range(1, attempts + 1):
        _throttle()
        resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)

        if resp.status_code != 429:
            resp.raise_for_status()
            return resp

        if attempt == attempts:
            resp.raise_for_status()

        logger.info("S2 rate limited (attempt %d/%d) - retrying in %.1fs", attempt, attempts, delay)
        time.sleep(delay)
        delay *= 2

    raise RuntimeError("unreachable: loop always returns or raises")


def _post_with_backoff(url: str, *, params: dict, body: dict) -> requests.Response:
    """POST, retrying on rate limits with exponential backoff.

    Only 429 is retried. Other failures (bad request, server error) are raised
    immediately - retrying those just wastes the rate-limit budget on a request
    that will keep failing the same way.
    """
    delay = RETRY_BASE_DELAY
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        _throttle()
        resp = requests.post(url, params=params, json=body, headers=_headers(), timeout=60)

        if resp.status_code != 429:
            resp.raise_for_status()
            return resp

        if attempt == RETRY_ATTEMPTS:
            resp.raise_for_status()

        logger.info("S2 rate limited (attempt %d/%d) - retrying in %.0fs", attempt, RETRY_ATTEMPTS, delay)
        time.sleep(delay)
        delay *= 2

    raise RuntimeError("unreachable: loop always returns or raises")


def bare_arxiv_id(arxiv_id: str) -> str:
    """`2401.00611v3` -> `2401.00611`. S2 keys off the version-less id; passing the
    versioned form silently misses."""
    return _VERSION_SUFFIX.sub("", arxiv_id)


# Both arXiv id styles: modern (2401.00611) and pre-2007 (cond-mat/0309395).
_ARXIV_ID_SHAPE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})(v\d+)?$")


def _batch_id(paper_id: str) -> str:
    """Formats one id for the batch endpoint, which accepts several namespaces.

    An arXiv id needs the `ARXIV:` prefix; a Semantic Scholar paperId (40-char hex)
    is already native and must be passed through untouched. Prefixing an S2 hash
    with `ARXIV:` doesn't error - it just silently returns no match, which is how
    this went unnoticed until a corpus count came up short.
    """
    if _ARXIV_ID_SHAPE.match(paper_id):
        return f"ARXIV:{bare_arxiv_id(paper_id)}"
    return paper_id


def get_metrics_batch(arxiv_ids: list[str]) -> dict[str, S2Metrics]:
    """Metrics for up to BATCH_LIMIT arXiv papers in ONE request, keyed by the
    original (possibly versioned) id you passed in.

    Papers S2 doesn't know are simply absent from the returned mapping rather than
    raising - a corpus crawl shouldn't die because one preprint isn't indexed yet.

    Batching matters a lot here: one-at-a-time under the 3s rate limit is roughly
    four hours for a 5k-paper corpus, versus a couple of minutes this way.
    """
    if not arxiv_ids:
        return {}
    if len(arxiv_ids) > BATCH_LIMIT:
        raise ValueError(f"batch of {len(arxiv_ids)} exceeds S2's limit of {BATCH_LIMIT}")

    try:
        resp = _post_with_backoff(
            f"{API_BASE}/paper/batch",
            params={"fields": _BATCH_FIELDS},
            body={"ids": [_batch_id(i) for i in arxiv_ids]},
        )
    except requests.HTTPError as exc:
        # S2 answers 400 "No valid paper ids given" when it recognizes NONE of the
        # ids in a batch - which is a normal outcome, not a malformed request: very
        # recent preprints take days or weeks to appear in their index. (A batch with
        # even one known id returns a list with nulls for the rest, so this response
        # unambiguously means "none of these are indexed yet".) Treating it as an
        # empty result lets those papers stay queued and get picked up by a later run
        # once S2 catches up, instead of surfacing as an alarming crawl failure.
        if _is_no_valid_ids(exc):
            logger.info("S2 has none of these %d papers indexed yet", len(arxiv_ids))
            return {}
        raise

    # The response is a list positionally aligned with the ids we sent, with a null
    # entry wherever S2 has no record - so zip against the input rather than trying
    # to match on the returned externalIds (which are missing on exactly those nulls).
    metrics: dict[str, S2Metrics] = {}
    for original_id, record in zip(arxiv_ids, resp.json()):
        if not record:
            continue
        metrics[original_id] = S2Metrics(
            s2_paper_id=record["paperId"],
            citation_count=record.get("citationCount"),
            influential_citation_count=record.get("influentialCitationCount"),
            reference_count=record.get("referenceCount"),
            venue=record.get("venue") or None,
            year=record.get("year"),
        )
    return metrics
