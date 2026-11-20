"""Crossref client: resolving a misremembered title to the paper it names.

Crossref's `query.bibliographic` is built for exactly the job the other sources do
badly: take a citation-shaped string - a title from memory, possibly with authors or a
year attached - and return the registered work it refers to. It is tolerant of a wrong
word where full-text search is not. Measured 2026-11-20 with the panel's own paper
recalled as "error weighted drift ADOPTING ensemble framework": Crossref returned it
first, while OpenAlex's search did not return it in its top ten.

What Crossref usually lacks is an abstract, so it is used to identify the paper and
obtain its DOI, and OpenAlex is then asked for the full record by that DOI.
No key is required.
"""

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

import requests

from app.cache import cached
from app.config import API_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

API_BASE = "https://api.crossref.org"
USER_AGENT = "kenkyuu-research-agent/0.1"
MIN_REQUEST_INTERVAL = 0.5
RETRY_ATTEMPTS = 2
RETRY_BASE_DELAY = 1.0

_last_request_at = 0.0
_lock = Lock()


@dataclass
class CrossrefWork:
    doi: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None


def _throttle() -> None:
    global _last_request_at
    with _lock:
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _year(item: dict) -> int | None:
    for key in ("published", "published-print", "published-online", "issued"):
        parts = (item.get(key) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            return parts[0][0]
    return None


@cached(ttl_seconds=API_CACHE_TTL_SECONDS)
def match_bibliographic(citation: str, rows: int = 5) -> list[CrossrefWork]:
    """Registered works best matching a citation-like string, best first."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        _throttle()
        resp = requests.get(
            f"{API_BASE}/works",
            params={
                "query.bibliographic": citation,
                "rows": rows,
                "select": "DOI,title,author,published,published-print,published-online,"
                          "issued,container-title,is-referenced-by-count,type",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        if resp.status_code not in (429, 500, 502, 503, 504):
            resp.raise_for_status()
            break
        if attempt == RETRY_ATTEMPTS:
            resp.raise_for_status()
        time.sleep(delay)
        delay *= 2

    works = []
    for item in resp.json().get("message", {}).get("items") or []:
        title = (item.get("title") or [""])[0]
        # Crossref also registers figures, tables and dataset components under their
        # own DOIs ("Figure 3: Confusion matrices for ..."). They match citation strings
        # well and are never what anyone is looking for.
        if not title or item.get("type") in ("component", "dataset", "peer-review"):
            continue
        works.append(CrossrefWork(
            doi=item["DOI"],
            title=title,
            authors=[
                " ".join(p for p in (a.get("given"), a.get("family")) if p)
                for a in item.get("author") or []
                if a.get("family")
            ],
            year=_year(item),
            venue=(item.get("container-title") or [None])[0],
            citation_count=item.get("is-referenced-by-count"),
        ))
    return works
