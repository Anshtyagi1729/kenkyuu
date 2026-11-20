"""Two-tier ingestion.

Tier 1 (search_and_record) is cheap: metadata + a single abstract embedding
per result, no PDF download. This is what runs on every search query.

Tier 2 (ingest_full_text) is expensive: PDF download, parsing, chunking, and
embedding the whole paper. It's lazy — only called for a paper the user or
agent actually needs deep grounding on (opened, saved, being compared) — and
cached, so re-requesting an already-ingested paper is a no-op.
"""

import json
import logging
import tempfile
from pathlib import Path

import requests

from app.ingestion import arxiv_client, embed, openalex_client, pdf_parser, semantic_scholar_client
from app.storage import db, vector_store

logger = logging.getLogger(__name__)

ABSTRACT_CHUNK_INDEX = -1


def abstract_chunk_text(title: str, abstract: str) -> str:
    """The text indexed for a paper's abstract chunk: title first, then abstract.

    Including the title is load-bearing, not cosmetic. Indexing the abstract alone
    means a paper's own title is absent from the index, so a query naming the paper
    cannot match it - "Attention Is All You Need" appears nowhere in 1706.03762's
    abstract, while a 2025 paper *about* that title quotes it verbatim and wins. That
    is a guaranteed loss on exactly the find-this-specific-paper queries the title
    would settle instantly.

    Helps both retrievers: BM25 gets the exact tokens that identify the paper, and
    the embedding gets the author's own one-line summary of the contribution. Pairing
    title with abstract is standard practice in scholarly retrieval (SPECTER does it).
    """
    return f"{title}\n\n{abstract}"


def _index_abstract(
    paper_id: str,
    title: str,
    abstract: str | None,
    authors: list[str] | None = None,
) -> None:
    """Stores and embeds one paper's identifying chunk, so it is searchable immediately.

    Shared by the arXiv and Semantic Scholar record paths. Extracted rather than
    duplicated because the two differ only in where the metadata came from, and a
    second copy of the embed-and-upsert logic is how the two sources would drift into
    indexing subtly different text.

    **Indexes a title-only chunk when there is no abstract**, rather than skipping the
    paper. Many non-arXiv records have no abstract on file - the panel's own Springer
    paper is one - and the earlier behaviour of indexing nothing meant the paper was
    stored, counted, and permanently unretrievable. A paper that retrieval can never
    return is a paper the system does not have.

    Indexing the title alone is not a consolation prize here. This project's largest
    single measured gain (+0.0847 nDCG) came from putting titles into the indexed text,
    because the title is what carries a paper's IDENTITY - and a query naming a paper
    is exactly the case where an abstractless record still needs to be findable.

    The absence is written into the chunk text so synthesis can see it. Without that
    marker the model receives a title and may write about the paper's contents as
    though it had read them.
    """
    chunk_id = f"{paper_id}::abstract"
    if db.get_chunk_text(chunk_id) is not None:
        return

    if abstract:
        text = abstract_chunk_text(title, abstract)
    else:
        byline = f"Authors: {', '.join(authors)}\n" if authors else ""
        text = (
            f"{title}\n{byline}\n"
            "(No abstract is available for this paper on record - only its title, "
            "authors and citation metrics are known.)"
        )
    with db.connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, paper_id, chunk_index, text, start_page, end_page) "
            "VALUES (?, ?, ?, ?, NULL, NULL)",
            (chunk_id, paper_id, ABSTRACT_CHUNK_INDEX, text),
        )
    embedding = embed.embed_texts([text])[0]
    vector_store.add_chunks([chunk_id], [text], [embedding], paper_id, pages=[(None, None)])


def record_search_result(paper: arxiv_client.ArxivPaper) -> None:
    db.upsert_paper(
        paper_id=paper.arxiv_id,
        source="arxiv",
        title=paper.title,
        abstract=paper.abstract,
        authors=paper.authors,
        published=paper.published,
        external_ids={"ArXiv": paper.arxiv_id},
    )
    # Abstract embedding lets search/retrieval find this paper immediately,
    # even before (or if never) its full text gets ingested.
    _index_abstract(paper.arxiv_id, paper.title, paper.abstract, paper.authors)


def record_external(
    *,
    source: str,
    title: str,
    abstract: str | None,
    authors: list[str],
    year: int | None,
    publication_date: str | None = None,
    arxiv_id: str | None = None,
    doi: str | None = None,
    fallback_id: str | None = None,
    citation_count: int | None = None,
    venue: str | None = None,
    s2_paper_id: str | None = None,
    extra_ids: dict | None = None,
    index: bool = True,
) -> str | None:
    """Records a paper found by any live source. Returns the corpus id it lives under.

    One path for Semantic Scholar, OpenAlex and Crossref, because the rules that matter
    are the same for all three and each is easy to get wrong:

    1. **Resolve to an existing row before writing.** By arXiv id (version-insensitive),
       then by DOI, so a paper found through two sources is one paper. The id falls
       back to the source's own identifier only for a paper the corpus has never seen.
       Writing under the caller's id instead would create a second abstract chunk for a
       paper already indexed - `1706.03762` beside `1706.03762v7` - and the duplicate
       would compete with the original in every ranking.
    2. **Fill gaps, never overwrite.** A live source must not erase an abstract, rewrite
       a real submission date, or replace crawl metrics the ranker reads as features.
    3. **Index something even without an abstract**, so the paper is retrievable at all.

    Ids must be slash-free because they appear in URL paths; DOIs therefore cannot be
    ids, and an OpenAlex work id ("W4391...") is used instead.
    """
    existing = (db.get_paper(arxiv_id) if arxiv_id else None) or (
        db.get_paper(db.find_paper_by_doi(doi)) if doi and db.find_paper_by_doi(doi) else None
    )
    paper_id = existing["paper_id"] if existing else (arxiv_id or fallback_id)
    if not paper_id or "/" in paper_id:
        return None

    ids = dict(extra_ids or {})
    if arxiv_id:
        ids["ArXiv"] = arxiv_id
    if doi:
        ids["DOI"] = doi

    db.merge_external_paper(
        paper_id=paper_id,
        source=source,
        title=title,
        abstract=abstract,
        authors=authors,
        # A bare year is stored as just the year. It used to be padded to January 1st,
        # and the paper page then displayed "2026-01-01" for a paper whose day and
        # month nobody knows. "2026" still sorts correctly against full ISO dates.
        published=publication_date or (str(year) if year else None),
        external_ids=ids,
    )
    db.insert_metrics_if_absent(paper_id, s2_paper_id, citation_count, venue, year)
    # Graph neighbours pass index=False: a citation graph surfaces up to forty papers at
    # once, and embedding all of them would make the page wait on the embedding model
    # for papers nobody has asked about yet. They are stored so their node opens, and are
    # embedded later by the corpus "index" stage if they turn out to matter.
    if index:
        row = db.get_paper(paper_id)
        _index_abstract(paper_id, row["title"], row["abstract"], json.loads(row["authors"] or "[]"))
    return paper_id


def record_s2_paper(paper: semantic_scholar_client.S2Paper) -> str | None:
    """Records a Semantic Scholar paper. See record_external for the rules applied."""
    ids = paper.external_ids or {}
    return record_external(
        source="semantic_scholar",
        title=paper.title,
        abstract=paper.abstract,
        authors=paper.authors,
        year=paper.year,
        arxiv_id=ids.get("ArXiv"),
        doi=ids.get("DOI"),
        fallback_id=paper.paper_id,
        citation_count=paper.citation_count,
        s2_paper_id=paper.paper_id,
        extra_ids={k: v for k, v in ids.items() if k not in ("ArXiv", "DOI")},
    )


def record_openalex_work(work: openalex_client.OpenAlexWork, index: bool = True) -> str | None:
    """Records an OpenAlex work, keeping its open-access PDF link for full-text reading."""
    extra = {"OpenAlex": work.openalex_id}
    if work.pdf_url:
        extra["OpenAccessPdf"] = work.pdf_url
    return record_external(
        source="openalex",
        title=work.title,
        abstract=work.abstract,
        authors=work.authors,
        year=work.year,
        publication_date=work.publication_date,
        arxiv_id=work.arxiv_id,
        doi=work.doi,
        fallback_id=work.openalex_id,
        citation_count=work.citation_count,
        venue=work.venue,
        extra_ids=extra,
        index=index,
    )


def index_abstracts(
    paper_ids: list[str],
    batch_size: int = 64,
    collection: str = vector_store.COLLECTION_NAME,
    model_name: str | None = None,
) -> int:
    """Embeds abstracts for already-stored papers, in batches. Returns the count indexed.

    Bulk counterpart to record_search_result's one-at-a-time path. Corpus building
    stores metadata without embedding anything, so without this the vector index holds
    only the handful of papers that arrived through live searches - and retrieval can
    physically never return the rest, no matter how good the ranking gets.

    Batching is the whole point: sentence-transformers amortizes model overhead across
    a batch, so embedding 64 abstracts at once is far faster than 64 separate calls.
    """
    indexed = 0
    for start in range(0, len(paper_ids), batch_size):
        batch_ids = paper_ids[start : start + batch_size]

        # One query for the batch rather than one per paper - at corpus scale the
        # per-row round trips cost more than the embedding itself.
        placeholders = ",".join("?" * len(batch_ids))
        with db.connect() as conn:
            rows = conn.execute(
                f"SELECT paper_id, title, abstract FROM papers "
                f"WHERE paper_id IN ({placeholders}) AND abstract IS NOT NULL AND abstract != ''",
                batch_ids,
            ).fetchall()

        if not rows:
            continue

        ids = [row["paper_id"] for row in rows]
        texts = [abstract_chunk_text(row["title"], row["abstract"]) for row in rows]
        chunk_ids = [f"{pid}::abstract" for pid in ids]

        with db.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks (chunk_id, paper_id, chunk_index, text, start_page, end_page) "
                "VALUES (?, ?, ?, ?, NULL, NULL)",
                [(cid, pid, ABSTRACT_CHUNK_INDEX, text) for cid, pid, text in zip(chunk_ids, ids, texts)],
            )

        embeddings = embed.embed_texts(texts, model_name=model_name)
        vector_store.add_abstract_chunks(chunk_ids, texts, embeddings, ids, collection=collection)
        indexed += len(ids)

    return indexed


def unindexed_paper_ids() -> list[str]:
    """Papers stored but with no abstract embedding yet - the indexing work queue."""
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT p.paper_id
            FROM papers p
            LEFT JOIN chunks c ON c.chunk_id = p.paper_id || '::abstract'
            WHERE c.chunk_id IS NULL AND p.abstract IS NOT NULL AND p.abstract != ''
            ORDER BY p.paper_id
            """
        )
        return [row["paper_id"] for row in rows]


def search_and_record(query: str, max_results: int = 10, sort: str = "relevance") -> list[arxiv_client.ArxivPaper]:
    """Fast path: search arXiv, store metadata + abstract embeddings. No PDFs fetched.

    Deliberately UNSCOPED by category, unlike `arxiv_client.search`'s default. The
    deep-learning scope is a property of the curated corpus and the scientometric
    baselines computed over it - it is not a limit on what a professor is allowed to
    ask about. Inheriting the corpus scope here would mean a question about quantum
    machine learning or computational paleography silently returns only whatever
    happens to be cross-listed into cs.LG, with nothing in the answer explaining why.

    This was previously scoped in name only: the category filter was being dropped by
    arXiv's parser, so this path behaved as unscoped by accident. Fixing the filter
    without making this explicit would have turned a latent bug into a real one.
    """
    papers = arxiv_client.search(query, max_results=max_results, sort=sort, categories=None)
    for paper in papers:
        record_search_result(paper)
    return papers


def search_author_and_record(
    author: str,
    max_results: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[arxiv_client.ArxivPaper]:
    """Author search, storing metadata + abstract embeddings like `search_and_record`.

    Recording matters as much here as for topic search: the papers have to be in the
    index before any of them can be quoted in a grounded answer, and an author lookup
    is exactly the case where the professor then asks a follow-up about one of the
    results.
    """
    papers = arxiv_client.search_by_author(
        author, max_results=max_results, year_from=year_from, year_to=year_to
    )
    for paper in papers:
        record_search_result(paper)
    return papers


OPEN_ACCESS_USER_AGENT = "kenkyuu-research-agent/0.1"
MAX_PDF_BYTES = 40 * 1024 * 1024


def full_text_source(paper_row: dict) -> tuple[str, str] | None:
    """Where this paper's full text can be read, as ("arxiv" | "open_access", url).

    arXiv first, because its PDFs are reliable and consistently formatted. Otherwise
    the open-access PDF OpenAlex reported. None when neither exists - which for many
    journal papers is simply the truth, and must be reported as such rather than
    papered over by passing the abstract off as the paper.

    Previously every paper was fetched from arxiv.org/pdf/<paper_id>, which for a
    journal paper keyed by an OpenAlex or Semantic Scholar id is a guaranteed 404: no
    non-arXiv paper could ever be read in full, open access or not.
    """
    ids = json.loads(paper_row.get("external_ids") or "{}")
    arxiv_id = ids.get("ArXiv")
    if not arxiv_id and paper_row.get("source") == "arxiv":
        arxiv_id = paper_row["paper_id"]
    if arxiv_id:
        return "arxiv", arxiv_client.pdf_url_for(arxiv_id)
    if ids.get("OpenAccessPdf"):
        return "open_access", ids["OpenAccessPdf"]
    return None


def _fetch_open_access_pdf(url: str, dest_dir: str) -> str | None:
    """Downloads an open-access PDF, or returns None if what came back is not one.

    The check on the first bytes is the important part. "Open access" links frequently
    resolve to an HTML landing page, a cookie wall or a login form, which the PDF
    parser would otherwise attempt to read as a paper.
    """
    try:
        resp = requests.get(
            url, headers={"User-Agent": OPEN_ACCESS_USER_AGENT}, timeout=45, allow_redirects=True
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.info("open-access PDF fetch failed for %s: %s", url, str(exc)[:120])
        return None
    if not resp.content.startswith(b"%PDF") or len(resp.content) > MAX_PDF_BYTES:
        logger.info("open-access link did not return a usable PDF: %s", url)
        return None
    dest = Path(dest_dir) / "paper.pdf"
    dest.write_bytes(resp.content)
    return str(dest)


def ingest_full_text(paper_id: str) -> int:
    """Lazy, cached full-text ingestion. Returns the number of body chunks stored
    (0 if the paper isn't known yet, or if it was already ingested)."""
    if db.has_full_text(paper_id):
        return 0

    paper_row = db.get_paper(paper_id)
    if paper_row is None:
        return 0
    paper_id = paper_row["paper_id"]
    if db.has_full_text(paper_id):
        return 0

    source = full_text_source(paper_row)
    if source is None:
        logger.info("no openly available full text for %s", paper_id)
        return 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        kind, url = source
        if kind == "arxiv":
            pdf_path = arxiv_client.fetch_pdf(url, tmp_dir)
        else:
            pdf_path = _fetch_open_access_pdf(url, tmp_dir)
            if pdf_path is None:
                return 0
        chunks = pdf_parser.parse_and_chunk(pdf_path)
        # PDF is only needed transiently to extract text; we don't keep it on disk.

    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    pages = [(c.start_page, c.end_page) for c in chunks]
    chunk_ids = db.insert_chunks(paper_id, texts, pages=pages)
    embeddings = embed.embed_texts(texts)
    vector_store.add_chunks(chunk_ids, texts, embeddings, paper_id, pages=pages)
    return len(chunks)
