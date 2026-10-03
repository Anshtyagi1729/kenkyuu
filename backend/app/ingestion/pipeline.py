"""Two-tier ingestion.

Tier 1 (search_and_record) is cheap: metadata + a single abstract embedding
per result, no PDF download. This is what runs on every search query.

Tier 2 (ingest_full_text) is expensive: PDF download, parsing, chunking, and
embedding the whole paper. It's lazy — only called for a paper the user or
agent actually needs deep grounding on (opened, saved, being compared) — and
cached, so re-requesting an already-ingested paper is a no-op.
"""

import tempfile

from app.ingestion import arxiv_client, embed, pdf_parser, semantic_scholar_client
from app.storage import db, vector_store

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


def record_s2_paper(paper: semantic_scholar_client.S2Paper) -> str | None:
    """Records a Semantic Scholar paper into the corpus. Returns the id it was stored
    under, or None if it carried no usable identifier.

    This closes a real gap. Until now only arXiv papers could be ingested, so a paper
    published at IEEE, Springer, Elsevier or ACM with no preprint could be FOUND via
    Semantic Scholar and then never cited in an answer - synthesis quotes only
    retrieved passages, and nothing had put one in the index. A professor's own journal
    paper was therefore permanently unquotable, which is close to the worst possible
    gap for the intended user.

    Keyed on the arXiv id when S2 reports one, so a paper reachable both ways collapses
    onto a single row instead of appearing twice under two identifiers. Falls back to
    S2's own paperId, which is stable.

    Citation metrics are written alongside, so the scorecard works for these papers
    too - otherwise a non-arXiv paper would display as having no measured impact, which
    reads as "uncited" rather than "never looked up".
    """
    ids = paper.external_ids or {}
    paper_id = ids.get("ArXiv") or paper.paper_id
    if not paper_id:
        return None

    db.upsert_paper(
        paper_id=paper_id,
        source="semantic_scholar",
        title=paper.title,
        abstract=paper.abstract,
        # S2 gives a year, not a date. Stored as January 1st so the column stays a
        # sortable ISO date; the day and month are not real and nothing should read
        # them as such.
        authors=paper.authors,
        published=f"{paper.year}-01-01" if paper.year else None,
        external_ids=ids,
    )
    db.upsert_paper_metrics(
        paper_id=paper_id,
        s2_paper_id=paper.paper_id,
        citation_count=paper.citation_count,
        influential_citation_count=None,
        reference_count=None,
        venue=None,
        cohort_year=paper.year,
        primary_category=None,
    )
    _index_abstract(paper_id, paper.title, paper.abstract, paper.authors)
    return paper_id


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


def ingest_full_text(paper_id: str) -> int:
    """Lazy, cached full-text ingestion. Returns the number of body chunks stored
    (0 if the paper isn't known yet, or if it was already ingested)."""
    if db.has_full_text(paper_id):
        return 0

    paper_row = db.get_paper(paper_id)
    if paper_row is None:
        return 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = arxiv_client.fetch_pdf(arxiv_client.pdf_url_for(paper_id), tmp_dir)
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
