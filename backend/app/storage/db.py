"""SQLite metadata store: papers, chunks, citation edges, profile/logs.

Chunk *text* also lives here (source of truth); Chroma stores the same
chunk_id -> embedding mapping for vector search. Keeping text in SQLite
means we never need to round-trip large payloads through the vector store.
"""

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import SQLITE_PATH

_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$")


def canonical_paper_id(paper_id: str) -> str:
    """`1810.04805v2` -> `1810.04805`. Non-arXiv ids (Semantic Scholar hashes) pass
    through unchanged, since they carry no version suffix to strip."""
    return _ARXIV_VERSION_SUFFIX.sub("", paper_id)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,       -- arXiv id or Semantic Scholar id, as first seen
    canonical_id TEXT,               -- version-stripped id; the join key for analysis
    source TEXT NOT NULL,            -- 'arxiv' | 'semantic_scholar'
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,                    -- JSON list
    published TEXT,
    pdf_path TEXT,
    external_ids TEXT                -- JSON dict, e.g. {"ArXiv": "...", "DOI": "..."}
);

-- Why canonical_id exists: the same paper arrives under different ids depending on
-- the path it came in by. arXiv search yields a versioned id (1810.04805v2); the
-- citation graph yields the bare id (1810.04805) because that's what Semantic
-- Scholar uses. Left alone, BERT is two rows - which splits its cohort statistics
-- and, worse, splits it into two nodes in the citation graph, quietly halving the
-- centrality of exactly the influential papers centrality is meant to surface.
-- paper_id stays untouched so existing folders, sessions and chunk ids keep working.
CREATE INDEX IF NOT EXISTS idx_papers_canonical ON papers (canonical_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,       -- f"{paper_id}::{chunk_index}"
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_page INTEGER,              -- NULL for the synthetic abstract chunk
    end_page INTEGER
);

CREATE TABLE IF NOT EXISTS citations (
    citing_paper_id TEXT NOT NULL,
    cited_paper_id TEXT NOT NULL,
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);

CREATE TABLE IF NOT EXISTS interests (
    keyword TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS search_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS folders (
    folder_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Many-to-many: a paper can live in more than one folder (e.g. a general
-- "Want to read" pile and a topic-specific one at the same time).
CREATE TABLE IF NOT EXISTS saved_papers (
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    folder_id INTEGER NOT NULL REFERENCES folders(folder_id),
    saved_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (paper_id, folder_id)
);

-- Scientometric data, kept separate from `papers` on purpose: bibliographic
-- identity (title, authors, abstract) is immutable once published, while citation
-- counts drift every month. Different refresh cadences, different tables - so a
-- metrics refresh never risks rewriting the bibliographic record.
CREATE TABLE IF NOT EXISTS paper_metrics (
    paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id),
    s2_paper_id TEXT,                    -- Semantic Scholar's own id, for citation-graph calls
    citation_count INTEGER,
    influential_citation_count INTEGER,  -- S2's classifier: substantive citations only
    reference_count INTEGER,
    venue TEXT,
    cohort_year INTEGER,                 -- from the arXiv submission date, not the id prefix
    primary_category TEXT,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_metrics_cohort ON paper_metrics (primary_category, cohort_year);

-- Generic key-value scratchpad for long-running crawls, so a run interrupted after
-- three hours resumes where it stopped instead of re-walking the whole corpus.
CREATE TABLE IF NOT EXISTS crawl_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

DEFAULT_FOLDER_NAME = "Want to read"


def init_db() -> None:
    """Three ordered steps, and the order matters.

    SCHEMA is a single executescript, so anything it references must already exist
    by the time it runs - notably its CREATE INDEX statements, which fail outright
    against a column that a migration hasn't added yet. Hence the split: structural
    repairs first, then the schema script, then data-level backfills that need the
    finished schema in place.
    """
    Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        _align_existing_tables(conn)
        conn.executescript(SCHEMA)
        _backfill(conn)


def _align_existing_tables(conn: sqlite3.Connection) -> None:
    """Brings pre-existing tables in line with what SCHEMA is about to assume.

    Runs BEFORE the schema script. Two kinds of work live here: adding columns that
    SCHEMA's indexes reference, and dropping tables whose shape changed in a way
    ALTER TABLE cannot express (a primary-key change) so executescript recreates
    them. A drop is only ever used on a table confirmed unused at the time of the
    change - verified by grep before writing it - never as a general pattern.
    """
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    if "saved_papers" in tables:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(saved_papers)")}
        if "folder_id" not in columns:
            conn.execute("DROP TABLE saved_papers")

    if "papers" in tables:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(papers)")}
        if "canonical_id" not in columns:
            conn.execute("ALTER TABLE papers ADD COLUMN canonical_id TEXT")

    if "paper_metrics" in tables:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(paper_metrics)")}
        # Derived scientometric scores, recomputed from the graph and cohort stats
        # rather than fetched - kept alongside the raw counts so the ranker reads
        # every feature from one row instead of rebuilding the graph per query.
        for column, sql_type in (("pagerank", "REAL"), ("cnci", "REAL"), ("in_degree", "INTEGER")):
            if column not in columns:
                conn.execute(f"ALTER TABLE paper_metrics ADD COLUMN {column} {sql_type}")

    if "chunks" in tables:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
        if "start_page" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN start_page INTEGER")
        if "end_page" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN end_page INTEGER")


def _backfill(conn: sqlite3.Connection) -> None:
    """Data-level migrations, run once the schema is settled."""
    # SQLite, unlike standard SQL, permits NULL in a `TEXT PRIMARY KEY` column, so
    # rows with no paper_id could be written before upsert_paper started rejecting
    # them. They only ever came from Semantic Scholar reference entries that failed
    # to parse into a real record (titles like "2023. Survey of hallucination in..."),
    # and a paper with no id is unreachable by every lookup in the app - so they're
    # junk by definition, not data anyone can miss.
    conn.execute("DELETE FROM papers WHERE paper_id IS NULL OR paper_id = ''")

    # Covers both a freshly-added column and rows written by an older code path.
    missing = conn.execute("SELECT paper_id FROM papers WHERE canonical_id IS NULL").fetchall()
    for row in missing:
        conn.execute(
            "UPDATE papers SET canonical_id = ? WHERE paper_id = ?",
            (canonical_paper_id(row["paper_id"]), row["paper_id"]),
        )


@contextmanager
def connect():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_paper(
    paper_id: str,
    source: str,
    title: str,
    abstract: str | None,
    authors: list[str],
    published: str | None,
    pdf_path: str | None = None,
    external_ids: dict | None = None,
) -> None:
    # Guard the single write chokepoint rather than each caller: SQLite would happily
    # accept a NULL into this TEXT PRIMARY KEY, producing a row no lookup can ever
    # reach. Failing loudly here beats silently accumulating unreferenceable rows.
    if not paper_id:
        raise ValueError(f"paper_id is required (got {paper_id!r}, title={title!r})")

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO papers (
                paper_id, canonical_id, source, title, abstract, authors, published, pdf_path, external_ids
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_id) DO UPDATE SET
                canonical_id=excluded.canonical_id,
                title=excluded.title, abstract=excluded.abstract, authors=excluded.authors,
                published=excluded.published, pdf_path=excluded.pdf_path, external_ids=excluded.external_ids
            """,
            (
                paper_id,
                canonical_paper_id(paper_id),
                source,
                title,
                abstract,
                json.dumps(authors),
                published,
                pdf_path,
                json.dumps(external_ids or {}),
            ),
        )


def insert_chunks(paper_id: str, texts: list[str], pages: list[tuple[int, int]] | None = None) -> list[str]:
    """`pages`, if given, is [(start_page, end_page), ...] aligned with `texts`."""
    chunk_ids = [f"{paper_id}::{i}" for i in range(len(texts))]
    page_pairs = pages if pages is not None else [(None, None)] * len(texts)
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks (chunk_id, paper_id, chunk_index, text, start_page, end_page) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (cid, paper_id, i, text, start, end)
                for i, (cid, text, (start, end)) in enumerate(zip(chunk_ids, texts, page_pairs))
            ],
        )
    return chunk_ids


def get_paper(paper_id: str) -> dict | None:
    """Looks up a paper by exact id, falling back to its canonical (version-stripped)
    id.

    The fallback is required, not a convenience. Ids reach this function from several
    places that disagree about versions - a retrieved chunk, a citation edge, a URL a
    user pasted - and the corpus stores `1706.03762v7` while every one of those may
    say `1706.03762`. Without the fallback, opening the transformer paper from a
    source card returned 404 for the single most-cited paper in the corpus.

    Exact match is tried first so that when both a bare and a versioned row exist,
    asking for one specific version still gets that version.
    """
    with connect() as conn:
        row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT * FROM papers WHERE canonical_id = ? ORDER BY paper_id LIMIT 1",
                (canonical_paper_id(paper_id),),
            ).fetchone()
        return dict(row) if row else None


def get_chunk_text(chunk_id: str) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return row["text"] if row else None


def has_full_text(paper_id: str) -> bool:
    """True once the paper's full text has been chunked (not just its abstract)."""
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM chunks WHERE paper_id = ? AND chunk_index >= 0 LIMIT 1", (paper_id,)
        ).fetchone()
        return row is not None


def set_pdf_path(paper_id: str, pdf_path: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE papers SET pdf_path = ? WHERE paper_id = ?", (pdf_path, paper_id))


def add_citation(citing_paper_id: str, cited_paper_id: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO citations (citing_paper_id, cited_paper_id) VALUES (?, ?)",
            (citing_paper_id, cited_paper_id),
        )


def log_search(query: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO search_log (query) VALUES (?)", (query,))


# --- Corpus building -------------------------------------------------------


def get_crawl_state(key: str, default: str = "") -> str:
    with connect() as conn:
        row = conn.execute("SELECT value FROM crawl_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_crawl_state(key: str, value: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO crawl_state (key, value, updated_at) VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value),
        )


def count_papers() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"]


def upsert_paper_metrics(
    paper_id: str,
    s2_paper_id: str | None,
    citation_count: int | None,
    influential_citation_count: int | None,
    reference_count: int | None,
    venue: str | None,
    cohort_year: int | None,
    primary_category: str | None,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO paper_metrics (
                paper_id, s2_paper_id, citation_count, influential_citation_count,
                reference_count, venue, cohort_year, primary_category, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(paper_id) DO UPDATE SET
                s2_paper_id=excluded.s2_paper_id,
                citation_count=excluded.citation_count,
                influential_citation_count=excluded.influential_citation_count,
                reference_count=excluded.reference_count,
                venue=excluded.venue,
                cohort_year=excluded.cohort_year,
                primary_category=excluded.primary_category,
                fetched_at=excluded.fetched_at
            """,
            (
                paper_id,
                s2_paper_id,
                citation_count,
                influential_citation_count,
                reference_count,
                venue,
                cohort_year,
                primary_category,
            ),
        )


def paper_ids_missing_metrics(limit: int | None = None) -> list[str]:
    """Papers with no metrics row yet - the work queue for the enrichment stage,
    and what makes it resumable after an interrupt."""
    sql = """
        SELECT p.paper_id
        FROM papers p
        LEFT JOIN paper_metrics m ON m.paper_id = p.paper_id
        WHERE m.paper_id IS NULL
        ORDER BY p.paper_id
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with connect() as conn:
        return [row["paper_id"] for row in conn.execute(sql)]


def count_metrics() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM paper_metrics").fetchone()["n"]


def list_folders() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT f.folder_id, f.name, f.created_at, COUNT(sp.paper_id) AS paper_count
            FROM folders f
            LEFT JOIN saved_papers sp ON sp.folder_id = f.folder_id
            GROUP BY f.folder_id
            ORDER BY f.created_at ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def create_folder(name: str) -> int:
    name = name.strip()
    with connect() as conn:
        conn.execute("INSERT OR IGNORE INTO folders (name) VALUES (?)", (name,))
        row = conn.execute("SELECT folder_id FROM folders WHERE name = ?", (name,)).fetchone()
        return row["folder_id"]


def get_or_create_default_folder() -> int:
    return create_folder(DEFAULT_FOLDER_NAME)


def delete_folder(folder_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM saved_papers WHERE folder_id = ?", (folder_id,))
        conn.execute("DELETE FROM folders WHERE folder_id = ?", (folder_id,))


def save_paper_to_folder(paper_id: str, folder_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO saved_papers (paper_id, folder_id) VALUES (?, ?)", (paper_id, folder_id)
        )


def unsave_paper(paper_id: str, folder_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "DELETE FROM saved_papers WHERE paper_id = ? AND folder_id = ?", (paper_id, folder_id)
        )


def list_folder_papers(folder_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT p.*, sp.saved_at
            FROM saved_papers sp
            JOIN papers p ON p.paper_id = sp.paper_id
            WHERE sp.folder_id = ?
            ORDER BY sp.saved_at DESC
            """,
            (folder_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def list_paper_folder_ids(paper_id: str) -> list[int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT folder_id FROM saved_papers WHERE paper_id = ?", (paper_id,)
        ).fetchall()
        return [row["folder_id"] for row in rows]
