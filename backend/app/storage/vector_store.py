"""Chroma wrapper — embedded, no server, zero ops.

Two collections, and the split is a methodology requirement rather than a
performance one:

  paper_chunks       Live. The running app writes here, and it grows every time the
                     agent discovers a paper on arXiv. That is correct product
                     behaviour and should not be prevented.

  paper_chunks_eval  Frozen. Rebuilt deliberately from SQLite, never written to by
                     the app.

Measuring against the live collection would mean the corpus silently changes between
one configuration's measurement and the next - a newly ingested paper can outrank an
evaluation target - so a moved score could be the change under test or could be
corpus drift, with no way to tell them apart. Comparing rows of the results table
requires every row to have seen the same corpus.

SQLite stays the single source of truth; both collections are derived from it, so
there is nothing to keep in sync - the eval collection is simply rebuilt when the
corpus or the embedding model deliberately changes.
"""

from functools import lru_cache

import chromadb

from app.config import CHROMA_DIR

COLLECTION_NAME = "paper_chunks"
EVAL_COLLECTION_NAME = "paper_chunks_eval"


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _collection(name: str = COLLECTION_NAME):
    return _client().get_or_create_collection(name)


def add_chunks(
    chunk_ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    paper_id: str,
    pages: list[tuple[int | None, int | None]] | None = None,
    collection: str = COLLECTION_NAME,
) -> None:
    if not chunk_ids:
        return
    page_pairs = pages if pages is not None else [(None, None)] * len(chunk_ids)
    metadatas = []
    for start, end in page_pairs:
        # Chroma metadata values must be primitives - None isn't valid, so the
        # synthetic abstract chunk (no real page) just omits these keys entirely.
        meta: dict = {"paper_id": paper_id}
        if start is not None:
            meta["start_page"] = start
        if end is not None:
            meta["end_page"] = end
        metadatas.append(meta)
    _collection(collection).upsert(ids=chunk_ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def add_abstract_chunks(
    chunk_ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    paper_ids: list[str],
    collection: str = COLLECTION_NAME,
) -> None:
    """Bulk-upserts abstract chunks spanning MANY papers in one call.

    add_chunks above stamps a single paper_id onto every chunk, which is right when
    chunking one paper's full text but wrong for corpus indexing, where each chunk
    belongs to a different paper. Doing that through add_chunks means one Chroma
    round-trip per paper; this is one per batch, which is the difference between
    indexing a corpus in minutes and in most of an hour.

    No page metadata: an abstract has no page span, same as the abstract chunks
    written by record_search_result.
    """
    if not chunk_ids:
        return
    _collection(collection).upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"paper_id": pid} for pid in paper_ids],
    )


def query(
    embedding: list[float],
    top_k: int = 5,
    paper_id: str | None = None,
    collection: str = COLLECTION_NAME,
) -> list[dict]:
    kwargs = {"query_embeddings": [embedding], "n_results": top_k}
    if paper_id:
        kwargs["where"] = {"paper_id": paper_id}
    result = _collection(collection).query(**kwargs)
    hits = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "paper_id": meta["paper_id"],
                "distance": distance,
                "start_page": meta.get("start_page"),
                "end_page": meta.get("end_page"),
            }
        )
    return hits


def count(collection: str = COLLECTION_NAME) -> int:
    return _collection(collection).count()


def snapshot_to_eval(batch_size: int = 1000) -> int:
    """Replaces the eval collection with a copy of the live one. Returns vectors copied.

    Copies the stored vectors rather than re-embedding: it takes seconds instead of
    minutes, and more importantly it guarantees the snapshot is bit-identical to what
    was measured, where a re-embed could drift if the model or its version changed.

    Destructive by design - the eval collection is a snapshot, not an accumulator.
    Letting it merge old and new vectors would reintroduce exactly the silent-drift
    problem the split exists to prevent.
    """
    client = _client()
    try:
        client.delete_collection(EVAL_COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - absent on first run, which is fine
        pass

    source = _collection(COLLECTION_NAME)
    target = client.get_or_create_collection(EVAL_COLLECTION_NAME)

    total = source.count()
    copied = 0
    while copied < total:
        batch = source.get(
            limit=batch_size,
            offset=copied,
            include=["embeddings", "documents", "metadatas"],
        )
        if not batch["ids"]:
            break
        target.upsert(
            ids=batch["ids"],
            embeddings=batch["embeddings"],
            documents=batch["documents"],
            metadatas=batch["metadatas"],
        )
        copied += len(batch["ids"])

    return copied
