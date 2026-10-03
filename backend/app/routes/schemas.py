"""Response models shared across routes."""

import json

from pydantic import BaseModel


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str | None
    authors: list[str]
    published: str | None
    external_ids: dict


def paper_from_row(row: dict) -> Paper:
    return Paper(
        paper_id=row["paper_id"],
        title=row["title"],
        abstract=row["abstract"],
        authors=json.loads(row["authors"] or "[]"),
        published=row["published"],
        external_ids=json.loads(row["external_ids"] or "{}"),
    )


class Source(BaseModel):
    paper_id: str
    chunk_id: str
    text: str
    start_page: int | None = None
    end_page: int | None = None


def source_from_dict(chunk: dict) -> Source:
    return Source(
        paper_id=chunk["paper_id"],
        chunk_id=chunk["chunk_id"],
        text=chunk["text"],
        start_page=chunk.get("start_page"),
        end_page=chunk.get("end_page"),
    )


class PaperMetrics(BaseModel):
    """A paper's scientometric scorecard.

    Every numeric field is nullable, and null means "not computed", never zero. The
    frontend has to render those two cases differently - a 2026 preprint with 0
    citations is a fact, while a paper whose cohort was too small to normalize has no
    field-normalized impact at all, and showing "0x the field median" for the second
    would be a fabrication.
    """

    paper_id: str
    title: str
    citation_count: int | None
    influential_citation_count: int | None
    influential_ratio: float | None
    cnci: float | None
    cnci_percentile: float | None
    pagerank: float | None
    pagerank_percentile: float | None
    in_degree: int | None
    venue: str | None
    cohort_year: int | None
    primary_category: str | None
    summary: str
