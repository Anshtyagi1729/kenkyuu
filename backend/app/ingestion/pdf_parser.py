"""PDF text extraction and chunking.

Deliberately simple: plain per-page text extraction and paragraph-based
chunking, no layout/section-structure parsing. Good enough for RAG recall;
table/figure-aware extraction is an explicit non-goal for sem1.

Page numbers ARE tracked per chunk (1-indexed), though - a citation that
just says "paper X" without a page number is a lot less useful for actually
going and checking the claim than one that says "paper X, p. 4".
"""

from dataclasses import dataclass

import fitz  # PyMuPDF

TARGET_CHUNK_CHARS = 1500
MIN_CHUNK_CHARS = 200


@dataclass
class Chunk:
    text: str
    chunk_index: int
    start_page: int
    end_page: int


def _extract_paragraphs(pdf_path: str) -> list[tuple[int, str]]:
    """Returns [(page_number, paragraph_text), ...], page_number is 1-indexed."""
    pairs: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            for para in page.get_text().split("\n\n"):
                para = para.strip()
                if para:
                    pairs.append((page_number, para))
    return pairs


def chunk_text(paragraphs: list[tuple[int, str]]) -> list[Chunk]:
    chunks: list[tuple[str, int, int]] = []  # (text, start_page, end_page)
    buffer = ""
    buffer_start_page: int | None = None
    buffer_end_page: int | None = None

    for page_number, para in paragraphs:
        if buffer and len(buffer) + len(para) > TARGET_CHUNK_CHARS:
            chunks.append((buffer.strip(), buffer_start_page, buffer_end_page))  # type: ignore[arg-type]
            buffer = ""
            buffer_start_page = None
        if buffer_start_page is None:
            buffer_start_page = page_number
        buffer_end_page = page_number
        buffer = f"{buffer}\n\n{para}" if buffer else para

    if buffer.strip():
        chunks.append((buffer.strip(), buffer_start_page, buffer_end_page))  # type: ignore[arg-type]

    # Fold any trailing too-small chunk into the previous one rather than
    # storing a near-empty embedding. Its page range extends the previous one's.
    if len(chunks) > 1 and len(chunks[-1][0]) < MIN_CHUNK_CHARS:
        prev_text, prev_start, _prev_end = chunks[-2]
        last_text, _last_start, last_end = chunks[-1]
        chunks[-2] = (f"{prev_text}\n\n{last_text}", prev_start, last_end)
        chunks.pop()

    return [
        Chunk(text=text, chunk_index=i, start_page=start, end_page=end)
        for i, (text, start, end) in enumerate(chunks)
    ]


def parse_and_chunk(pdf_path: str) -> list[Chunk]:
    return chunk_text(_extract_paragraphs(pdf_path))
