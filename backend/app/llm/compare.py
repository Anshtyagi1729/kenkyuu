"""Structured multi-paper comparison.

Unlike search's open-ended tool loop, comparison already knows which papers
to look at - no discovery needed, so no orchestration loop. It ensures full
text is ingested for each paper (lazy, cached - a no-op if already done),
retrieves the most relevant chunks from EACH paper individually (never
pooling candidates across papers before retrieval, or a longer/denser paper
would crowd out a shorter one), and makes one synthesis call.
"""

from dataclasses import dataclass, field

from app.ingestion import pipeline, retrieval
from app.llm import router
from app.storage import db

DEFAULT_FOCUS = "methodology, results, and key contributions and claims"
CHUNKS_PER_PAPER = 4
CHUNK_CHAR_LIMIT = 800  # per chunk - keeps prompts well under free-tier TPM caps (e.g.
# Groq: 8k TPM). Total prompt size still grows with paper count, so comparing many
# papers at once can still hit the cap - that's a provider-fallback problem, not this.

COMPARISON_SYSTEM_PROMPT = """\
You are comparing academic papers for a professor. Using ONLY the retrieved passages \
below - grouped by paper - write a structured comparison covering methodology, results, \
and where the papers agree or disagree. Cite claims inline using [paper_id]. If a \
paper's retrieved passages don't cover some aspect, say so rather than guessing.

Format any mathematical notation (variables, subscripts, symbols, equations) as LaTeX \
wrapped in $...$ for inline or $$...$$ for display equations - e.g. $x_i$, not x_i. \
This is rendered downstream, so an unwrapped underscore or asterisk in a variable name \
will be misread as markdown emphasis and garble the output."""


@dataclass
class ComparisonAnswer:
    text: str
    sources: list[dict] = field(default_factory=list)


def compare_papers(paper_ids: list[str], focus: str | None = None) -> ComparisonAnswer:
    if len(paper_ids) < 2:
        raise ValueError("Comparison needs at least two papers")

    query = focus or DEFAULT_FOCUS
    all_chunks: list[dict] = []
    sections = []

    for paper_id in paper_ids:
        pipeline.ingest_full_text(paper_id)  # lazy + cached, no-op if already ingested
        paper = db.get_paper(paper_id)
        title = paper["title"] if paper else paper_id

        chunks = retrieval.search(query, top_k=CHUNKS_PER_PAPER, paper_id=paper_id)
        all_chunks.extend(chunks)
        passages = "\n\n".join(f"- {c['text'][:CHUNK_CHAR_LIMIT]}" for c in chunks) or "(no relevant passages found)"
        sections.append(f"### [{paper_id}] {title}\n{passages}")

    prompt = f"Comparison focus: {query}\n\n" + "\n\n".join(sections)
    completion = router.generate(
        [
            {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tier="strong",
    )
    text = completion["choices"][0]["message"]["content"]
    return ComparisonAnswer(text=text, sources=all_chunks)
