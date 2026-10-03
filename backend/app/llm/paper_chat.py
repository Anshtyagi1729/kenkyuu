"""Chat scoped to a single paper.

Unlike the open-ended search agent, we already know which paper is being
discussed, so there's no discovery/tool-loop needed - just deterministic
retrieval within that paper's chunks (ensuring full text is ingested first,
lazily and cached) plus one synthesis call that's aware of the conversation
so far, for natural follow-ups.
"""

from app.ingestion import pipeline, retrieval
from app.storage import db
from app.llm import router

CHUNKS_PER_TURN = 6
CHUNK_CHAR_LIMIT = 1000  # more generous than multi-paper flows - only one paper's worth here

PAPER_CHAT_SYSTEM_PROMPT = """\
You are discussing one specific academic paper with a professor, across a multi-turn \
conversation. Answer using ONLY the retrieved passages from this paper provided below \
and the conversation so far - do not use outside knowledge, and do not invent details \
not present in the passages. If the passages don't cover what's asked, say so rather \
than guessing.

Format any mathematical notation (variables, subscripts, symbols, equations) as LaTeX \
wrapped in $...$ for inline or $$...$$ for display equations - e.g. $x_i$, not x_i. \
This is rendered downstream, so an unwrapped underscore or asterisk in a variable name \
will be misread as markdown emphasis and garble the output."""


def _build_prompt(paper_id: str, history: list[dict]) -> tuple[str, list[dict]]:
    if not history or history[-1]["role"] != "user":
        raise ValueError("history must be non-empty and end with a user message")

    paper = db.get_paper(paper_id)
    if paper is None:
        raise ValueError(f"unknown paper_id: {paper_id}")

    pipeline.ingest_full_text(paper_id)  # lazy + cached, no-op if already ingested

    question = history[-1]["content"]
    chunks = retrieval.search(question, top_k=CHUNKS_PER_TURN, paper_id=paper_id)
    passages = "\n\n".join(f"- {c['text'][:CHUNK_CHAR_LIMIT]}" for c in chunks) or "(no relevant passages found)"

    parts = [f"Paper: {paper['title']} [{paper_id}]"]
    if len(history) > 1:
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[:-1])
        parts.append(f"Conversation so far:\n{convo}")
    parts.append(f"Latest question: {question}")
    parts.append(f"Retrieved passages:\n\n{passages}")

    return "\n\n".join(parts), chunks


def chat_about_paper_stream(paper_id: str, history: list[dict]):
    """Same as chat_about_paper, but streams the answer. Yields ("token", str)
    chunks while the answer is being written, then one final ("done", list[dict])."""
    prompt, chunks = _build_prompt(paper_id, history)
    messages = [
        {"role": "system", "content": PAPER_CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    for delta in router.generate_stream(messages, tier="strong"):
        yield "token", delta
    yield "done", chunks
