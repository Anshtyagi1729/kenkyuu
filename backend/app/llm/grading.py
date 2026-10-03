"""Query analysis and retrieval grading — the self-checking parts of the agent loop.

Today's loop retrieves and then trusts whatever comes back. When retrieval is poor it
has no way to notice and no way to recover, so a bad answer and a good one are
produced by the same code path and look alike from the outside.

Two cheap checks fix that, both on the cheap model tier:

  classify_intent   What KIND of question is this? Drives retrieval strategy now and
                    becomes a ranking feature later - the Phase 1 bake-off showed
                    lexical matching helps specific-paper queries (+0.053 nDCG) while
                    hurting topical ones (-0.133), so knowing the intent is worth real
                    accuracy rather than being merely tidy.

  grade_retrieval   Do these passages actually answer the question? Replaces a blind
                    fallback that fired only when retrieval returned literally nothing,
                    and so never caught the more common failure: plenty of passages,
                    none of them relevant.

Both degrade safely. If the LLM is unavailable or answers in an unexpected format,
each returns the permissive default and the agent proceeds exactly as it does today -
a broken grader must not be able to block answers.

Technique provenance: retrieval grading and corrective retry follow Corrective RAG
(Yan et al., 2024, arXiv:2401.15884); the groundedness check in verify.py follows
Self-RAG (Asai et al., 2023, arXiv:2310.11511).
"""

import json
import logging
from dataclasses import dataclass
from typing import Literal

from app.llm import router

logger = logging.getLogger(__name__)

Intent = Literal["specific_paper", "topical", "recency", "comparison"]
VALID_INTENTS: tuple[Intent, ...] = ("specific_paper", "topical", "recency", "comparison")

Grade = Literal["sufficient", "partial", "irrelevant"]
VALID_GRADES: tuple[Grade, ...] = ("sufficient", "partial", "irrelevant")

# Passages are truncated hard before grading. The grader only needs to judge topical
# relevance, which the opening lines settle, and the cheap tier runs under a tight
# tokens-per-minute cap on the free tiers this project targets.
GRADING_CHUNK_CHARS = 300
MAX_CHUNKS_TO_GRADE = 6

INTENT_PROMPT = """\
Classify the researcher's question into exactly one category.

specific_paper - asks for one particular paper by name, title, author, or by what it \
introduced ("show me the original transformer paper", "find the LoRA paper")
topical - asks how something works or what is known about a subject ("how does \
retrieval improve factual accuracy")
recency - asks what is new, latest, or currently happening in an area
comparison - asks to contrast two or more specific things

Answer with the category name alone, nothing else."""

GRADING_PROMPT = """\
You are judging whether retrieved passages can answer a researcher's question. Judge \
only what is present - do not use outside knowledge, and do not guess at what other \
passages might exist.

Reply with JSON only: {"grade": "...", "reason": "..."}

grade must be one of:
sufficient - the passages contain what is needed to answer
partial - related material, but the question cannot be fully answered from it
irrelevant - the passages do not address the question

Keep reason under 15 words."""


@dataclass
class RetrievalGrade:
    grade: Grade
    reason: str

    @property
    def needs_retry(self) -> bool:
        """Only a clear miss is worth a retry. Retrying on 'partial' would burn a
        second retrieval round on questions the corpus simply cannot fully answer,
        and the honest response there is a partial answer that says so."""
        return self.grade == "irrelevant"


def classify_intent(query: str) -> Intent:
    """Best-effort intent classification. Falls back to 'topical', the most common
    case and the one whose retrieval strategy is the least specialized."""
    try:
        completion = router.generate(
            [
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": query},
            ],
            tier="cheap",
        )
        answer = (completion["choices"][0]["message"]["content"] or "").strip().lower()
    except Exception as exc:  # noqa: BLE001 - classification must never block answering
        logger.warning("Intent classification failed (%s) - defaulting to topical", exc)
        return "topical"

    for intent in VALID_INTENTS:
        if intent in answer:
            return intent

    logger.info("Unrecognized intent response %r - defaulting to topical", answer[:60])
    return "topical"


def grade_retrieval(query: str, chunks: list[dict]) -> RetrievalGrade:
    """Judges whether retrieved passages can answer the question."""
    if not chunks:
        return RetrievalGrade("irrelevant", "no passages retrieved")

    passages = "\n\n".join(
        f"[{c['paper_id']}] {c['text'][:GRADING_CHUNK_CHARS]}" for c in chunks[:MAX_CHUNKS_TO_GRADE]
    )
    user_content = f"Question: {query}\n\nRetrieved passages:\n\n{passages}"

    try:
        completion = router.generate(
            [
                {"role": "system", "content": GRADING_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tier="cheap",
        )
        raw = completion["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001
        # Assume the passages are usable: a grader outage must not turn into a
        # refusal to answer with material that may well be perfectly good.
        logger.warning("Retrieval grading failed (%s) - assuming sufficient", exc)
        return RetrievalGrade("sufficient", "grader unavailable")

    return _parse_grade(raw)


def _parse_grade(raw: str) -> RetrievalGrade:
    """Extracts a grade from the model's reply, tolerating prose around the JSON."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            grade = str(parsed.get("grade", "")).strip().lower()
            if grade in VALID_GRADES:
                return RetrievalGrade(grade, str(parsed.get("reason", ""))[:120])
        except (ValueError, AttributeError):
            pass

    # No usable JSON - fall back to scanning for a bare grade word, then to the
    # permissive default rather than discarding retrieved passages over formatting.
    lowered = text.lower()
    for grade in VALID_GRADES:
        if grade in lowered:
            return RetrievalGrade(grade, "parsed from unstructured reply")

    logger.info("Ungradeable response %r - assuming sufficient", text[:60])
    return RetrievalGrade("sufficient", "grade not parseable")
