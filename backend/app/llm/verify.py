"""Groundedness verification — does the answer actually rest on retrieved passages?

Following Self-RAG (Asai et al., 2023, arXiv:2310.11511), which has the model critique
its own output before it is returned.

The existing check in agent.py is a regex over paper ids in the answer text. That
verifies CITATION FORMAT, not grounding: an answer can cite [2005.11401] beside a
claim that paper never makes, and the regex is satisfied. It catches a missing
citation and is blind to a wrong one - which is the failure that actually matters for
a research tool, because a confident wrong citation is worse than no citation.

This checks the other direction: for each claim, is it supported by the passages we
actually retrieved?

Deliberately advisory. It returns a verdict and unsupported claims; it does not
rewrite or suppress the answer. Silently dropping content on a verifier's say-so
would make failures invisible, and the verifier is a cheap-tier model that is itself
sometimes wrong. Surfacing the doubt beats hiding the answer.
"""

import json
import logging
from dataclasses import dataclass, field

from app.llm import router

logger = logging.getLogger(__name__)

VERIFY_CHUNK_CHARS = 400
MAX_CHUNKS = 8
MAX_ANSWER_CHARS = 2000

VERIFY_PROMPT = """\
You are checking whether an answer is supported by the passages it was written from.

Consider only factual claims about papers - their methods, results, or findings. \
Ignore hedging, framing, and statements about what is absent from the material.

A claim is supported if the passages state it or directly imply it. A claim is \
unsupported if it adds specifics the passages do not contain, or attributes something \
to a paper the passages do not show that paper saying.

Reply with JSON only:
{"verdict": "grounded" | "partially_grounded" | "ungrounded", "unsupported": ["..."]}

List at most 3 unsupported claims, quoted briefly. Empty list if all are supported."""


@dataclass
class Groundedness:
    verdict: str
    unsupported: list[str] = field(default_factory=list)

    @property
    def is_trustworthy(self) -> bool:
        return self.verdict in ("grounded", "partially_grounded")


def verify_answer(answer: str, chunks: list[dict]) -> Groundedness:
    """Checks an answer against the passages it was generated from.

    Returns 'grounded' whenever verification cannot meaningfully run - no passages, no
    answer, or an LLM failure. A verifier that fails closed would flag every answer
    during an outage, training whoever reads the warnings to ignore them.
    """
    if not chunks or not answer.strip():
        return Groundedness("grounded")

    passages = "\n\n".join(f"[{c['paper_id']}] {c['text'][:VERIFY_CHUNK_CHARS]}" for c in chunks[:MAX_CHUNKS])
    user_content = f"Passages:\n\n{passages}\n\nAnswer to check:\n\n{answer[:MAX_ANSWER_CHARS]}"

    try:
        completion = router.generate(
            [
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tier="cheap",
        )
        raw = completion["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groundedness check failed (%s) - not flagging", exc)
        return Groundedness("grounded")

    return _parse(raw)


def _parse(raw: str) -> Groundedness:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return Groundedness("grounded")

    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return Groundedness("grounded")

    verdict = str(parsed.get("verdict", "grounded")).strip().lower()
    if verdict not in ("grounded", "partially_grounded", "ungrounded"):
        verdict = "grounded"

    unsupported = parsed.get("unsupported") or []
    if not isinstance(unsupported, list):
        unsupported = []

    return Groundedness(verdict, [str(claim)[:200] for claim in unsupported[:3]])
