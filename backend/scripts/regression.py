"""Smoke test: can the live system still find the papers we promise it can find?

    python -m scripts.regression              # retrieval only (fast, no LLM calls)
    python -m scripts.regression --agent      # also run the full agent on one paraphrase

Different question from `scripts.evaluate`, and deliberately a different mechanism.
`evaluate` measures RANKING QUALITY against a frozen snapshot so numbers stay
comparable across months. This asks whether specific, named papers are still
retrievable from the LIVE index - the index the running app actually serves - using
the wording a real person would type, including the exact wording that failed before.

A benchmark tells you whether the system got better on average. This tells you whether
it broke for a paper someone will demonstrate tomorrow.

Exit code is non-zero when any paper is not found, so this can gate a release.
"""

import argparse
import json
import sys
from pathlib import Path

from app.config import BASE_DIR
from app.ranking import rerank
from app.storage.db import canonical_paper_id

QUERIES_PATH = Path(BASE_DIR) / "data" / "regression_queries.json"

# Found anywhere in the top 5 is a pass; found at rank 1 is what we want. Separating
# the two matters: "present but ranked 4th" and "absent" need different fixes, and
# collapsing them into one boolean hides which one you have.
TOP_K = 5
TARGET_RANK = 1


def rank_of(paper_id: str, query: str, k: int = TOP_K) -> int | None:
    """1-based rank of `paper_id` in the results for `query`, or None if absent.

    Uses `rerank.search_ranked`, the exact path the agent's query_index tool uses, so a
    pass here means the SHIPPED configuration finds it - not that some other
    configuration could have. That distinction caught a real failure: the fixed-weight
    path this replaced put the panel's paper outside the top 5 while a direct vector
    search had it at rank 1.
    """
    hits = rerank.search_ranked(query, top_k=k)
    wanted = canonical_paper_id(paper_id)
    seen: list[str] = []
    for hit in hits:
        pid = canonical_paper_id(hit["paper_id"])
        if pid not in seen:
            seen.append(pid)
    for i, pid in enumerate(seen, start=1):
        if pid == wanted:
            return i
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", action="store_true", help="also run the full agent on one paraphrase per paper")
    args = parser.parse_args()

    if not QUERIES_PATH.exists():
        sys.exit(f"No regression queries at {QUERIES_PATH}")

    spec = json.loads(QUERIES_PATH.read_text())
    failures: list[str] = []
    off_target: list[str] = []

    for entry in spec["queries"]:
        print(f"\n{entry['title'][:72]}")
        print(f"  {entry['paper_id'][:16]}")
        for phrase in entry["paraphrases"]:
            rank = rank_of(entry["paper_id"], phrase)
            if rank is None:
                mark, detail = "FAIL", f"not in top {TOP_K}"
                failures.append(f"{entry['id']}: {phrase!r}")
            elif rank > TARGET_RANK:
                mark, detail = "warn", f"rank {rank}"
                off_target.append(f"{entry['id']}: {phrase!r} at rank {rank}")
            else:
                mark, detail = "ok  ", "rank 1"
            print(f"    [{mark}] {detail:<16} {phrase}")

    print("\n" + "=" * 72)
    if failures:
        print(f"{len(failures)} NOT FOUND:")
        for f in failures:
            print(f"  - {f}")
    if off_target:
        print(f"{len(off_target)} found but not at rank 1:")
        for f in off_target:
            print(f"  - {f}")
    if not failures and not off_target:
        print("All papers found at rank 1.")

    if args.agent:
        _run_agent_checks(spec)

    # Only a miss fails the run. A paper at rank 2 is visible to the user and to the
    # agent's synthesis step; a paper absent from the top 5 effectively does not exist.
    sys.exit(1 if failures else 0)


def _run_agent_checks(spec: dict) -> None:
    """Runs the full agent on the first paraphrase of each paper.

    Separate and opt-in because each of these costs real LLM calls against a free tier
    and takes tens of seconds. Retrieval is what regresses; the agent path is checked
    when you want end-to-end confidence before a demo.
    """
    from app.llm import agent

    print("\n" + "=" * 72)
    print("AGENT END-TO-END (first paraphrase per paper)\n")
    for entry in spec["queries"]:
        phrase = entry["paraphrases"][0]
        print(f"  Q: {phrase}")
        answer = ""
        sources: list[dict] = []
        for kind, payload in agent.answer_query_stream([{"role": "user", "content": phrase}]):
            if kind == "token":
                answer += payload
            elif kind == "done":
                sources = payload
        cited = {canonical_paper_id(s["paper_id"]) for s in sources}
        hit = canonical_paper_id(entry["paper_id"]) in cited
        print(f"     [{'ok' if hit else 'MISS'}] cited the target: {hit}")
        print(f"     answer: {answer[:160].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
