"""Tool-calling research agent: search -> retrieve -> synthesize.

Two-stage design:
  1. Orchestration loop (cheap model tier) - decides which tools to call
     (federated search over arXiv, OpenAlex, Crossref and Semantic Scholar, query the
     vector index, pull in a paper's full text)
     across up to MAX_TOOL_ITERATIONS steps. This is most of an agentic
     loop's calls, and doesn't need a strong model to pick a tool.
  2. Synthesis (strong model tier, single call, no tools) - given only the
     chunks actually retrieved during orchestration, writes one grounded,
     citation-backed answer. Kept as a separate call so the model doing the
     writing isn't distracted by tool-call bookkeeping.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from app.ingestion import (
    arxiv_client,
    crossref_client,
    openalex_client,
    pipeline,
    retrieval,
    semantic_scholar_client,
)
from app.storage import db
from app.llm import grading, router
from app.ranking import identity, rerank
from app.config import S2_API_KEY
from app.scientometrics import related, scorecard

logger = logging.getLogger(__name__)

# Tools that address specific papers by name or id, rather than guessing at relevance.
# Their results are not a retrieval hypothesis that could be wrong - they are the thing
# that was asked for - so the corrective loop must not second-guess them.
# Marks a chunk as coming from a tool that identified a paper by name or id, rather
# than guessing at relevance. Synthesis sees passages in list order and its budget
# truncates the tail, so provenance has to survive into the ordering or the budget
# decides relevance by accident.
PRIORITY_KEY = "_identified"

EXACT_LOOKUP_TOOLS = frozenset({"search_by_author", "get_paper_passages", "find_related_papers"})

MAX_TOOL_ITERATIONS = 5
RETRY_TOP_K = 8  # a corrective retry casts slightly wider than the first pass


# The ranker can only reorder what was retrieved, so the candidate pool has to be deep
# enough to contain a paper the cross-encoder has pushed down. A pool of 20 would have
# already discarded the transformer paper before citation evidence ever saw it.
#
# Used by the corrective retry. The main retrieval path takes its pool depth from
# rerank.POOL_SIZE, which is the same 60 for the same reason.
CANDIDATE_POOL = 60
ARXIV_ID_PATTERN = r"\d{4}\.\d{4,5}(?:v\d+)?"

# Every id form a corpus paper can carry: arXiv ids, OpenAlex work ids (journal papers
# recorded from OpenAlex) and Semantic Scholar hashes (papers recorded from S2). Citation
# matching that knew only the first form treated every non-arXiv citation as uncited.
PAPER_ID_PATTERN = r"(?<![\w/.])(?:\d{4}\.\d{4,5}(?:v\d+)?|W\d{7,}|[0-9a-f]{40})(?![\w/])"

# Written as a decision table rather than prose. The orchestrator runs on the cheap
# model tier and now picks among seven tools; the earlier prose version buried the
# deciding cue for each tool inside a paragraph, and measurably lost tool calls -
# asked to compare two papers' influence it skipped get_paper_metrics entirely and
# answered that it had no citation data. One line per tool, leading with the trigger,
# is what a small model can actually act on.
ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a research assistant helping a professor find and understand academic papers. \
Conversations are multi-turn; a later message may be a follow-up about papers already \
discussed.

PICK TOOLS BY WHAT THE QUESTION ASKS FOR:

- Names a PERSON ("papers by Kaiming He", "what has Bengio published lately", "my \
  recent work") -> search_by_author. Do NOT put a person's name into search_papers; \
  that returns papers that merely mention them.
- Asks for ONE SPECIFIC PAPER ("the original transformer paper", "the paper that \
  introduced X") -> search_papers, always, even if query_index already returns \
  something plausible. A survey or follow-up that merely discusses the target can \
  outrank the target itself on topic similarity. Search with just the title or core \
  phrase ("attention is all you need") - adding "original", "paper", or author names \
  dilutes the match and can surface a different paper on the same topic.
- Asks about a TOPIC -> query_index first (papers already ingested), then \
  search_papers to discover new ones.
- Asks what is RELATED to a known paper ("what else should I read", "what came out of \
  this") -> find_related_papers. This uses citation structure and finds work that \
  text search cannot.
- COMPARES specific papers, or asks what a specific paper says -> get_paper_passages \
  with those paper_ids.
- Asks how INFLUENTIAL / important / foundational something is -> get_paper_metrics.
- Needs detail beyond an abstract -> ingest_full_text for that one paper. Expensive; \
  never speculative.

CRITICAL: search_papers returns short listings for DISCOVERY only - they are not \
citable passages. After it finds relevant papers you MUST call query_index (or \
get_paper_passages) to retrieve the actual text the answer will be grounded in. \
search_by_author is the exception: its results are already citable, so answer a \
"what has X published" question directly from them rather than searching the index \
for the question, which would return whatever merely reads like it.

search_papers takes a `sort`: use "recency" when the question asks for recent, latest, \
new, or current work, or names a recent year - relevance ranking has no concept of \
time and will happily return a 2017 paper for "what's new". Use "relevance" (default) \
when the question wants the defining or foundational work.

search_papers labels every result with how well its title matches what was asked:

- exact_title_match   this IS the paper named. Safe to present as such.
- strong_title_match  the title contains the words asked for, plus more. Often right, \
  but it is also exactly how a paper ABOUT the target looks - a survey or commentary \
  quoting the title. Confirm before asserting identity.
- partial_title_match / topic_match_only  a paper on the same subject, NOT the paper \
  named. Never present one of these as the paper the user asked for.

If a user names a paper and nothing comes back above partial, say plainly that you \
could not find that exact paper, show the closest candidates, and ask them to confirm \
the title or give an author. Do not quietly substitute a different paper.

If the result carries `coverage_warning`, no source covering journals and conferences \
answered, so a journal paper could not have been found at all: say so, and never tell \
the user their paper does not exist. An individual entry in `sources_unavailable` is \
normal (Semantic Scholar is skipped without an API key) and needs no comment unless \
nothing useful came back.

Whenever you recommend, rank, or compare papers, call get_paper_metrics on them and \
quote real numbers. "Highly influential" is an opinion the professor cannot check; \
"6,446x the median 2017 paper in its field" is a measurement they can. Prefer the \
field-normalized figure over the raw citation count - a raw count largely reflects a \
paper's age.

If the latest message is a follow-up the conversation already answers, you may need no \
tools at all. Once you have enough retrieved context, stop calling tools and say you \
are ready to synthesize."""

SYNTHESIS_SYSTEM_PROMPT = """\
You are writing the final answer to the professor's latest message in an ongoing \
conversation, using ONLY the conversation so far and the retrieved passages provided \
below - do not use outside knowledge, and do not invent paper titles, authors, or \
findings that aren't in the passages or earlier turns.

Every factual claim must cite the paper it came from using [paper_id] inline. If the \
available material doesn't fully answer the question, say so explicitly rather than \
filling the gap with unsupported claims.

If the professor asked for one specific paper by name or title, only state that you \
found it when the retrieved material actually IS that paper. A paper on the same topic, \
or one that merely mentions or surveys the paper asked for, is not it. In that case say \
you could not find the exact paper, list the closest matches, and say what would help - \
an author name, the year, the full title. Presenting a near-miss as the requested paper \
is the worst failure this system can produce: the professor concludes their own work is \
missing from the literature, or cites the wrong paper.

A "Measured impact" section may follow the passages. Those figures are computed from \
the citation graph, not extracted from the passages, and they ARE part of the material \
you may use - quote them whenever the question touches on how influential, important, \
foundational or well-established a paper is, and whenever you recommend or rank papers. \
Prefer the field-normalized multiple over the raw citation count when both are given: a \
raw count largely reflects a paper's age, so comparing a 2017 paper to a 2024 one on raw \
counts compares their ages rather than their impact. Never report a metric that is not \
listed there, and never treat an absent metric as zero - if a paper's entry says its \
impact could not be normalized, say that rather than implying it has no impact.

Format any mathematical notation (variables, subscripts, symbols, equations) as LaTeX \
wrapped in $...$ for inline or $$...$$ for display equations - e.g. $x_i$, not x_i. \
This is rendered downstream, so an unwrapped underscore or asterisk in a variable name \
will be misread as markdown emphasis and garble the output."""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for papers by title or topic across arXiv, OpenAlex (journals, conferences and preprints) and Crossref, in parallel; Semantic Scholar too when configured. Any source that is down is skipped and reported, and the rest still answer. Each result is labelled with how exactly its title matches the query. Records metadata and abstract for each result (no full-text download).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "recency"],
                        "default": "relevance",
                        "description": "'recency' for recent/latest/new-work questions, 'relevance' for best textual match (e.g. finding a foundational paper).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_index",
            "description": "Semantic search over already-ingested paper chunks (abstracts + any full text). Reranked for precision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_metrics",
            "description": (
                "Citation and impact metrics for specific papers: total citations, "
                "field-normalized impact (how many times the median paper of the same "
                "field and year), citation-network centrality percentile, how many "
                "corpus papers cite it, and publication venue. Call this when the user "
                "asks how influential, important, foundational, well-cited or "
                "highly-regarded a paper is, when comparing the standing of two papers, "
                "or whenever a recommendation would be more useful with concrete numbers "
                "attached. Returns only measured values - a field is absent when it could "
                "not be computed, and absent never means zero."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arXiv ids, e.g. ['1706.03762', '1810.04805']",
                    }
                },
                "required": ["paper_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_author",
            "description": (
                "Find papers by a specific researcher, newest first, optionally within a "
                "year range. Use this whenever the question names a person - 'what has "
                "Yoshua Bengio published lately', 'papers by Kaiming He on detection', "
                "'my recent work'. A topic search with a name in the query is NOT the same "
                "thing and returns papers that merely mention the name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "author": {"type": "string", "description": "Full name, e.g. 'Yoshua Bengio'"},
                    "max_results": {"type": "integer", "default": 8},
                    "year_from": {"type": "integer", "description": "Earliest submission year, inclusive"},
                    "year_to": {"type": "integer", "description": "Latest submission year, inclusive"},
                },
                "required": ["author"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_related_papers",
            "description": (
                "Papers related to a given paper by CITATION STRUCTURE rather than wording - "
                "either works frequently cited alongside it (co-citation), or works that "
                "share its references (bibliographic coupling). Use this for 'what else "
                "should I read alongside this', 'what is related to this paper', 'what came "
                "out of this line of work'. This finds papers the index's text search cannot: "
                "two papers can share no vocabulary and still be each other's nearest "
                "neighbours because the field always cites them together. Returns nothing "
                "when the paper has no citation links inside the corpus, which is common and "
                "does not mean no related work exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "arXiv id, e.g. 1706.03762"},
                    "limit": {"type": "integer", "default": 8},
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_passages",
            "description": (
                "Retrieve the most relevant passages from EACH of several named papers "
                "separately, on a given aspect. Use this to compare or contrast specific "
                "papers, or to ask what a particular paper says about something. Retrieving "
                "per paper matters: a single pooled search lets one long or densely-worded "
                "paper supply every passage, so the comparison silently becomes a summary of "
                "that paper alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_ids": {"type": "array", "items": {"type": "string"}},
                    "focus": {
                        "type": "string",
                        "description": "What to retrieve about, e.g. 'training data and evaluation results'",
                    },
                },
                "required": ["paper_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_full_text",
            "description": "Downloads and indexes a paper's full text (expensive, cached). Only use when abstract-level context is insufficient.",
            "parameters": {
                "type": "object",
                "properties": {"paper_id": {"type": "string", "description": "arXiv id, e.g. 2005.11401v4"}},
                "required": ["paper_id"],
            },
        },
    },
]


def _activity_label(name: str, args: dict) -> str:
    """A plain-language description of one tool call, for the user to read while it runs.

    The agent takes tens of seconds on a real question, and the whole of that was
    previously three bouncing dots. Naming each step turns dead waiting into a visible
    trace - it tells the professor what the system is actually doing, and it surfaces
    capabilities (author lookup, citation-structure neighbours) that are otherwise
    invisible unless you already knew to ask for them.

    Phrased as what it means, not what it calls: "Looking up how often these are
    cited" rather than "get_paper_metrics".
    """
    if name == "search_papers":
        sort = " (newest first)" if args.get("sort") == "recency" else ""
        return f"Searching arXiv, OpenAlex and Crossref for \u201c{args.get('query', '')}\u201d{sort}"
    if name == "search_by_author":
        years = ""
        if args.get("year_from") or args.get("year_to"):
            years = f" ({args.get('year_from', '')}\u2013{args.get('year_to', '')})"
        return f"Looking up papers by {args.get('author', 'this author')}{years}"
    if name == "query_index":
        return f"Searching the indexed corpus for \u201c{args.get('query', '')}\u201d"
    if name == "find_related_papers":
        return f"Finding papers cited alongside {args.get('paper_id', 'this paper')}"
    if name == "get_paper_passages":
        count = len(args.get("paper_ids") or [])
        focus = args.get("focus")
        subject = f"{count} papers" if count != 1 else "this paper"
        return f"Reading {subject}" + (f" on {focus}" if focus else "")
    if name == "get_paper_metrics":
        return "Looking up citation impact and network centrality"
    if name == "ingest_full_text":
        return f"Downloading the full text of {args.get('paper_id', 'a paper')}"
    return f"Running {name}"


def _execute_tool(name: str, args: dict, collected_chunks: list[dict]) -> dict | list[dict]:
    # A tool failing (rate limit, transient network error, bad arg) shouldn't crash the
    # whole request - it should read as a normal tool result the model can react to
    # (try a different tool, proceed with what it already has, or tell the user why).
    logger.info("tool call: %s(%s)", name, json.dumps(args)[:200])
    try:
        return _run_tool(name, args, collected_chunks)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see comment above
        logger.warning("Tool %s failed: %s", name, exc)
        return {"error": f"{name} failed: {exc}"}


# A query is treated as naming a specific paper when enough of its words land in a
# candidate's title. Token overlap rather than edit distance, because the way people
# misremember a title is by reordering and dropping words - "error weighted drift
# adopting ensemble framework" for "An Error-Weighted Ensemble Framework for Concept
# Drift Adaptation" - which wrecks edit distance while leaving overlap high.
TITLE_MATCH_STRONG = 0.75   # almost certainly the paper being asked for
TITLE_MATCH_PARTIAL = 0.45  # plausibly it; the agent must not assert that it is


# Title tokenisation and similarity live in ranking/identity.py, shared with the
# retrieval path; aliased here so the call sites below keep reading naturally.
_title_tokens = identity.title_tokens
_title_similarity = identity.title_similarity


_QUESTION_OPENERS = (
    "what", "which", "who", "how", "why", "when", "where", "is ", "are ", "does ",
    "do ", "can ", "should ", "tell ", "explain ", "summarize ", "summarise ",
    "compare ",
)


def _looks_like_a_title(query: str) -> bool:
    """True when the query is naming a paper rather than asking a question.

    Gates the title-match lookup, which is one extra request. "what is LoRA" is a
    question and the exact-title endpoint has nothing useful to say about it; "error
    weighted drift adopting ensemble framework" is someone reciting a title from
    memory, and is exactly what that endpoint is for.

    Deliberately permissive - a false positive costs one cheap request whose result is
    then labelled honestly by title similarity anyway, while a false negative loses the
    only route to a non-arXiv paper.
    """
    q = query.strip().lower()
    if not q or q.endswith("?"):
        return False
    return not q.startswith(_QUESTION_OPENERS)


def _match_label(score: float, query: str = "", title: str = "") -> str:
    """Names how well a candidate's title accounts for the query.

    Four tiers rather than three, because "the title is what you asked for" and "the
    title contains what you asked for, plus a lot more" are different claims and the
    agent must not state the second as the first.
    """
    # Exact means the query IS the title, word for word in order (identity.is_known_item).
    # An earlier length-ratio heuristic let "Not All Attention Is All You Need" and "Is
    # Attention All You Need?" through as exact matches for the transformer paper.
    if query and title and identity.is_known_item(query, title):
        return "exact_title_match"
    if score >= TITLE_MATCH_STRONG:
        return "strong_title_match"
    if score >= TITLE_MATCH_PARTIAL:
        return "partial_title_match"
    return "topic_match_only"


# How long a search waits for slow sources before answering with what has arrived.
# Long enough for a healthy source (all answer in 1-3s when up), short enough that one
# throttled API cannot hold the whole answer hostage.
SEARCH_DEADLINE_SECONDS = 15.0


@dataclass
class _Candidate:
    """One paper as reported by one or more sources, merged before recording."""

    title: str
    abstract: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    sources: list[str] = field(default_factory=list)
    arxiv_paper: object | None = None      # the ArxivPaper, when arXiv returned it
    openalex_work: object | None = None    # the OpenAlexWork, when OpenAlex returned it
    s2_paper: object | None = None
    rank: int = 0                          # best position any source gave it

    def keys(self) -> set:
        # Title words IN ORDER. A word-set key merged "Is Attention All You Need?" into
        # "Attention Is All You Need", and thread timing decided which title survived.
        keys = {("title", identity.title_sequence(self.title))}
        if self.arxiv_id:
            keys.add(("arxiv", db.canonical_paper_id(self.arxiv_id)))
        if self.doi:
            keys.add(("doi", self.doi.lower()))
        return keys

    def absorb(self, other: "_Candidate") -> None:
        """Fills this candidate's gaps from another report of the same paper."""
        for attr in ("abstract", "year", "venue", "doi", "arxiv_id", "arxiv_paper", "openalex_work", "s2_paper"):
            if getattr(self, attr) is None and getattr(other, attr) is not None:
                setattr(self, attr, getattr(other, attr))
        if not self.authors:
            self.authors = other.authors
        if other.citation_count is not None:
            self.citation_count = max(self.citation_count or 0, other.citation_count)
        self.sources += [s for s in other.sources if s not in self.sources]
        self.rank = min(self.rank, other.rank)


def _from_arxiv(papers) -> list[_Candidate]:
    return [
        _Candidate(title=p.title, abstract=p.abstract, authors=p.authors,
                   year=int(p.published[:4]) if p.published else None,
                   arxiv_id=p.arxiv_id, sources=["arxiv"], arxiv_paper=p, rank=i)
        for i, p in enumerate(papers)
    ]


def _from_openalex(works, source: str = "openalex") -> list[_Candidate]:
    return [
        _Candidate(title=w.title, abstract=w.abstract, authors=w.authors, year=w.year,
                   venue=w.venue, citation_count=w.citation_count, doi=w.doi,
                   arxiv_id=w.arxiv_id, sources=[source], openalex_work=w, rank=i)
        for i, w in enumerate(works)
    ]


def _from_s2(papers) -> list[_Candidate]:
    out = []
    for i, p in enumerate(papers):
        ids = p.external_ids or {}
        out.append(_Candidate(title=p.title, abstract=p.abstract, authors=p.authors, year=p.year,
                              citation_count=p.citation_count, doi=ids.get("DOI"),
                              arxiv_id=ids.get("ArXiv"), sources=["semantic_scholar"], s2_paper=p, rank=i))
    return out


def _resolve_title_via_crossref(query: str) -> list[_Candidate]:
    """Crossref identifies the paper a misremembered title means; OpenAlex fills it in.

    Crossref's bibliographic matcher tolerates a wrong word that full-text search does
    not - it found the panel's paper from "error weighted drift ADOPTING ensemble
    framework" when OpenAlex's search did not have it in its top ten - but it rarely
    holds abstracts. So its DOI is used to fetch the full OpenAlex record.
    """
    out = []
    for i, work in enumerate(crossref_client.match_bibliographic(query, rows=5)):
        if identity.title_similarity(query, work.title) < TITLE_MATCH_PARTIAL:
            continue
        enriched = openalex_client.get_by_doi(work.doi)
        if enriched is not None:
            out.extend(_from_openalex([enriched], source="crossref"))
        else:
            out.append(_Candidate(title=work.title, authors=work.authors, year=work.year,
                                  venue=work.venue, citation_count=work.citation_count,
                                  doi=work.doi, sources=["crossref"]))
        out[-1].rank = i
        if len(out) >= 2:
            break
    return out


def _merge(candidates: list[_Candidate]) -> list[_Candidate]:
    """Collapses reports of the same paper from different sources into one.

    Matched on arXiv id, DOI, or identical title words - any one is enough. Without
    this the agent sees the transformer paper three times (arXiv, OpenAlex, S2) and
    spends three of its result slots, and its reader's attention, on one paper.
    """
    merged: list[_Candidate] = []
    for cand in candidates:
        home = next((m for m in merged if m.keys() & cand.keys()), None)
        if home is None:
            merged.append(cand)
        else:
            home.absorb(cand)
    return merged


def _record(cand: _Candidate) -> str | None:
    """Writes a merged candidate into the corpus and returns its id.

    arXiv's own record goes in first when there is one, since it is the authority on a
    preprint's text and date; the other sources then fill gaps (venue, DOI, citations)
    without overwriting it.
    """
    pid = None
    if cand.arxiv_paper is not None:
        pipeline.record_search_result(cand.arxiv_paper)
        pid = db.get_paper(cand.arxiv_paper.arxiv_id)["paper_id"]
    if cand.openalex_work is not None:
        pid = pipeline.record_openalex_work(cand.openalex_work) or pid
    if cand.s2_paper is not None:
        pid = pipeline.record_s2_paper(cand.s2_paper) or pid
    if pid is None:  # Crossref-only: no abstract and no OpenAlex record
        pid = pipeline.record_external(
            source="crossref", title=cand.title, abstract=None, authors=cand.authors,
            year=cand.year, doi=cand.doi, fallback_id=None, citation_count=cand.citation_count,
            venue=cand.venue,
        )
    return pid


def _search_papers(query: str, max_results: int, sort: str, collected_chunks: list[dict]) -> dict:
    """Federated search: every available source in parallel, merged, labelled, recorded.

    Previously this asked arXiv and then Semantic Scholar, one after the other. Both now
    fail routinely - arXiv throttles to one request per three seconds and has refused
    this machine outright for days, Semantic Scholar answers 429 to every search without
    a key - and when both did, the agent quietly answered from its local corpus. Now:

      arXiv       preprints, authoritative text
      OpenAlex    journals, conferences and preprints; abstracts, citations, venues
      Crossref    resolves a misremembered title to a DOI (title-shaped queries only)
      Semantic    used when S2_API_KEY is set; skipped, and said so, when it is not
      Scholar

    All run concurrently against one deadline, so the slowest source sets the latency
    only while it is healthy. Any that fails or times out is reported in
    `sources_unavailable`, and the answer is built from the rest.

    Fetching happens in worker threads; recording (database, embedding model, vector
    store) happens afterwards on this thread, so none of those are written concurrently.

    Each result carries a `match` label saying whether its title IS what was asked for
    (exact / strong) or merely shares its subject (partial / topic), so a paper about a
    title is never presented as the paper itself.
    """
    jobs = {
        "arxiv": lambda: _from_arxiv(arxiv_client.search(query, max_results=max_results, sort=sort, categories=None)),
        "openalex": lambda: _from_openalex(openalex_client.search(query, limit=max_results, sort=sort)),
    }
    if _looks_like_a_title(query):
        jobs["crossref"] = lambda: _resolve_title_via_crossref(query)
    if S2_API_KEY:
        jobs["semantic_scholar"] = lambda: _from_s2(semantic_scholar_client.search(query, limit=max_results))

    found: list[_Candidate] = []
    searched: list[str] = []
    unavailable: list[dict] = []
    if not S2_API_KEY:
        unavailable.append({"source": "semantic_scholar", "reason": "skipped: no S2_API_KEY configured"})

    pool = ThreadPoolExecutor(max_workers=len(jobs))
    futures = {pool.submit(fn): name for name, fn in jobs.items()}
    done, pending = wait(futures, timeout=SEARCH_DEADLINE_SECONDS)
    for fut in done:
        name = futures[fut]
        try:
            found.extend(fut.result())
            searched.append(name)
        except Exception as exc:  # noqa: BLE001 - one failed source must not lose the others
            logger.warning("%s search failed: %s", name, str(exc)[:160])
            unavailable.append({"source": name, "reason": str(exc)[:160]})
    for fut in pending:
        unavailable.append({"source": futures[fut], "reason": f"no answer within {SEARCH_DEADLINE_SECONDS:.0f}s"})
    # Do not wait for stragglers: they finish in the background and are discarded.
    pool.shutdown(wait=False, cancel_futures=True)

    merged = _merge(found)
    for cand in merged:
        cand.match = _match_label(_title_similarity(query, cand.title), query, cand.title)
    tier = {"exact_title_match": 0, "strong_title_match": 1, "partial_title_match": 2, "topic_match_only": 3}
    # Identity first, then papers several sources agree on, then each source's own order.
    merged.sort(key=lambda c: (tier[c.match], -len(c.sources), c.rank))
    merged = merged[: max_results * 2]

    results = []
    identified_ids = []
    for cand in merged:
        try:
            pid = _record(cand)
        except Exception as exc:  # noqa: BLE001 - one bad record must not lose the rest
            logger.info("could not record %r: %s", cand.title[:60], str(exc)[:120])
            pid = None
        entry = {
            "paper_id": pid or cand.doi or cand.title[:40],
            "title": cand.title,
            "match": cand.match,
            "sources": cand.sources,
            "year": cand.year,
        }
        for key, value in (("venue", cand.venue), ("citation_count", cand.citation_count),
                           ("doi", cand.doi), ("arxiv_id", cand.arxiv_id)):
            if value:
                entry[key] = value
        if cand.authors:
            entry["authors"] = ", ".join(cand.authors[:4]) + (" et al." if len(cand.authors) > 4 else "")
        if cand.abstract:
            entry["abstract"] = cand.abstract[:300]
        results.append(entry)
        if pid and cand.match in ("exact_title_match", "strong_title_match"):
            identified_ids.append(pid)

    # A paper whose title IS the query is guaranteed citable here, rather than left to a
    # later semantic re-query that can rank a paper ABOUT it higher.
    collected_chunks.extend(_mark_identified(_abstract_chunks(identified_ids)))

    out: dict = {
        "sources_searched": sorted(searched),
        "results": results,
        "best_title_match": results[0]["match"] if results else "none",
    }
    if unavailable:
        out["sources_unavailable"] = unavailable
    if not ({"openalex", "crossref", "semantic_scholar"} & set(searched)):
        out["coverage_warning"] = (
            "No source covering journal and conference papers answered (OpenAlex, "
            "Crossref, Semantic Scholar). Only preprints were searched, so do NOT tell "
            "the user a paper does not exist - a journal paper could not have been found."
        )
    if not results:
        out["result"] = "No papers matched this query from the sources that answered."
    return out


def _search_by_author(
    author: str,
    limit: int,
    year_from: int | None,
    year_to: int | None,
    collected_chunks: list[dict],
) -> dict:
    """A researcher's papers from OpenAlex and arXiv together, newest first.

    arXiv alone was the wrong source for this question. Most professors publish chiefly
    in journals and conferences, and the panel's own author is a clear case: OpenAlex
    resolves "Seba Susan" to Delhi Technological University with 171 works, almost none
    of which arXiv author search can see. OpenAlex is therefore primary; arXiv runs in
    parallel and adds any preprints, and both are merged by arXiv id, DOI and title.

    Which person was matched is reported back, because names collide and the agent must
    be able to say whose work it found rather than silently picking a namesake.

    The results are made citable directly, with the author list and date written INTO
    the passage text: the question is about authorship and dates, an abstract never
    names its own authors, and synthesis may only use what the passages contain.
    Answered from the list itself - running a semantic search for the question instead
    returned the transformer and RAG papers for "what has Bengio published recently".
    """
    matched = None

    def from_openalex():
        nonlocal matched
        matched = openalex_client.find_author(author)
        if matched is None:
            return []
        return _from_openalex(
            openalex_client.works_by_author(matched.openalex_id, limit=limit, year_from=year_from, year_to=year_to)
        )

    def from_arxiv():
        return _from_arxiv(arxiv_client.search_by_author(author, max_results=limit, year_from=year_from, year_to=year_to))

    found, searched, unavailable = [], [], []
    pool = ThreadPoolExecutor(max_workers=2)
    futures = {pool.submit(from_openalex): "openalex", pool.submit(from_arxiv): "arxiv"}
    done, pending = wait(futures, timeout=SEARCH_DEADLINE_SECONDS)
    for fut in done:
        try:
            found.extend(fut.result())
            searched.append(futures[fut])
        except Exception as exc:  # noqa: BLE001 - one failed source must not lose the other
            unavailable.append({"source": futures[fut], "reason": str(exc)[:160]})
    for fut in pending:
        unavailable.append({"source": futures[fut], "reason": f"no answer within {SEARCH_DEADLINE_SECONDS:.0f}s"})
    pool.shutdown(wait=False, cancel_futures=True)

    papers = sorted(_merge(found), key=lambda c: -(c.year or 0))[:limit]
    if not papers:
        return {"result": f"No papers found for author {author!r}.", "sources_searched": searched,
                **({"sources_unavailable": unavailable} if unavailable else {})}

    results = []
    for cand in papers:
        try:
            pid = _record(cand)
        except Exception as exc:  # noqa: BLE001
            logger.info("could not record %r: %s", cand.title[:60], str(exc)[:120])
            continue
        if not pid:
            continue
        venue = f"Venue: {cand.venue}\n" if cand.venue else ""
        body = cand.abstract or "(No abstract is available for this paper on record.)"
        collected_chunks.extend(_mark_identified([{
            "paper_id": pid,
            "chunk_id": f"{pid}::author_listing",
            "text": (
                f"{cand.title}\n"
                f"Authors: {', '.join(cand.authors) or 'unknown'}\n"
                f"Year: {cand.year or 'unknown'}\n"
                f"{venue}\n{body}"
            ),
        }]))
        entry = {"paper_id": pid, "title": cand.title, "year": cand.year, "sources": cand.sources}
        if cand.venue:
            entry["venue"] = cand.venue
        if cand.citation_count is not None:
            entry["citation_count"] = cand.citation_count
        results.append(entry)

    out: dict = {"sources_searched": sorted(searched), "results": results}
    if matched is not None:
        out["matched_author"] = {
            "name": matched.name,
            "institution": matched.institution,
            "total_works": matched.works_count,
        }
    if unavailable:
        out["sources_unavailable"] = unavailable
    return out


# Capped so a model that asks about "all the papers you just listed" cannot turn one
# tool call into a scorecard dump that crowds the retrieved passages out of the
# synthesis prompt's token budget.
METRICS_PAPER_LIMIT = 10

PASSAGES_PER_PAPER = 4
PASSAGE_PAPER_LIMIT = 6  # keeps the synthesis prompt inside free-tier token budgets
DEFAULT_PASSAGE_FOCUS = "methodology, results, and key contributions"


def _abstract_chunks(paper_ids) -> list[dict]:
    """Citable abstract passages for papers identified by id.

    Exists because a tool can know exactly which papers answer a question and still
    leave the synthesis step nothing it is allowed to quote. `find_related_papers`
    returns ids and strengths; `search_papers` returns listings. Synthesis is
    instructed to use only the retrieved passages, so without this the agent
    identifies the right papers and then reports that it cannot discuss them.

    Reads from the corpus rather than taking text from the caller: every tool that
    needs this has already recorded its papers (`search_and_record`) or found them in
    the citation graph, so the abstract is in the database by the time we get here,
    and going back to it keeps one definition of what an abstract chunk looks like.

    Matched on `canonical_id` as well as `paper_id`, because the graph is keyed
    canonically while search results carry versioned ids.

    The title is prepended for the same reason it is indexed that way - an abstract
    never contains its own title, and the question is frequently about the title.
    """
    ids = [pid for pid in paper_ids if pid]
    if not ids:
        return []

    placeholders = ",".join("?" * len(ids))
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT p.paper_id, COALESCE(p.canonical_id, p.paper_id) AS cid, p.title, "
            f"       p.abstract, c.text AS indexed_text "
            f"FROM papers p LEFT JOIN chunks c ON c.chunk_id = p.paper_id || '::abstract' "
            f"WHERE p.paper_id IN ({placeholders}) OR p.canonical_id IN ({placeholders}) "
            f"ORDER BY p.paper_id",
            ids + ids,
        ).fetchall()

    # One passage per CANONICAL paper. Matching on canonical_id deliberately finds a
    # versioned row from a bare id, but it also finds both rows when both are indexed -
    # and handing synthesis the same abstract twice under two ids spends the passage
    # budget on a duplicate and invites the answer to cite one paper as if it were two.
    # This is the same duplicate-counting problem that inflated nDCG above 1.0.
    seen: set[str] = set()
    chunks = []
    for row in rows:
        # The indexed chunk text is preferred over rebuilding it: for a paper with no
        # abstract it is the title-only record (title, authors, and an explicit "no
        # abstract on record" note). Skipping abstractless papers here meant the panel's
        # own paper - correctly identified by title - reached synthesis with nothing
        # citable attached.
        text = row["indexed_text"] or (
            f"{row['title']}\n\n{row['abstract']}" if row["abstract"] else None
        )
        if not text or row["cid"] in seen:
            continue
        seen.add(row["cid"])
        chunks.append({
            "paper_id": row["paper_id"],
            "chunk_id": f"{row['paper_id']}::abstract",
            "text": text,
        })
    return chunks


def _paper_metrics(paper_ids: list[str]) -> list[dict]:
    """Scorecards for named papers, as facts the synthesis step can quote.

    Only fields that were actually computed are returned. Sending `"cnci": null` would
    invite the model to render it as "0x the field median", which is a fabrication -
    the real meaning is "this paper's cohort was too small to trust a median", and the
    honest way to convey that is to omit the key and say so in `note`.
    """
    if not paper_ids:
        return [{"error": "no paper_ids given"}]

    wanted = paper_ids[:METRICS_PAPER_LIMIT]
    cards = scorecard.get_many(wanted)
    out: list[dict] = []
    for paper_id in wanted:
        card = cards.get(paper_id) or scorecard.get(paper_id)
        if card is None:
            out.append({"paper_id": paper_id, "error": "not in the corpus"})
            continue

        entry: dict = {"paper_id": card.paper_id, "title": card.title, "summary": card.summary}
        for key, value in (
            ("citations", card.citation_count),
            ("substantive_citations", card.influential_citation_count),
            ("field_normalized_impact", None if card.cnci is None else round(card.cnci, 1)),
            ("centrality_percentile", card.pagerank_percentile),
            ("citing_papers_in_corpus", card.in_degree),
            ("venue", card.venue),
            ("year", card.cohort_year),
        ):
            if value is not None:
                entry[key] = value
        if card.cnci_note:
            entry["note"] = "field-normalized impact not computable: " + card.cnci_note
        out.append(entry)
    return out


def _paper_passages(paper_ids: list[str], focus: str | None, collected_chunks: list[dict]) -> dict:
    """Relevant passages from each named paper, retrieved per paper rather than pooled.

    Mirrors the retrieval half of `llm/compare.py` deliberately, but stops before its
    synthesis call: the agent already has a synthesis step of its own, and nesting a
    second LLM call inside the tool loop would spend a strong-tier call to produce text
    the real synthesis then has to reconcile with everything else it gathered.

    Retrieving per paper is the point. A single pooled search lets one long or densely
    worded paper supply every passage, so a comparison silently becomes a summary of
    that paper alone.
    """
    if not paper_ids:
        return {"error": "no paper_ids given"}

    query = focus or DEFAULT_PASSAGE_FOCUS
    out = []
    for paper_id in paper_ids[:PASSAGE_PAPER_LIMIT]:
        paper = db.get_paper(paper_id)
        if paper is None:
            out.append({"paper_id": paper_id, "error": "not in the corpus - search for it first"})
            continue

        # Full text is pulled in first, lazily and cached. Without it this tool can
        # only ever see the abstract, and the questions it exists to serve - "compare
        # their evaluation setups", "what does this paper say about X" - are precisely
        # the ones an abstract does not answer. Measured: BERT had exactly one chunk
        # indexed, so the tool returned its abstract for a question about evaluation,
        # the orchestrator judged that unhelpful and called the same tool three more
        # times, exhausting its iteration budget.
        pipeline.ingest_full_text(paper["paper_id"])
        full_text = db.has_full_text(paper["paper_id"])
        chunks = retrieval.search(query, top_k=PASSAGES_PER_PAPER, paper_id=paper["paper_id"])
        collected_chunks.extend(_mark_identified(chunks))
        entry = {
            "paper_id": paper_id,
            "title": paper["title"],
            "full_text_read": full_text,
            "passages": [c["text"][:400] for c in chunks] or ["(no passage matched this focus)"],
        }
        # Said explicitly, because an answer drawn from an abstract reads exactly like one
        # drawn from the paper, and they support very different claims. Many journal
        # papers have no openly available PDF; that is a fact about the paper to report,
        # not a gap to paper over.
        if not full_text:
            entry["note"] = (
                "Only the abstract (or title) was available - no open-access full text "
                "exists for this paper. Say so, and do not present details as if read "
                "from the paper body."
            )
        out.append(entry)
    return {"focus": query, "papers": out}


def _run_tool(name: str, args: dict, collected_chunks: list[dict]) -> dict | list[dict]:
    if name == "search_papers":
        return _search_papers(args["query"], args.get("max_results", 5), args.get("sort", "relevance"), collected_chunks)

    if name == "query_index":
        # The learned ranker rather than a fixed citation weight. See
        # rerank.search_ranked for the measurement that forced this: no single weight
        # ranks both a freshly-discovered paper and a famous one correctly, because a
        # fixed weight cannot condition on which kind of paper the query is asking for.
        hits = rerank.search_ranked(args["query"], top_k=args.get("top_k", 5))
        collected_chunks.extend(hits)
        return [{"paper_id": h["paper_id"], "text": h["text"][:400]} for h in hits]

    if name == "get_paper_metrics":
        return _paper_metrics(args.get("paper_ids") or [])

    if name == "search_by_author":
        return _search_by_author(
            args["author"],
            limit=args.get("max_results", 8),
            year_from=args.get("year_from"),
            year_to=args.get("year_to"),
            collected_chunks=collected_chunks,
        )

    if name == "find_related_papers":
        results, method = related.find_best_effort(args["paper_id"], limit=args.get("limit", 8))
        if not results:
            return {
                "result": "No citation links for this paper inside the corpus - it is "
                          "cited by nothing here and its own references are not indexed. "
                          "This is a gap in coverage, not evidence that no related work exists."
            }
        # The method is returned, not hidden, because the two answer different
        # questions and an answer that conflates them is misleading: co-citation is
        # "the field groups these together", coupling is "these were built on the
        # same prior work", and only the first implies the field has judged anything.
        # Abstracts are attached so the related papers are citable. Without this the
        # tool returns ids and titles, nothing reaches collected_chunks, and the
        # answer is ungrounded - the grader then marks the turn irrelevant and the
        # retry runs a similarity search on the raw question. Measured: "what should
        # I read alongside 1706.03762?" returned a correct co-citation list of BERT,
        # Adam and GPT-3, then answered with Chinese reading-comprehension datasets,
        # because the phrase "what should I read alongside" is textually nearest to
        # dataset papers and nothing else was in the prompt to contradict it.
        collected_chunks.extend(_mark_identified(_abstract_chunks([r.paper_id for r in results])))

        return {
            "method": method,
            "explanation": (
                "papers frequently cited alongside this one; strength = number of papers citing both"
                if method == related.CO_CITATION
                else "papers sharing references with this one; strength = number of shared references"
            ),
            "papers": [
                {"paper_id": r.paper_id, "title": r.title, "strength": r.strength} for r in results
            ],
        }

    if name == "get_paper_passages":
        return _paper_passages(args.get("paper_ids") or [], args.get("focus"), collected_chunks)

    if name == "ingest_full_text":
        chunk_count = pipeline.ingest_full_text(args["paper_id"])
        return {"chunks_added": chunk_count}

    return {"error": f"unknown tool: {name}"}


# Synthesis runs against a free tier with a hard 8,000 tokens-per-minute cap on the
# whole request, and exceeding it is not a degraded answer - it is a 413 that the user
# sees as "No LLM provider available". So the passage block gets an explicit character
# budget rather than a per-chunk limit that silently scales with however many chunks
# the tools happened to gather.
#
# A fixed 800-chars-per-chunk was fine while everything came from abstracts, and broke
# as soon as get_paper_passages began ingesting full text: two papers at four body
# chunks each, plus a query_index call or two, put the request at 8,220 tokens.
#
# ~4 chars per token is the usual English rule of thumb; 11,000 characters is roughly
# 2,750 tokens, which leaves comfortable room for the system prompt, the conversation
# so far, the scorecards and the answer itself inside 8,000.
SYNTHESIS_CHAR_BUDGET = 11_000
SYNTHESIS_MIN_CHUNK_CHARS = 280  # below this a passage is too clipped to cite honestly
SYNTHESIS_MAX_CHUNK_CHARS = 800


def _fit_passages(chunks: list[dict]) -> list[str]:
    """Renders as many passages as the budget allows, best-ranked first.

    Chunks arrive in relevance order, so truncating the LIST is preferable to
    truncating every passage further: eight usable passages beat twenty clipped to
    280 characters each, where the clipping would cut mid-sentence and leave claims
    unsupportable. Per-chunk size is reduced first, down to a floor, and only then are
    trailing chunks dropped.
    """
    if not chunks:
        return []

    per_chunk = max(
        SYNTHESIS_MIN_CHUNK_CHARS,
        min(SYNTHESIS_MAX_CHUNK_CHARS, SYNTHESIS_CHAR_BUDGET // len(chunks)),
    )

    rendered, used = [], 0
    for chunk in chunks:
        text = chunk["text"][:per_chunk]
        entry = f"[{chunk['paper_id']}]: {text}"
        if used + len(entry) > SYNTHESIS_CHAR_BUDGET and rendered:
            break
        rendered.append(entry)
        used += len(entry)

    if len(rendered) < len(chunks):
        logger.info("synthesis budget: kept %d of %d passages", len(rendered), len(chunks))
    return rendered


def _build_synthesis_prompt(history: list[dict], chunks: list[dict]) -> str:
    prior_turns = history[:-1]
    latest_question = history[-1]["content"]

    parts = []
    if prior_turns:
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in prior_turns)
        parts.append(f"Conversation so far:\n{convo}")
    parts.append(f"Latest question: {latest_question}")

    passages = _fit_passages(chunks)
    if passages:
        parts.append("Retrieved passages:\n\n" + "\n\n".join(passages))
        # Scorecards are limited to the papers whose passages actually survived the
        # budget. Listing impact figures for a paper the model cannot see a passage
        # from invites it to write about that paper anyway, citing metrics as though
        # they were evidence of relevance.
        kept_ids = {c["paper_id"] for c in chunks[:len(passages)]}
        scorecards = _scorecard_context([c for c in chunks if c["paper_id"] in kept_ids])
        if scorecards:
            parts.append(f"Measured impact of these papers:\n\n{scorecards}")
    else:
        parts.append("(No new passages were retrieved for this question - answer from the conversation above only.)")

    return "\n\n".join(parts)


def _scorecard_context(chunks: list[dict]) -> str:
    """Impact metrics for every retrieved paper, attached to the synthesis prompt
    unconditionally.

    This is deliberately NOT left to the get_paper_metrics tool. Asked "how
    influential is the transformer paper compared to BERT", the cheap orchestrator
    model skipped the tool and answered that the passages contained no citation
    information and the comparison could not be made - while the numbers sat in a
    local table one query away. The same reasoning already applied to merging the two
    search sources into one tool: where a correct answer can be guaranteed by
    construction, it should not depend on a small model noticing an instruction in a
    long prompt.

    A paper's abstract never states its own citation count, so this is the only route
    by which a grounded, no-outside-knowledge answer can say anything quantitative
    about impact at all. The lookup is local and costs nothing.
    """
    paper_ids = list(dict.fromkeys(chunk["paper_id"] for chunk in chunks))
    cards = scorecard.get_many(paper_ids)
    lines = [f"[{pid}]: {cards[pid].summary}" for pid in paper_ids if pid in cards]
    return "\n".join(lines)


def _gather_chunks(history: list[dict]):
    """Runs the tool-calling orchestration loop and returns the deduped chunks it
    retrieved. Shared by the blocking and streaming answer paths - only the
    synthesis call after this differs between them.

    Written as a GENERATOR: it yields a human-readable description of each step as it
    happens and returns the gathered chunks at the end, so a caller streaming to a UI
    can show real progress instead of a spinner. Drive it with next() and read the
    result from StopIteration.value.
    """

    messages = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}, *history]
    collected_chunks: list[dict] = []
    any_tool_called = False
    tools_used: set[str] = set()
    seen_calls: set[tuple[str, str]] = set()

    for _ in range(MAX_TOOL_ITERATIONS):
        completion = router.generate(messages, tools=TOOL_SCHEMAS, tier="cheap")
        message = completion["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            break
        any_tool_called = True

        # Echo back only the fields the chat-completions format actually needs.
        # completion.model_dump() includes extra provider-specific fields
        # (annotations, refusal, audio, ...) that aren't valid to send back as
        # input on some providers (Groq rejects an echoed 'annotations' field).
        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})
        for call in tool_calls:
            raw_args = call["function"]["arguments"] or "{}"
            args = json.loads(raw_args)
            tool_name = call["function"]["name"]
            tools_used.add(tool_name)

            # A repeat of an identical call is answered from the record rather than
            # re-executed. Small models loop when a tool returns something they judge
            # unhelpful, re-asking the same question and expecting a different answer;
            # observed here as four consecutive get_paper_passages calls with the same
            # arguments, which spent the whole iteration budget and, since that tool
            # now downloads PDFs, would re-fetch them each time. Telling the model it
            # already has the result is what actually breaks the loop - silently
            # returning the cached value again just feeds it the input it looped on.
            signature = (tool_name, json.dumps(args, sort_keys=True))
            if signature not in seen_calls:
                yield _activity_label(tool_name, args)
            if signature in seen_calls:
                logger.info("tool call: %s (repeat suppressed)", tool_name)
                result = {
                    "note": f"{tool_name} was already called with these exact arguments. "
                            "Use the earlier result - calling again returns the same thing. "
                            "Either try different arguments or stop and synthesize."
                }
            else:
                seen_calls.add(signature)
                result = _execute_tool(tool_name, args, collected_chunks)

            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})

    # Safety net: query_index is a free local call (no API cost, no rate limit), so if
    # the orchestrator called a tool at all but finished without ever retrieving any
    # passages - e.g. it called search_papers to discover new papers but (as cheap-tier
    # models sometimes do) forgot the required query_index follow-up - try once
    # automatically rather than handing synthesis an empty context it can't do anything
    # with. Gated on any_tool_called: if the orchestrator made zero tool calls, that's a
    # deliberate judgment call (a plain greeting, a follow-up the conversation already
    # answers) - forcing a local-index lookup there would surface irrelevant papers as
    # "Sources" under an answer that was never grounded in the corpus to begin with.
    if any_tool_called and not collected_chunks:
        collected_chunks.extend(rerank.search_ranked(history[-1]["content"], top_k=5))

    unique_chunks = _dedupe(_identified_first(collected_chunks))

    # Corrective pass (Yan et al., 2024, arXiv:2401.15884). The orchestrator has no
    # view of whether what it gathered actually answers the question - it stops when
    # it believes it is done, which is not the same thing. Grading the result catches
    # the common failure the old empty-chunks fallback could not: plenty of passages
    # retrieved, none of them relevant.
    #
    # Only runs when tools were called. A deliberate no-tool turn (a greeting, a
    # follow-up the conversation already answers) has nothing to grade, and grading it
    # would only invent a reason to attach irrelevant sources to a conversational reply.
    # Skipped when an exact-lookup tool supplied the chunks. Grading an author lookup
    # measures the wrong thing and then acts on it: asked "what has Bengio published
    # recently", search_by_author returned his eight most recent papers and the grader
    # marked them irrelevant because "passages do not mention Yoshua Bengio" - which is
    # true, since an abstract does not list its own authors, and entirely beside the
    # point. The retry then ran a similarity search for the question itself, which
    # returned the transformer and RAG papers, and because retry results are placed
    # FIRST they evicted the correct answer from the synthesis prompt.
    #
    # Corrective RAG (Yan et al. 2024) corrects retrieval that may have missed. A
    # lookup by author name or paper id cannot have missed in that sense, so there is
    # nothing for it to correct and every opportunity for it to do harm.
    if any_tool_called and not (tools_used & EXACT_LOOKUP_TOOLS):
        question = history[-1]["content"]
        yield "Checking whether the results actually answer the question"
        grade = grading.grade_retrieval(question, unique_chunks)
        logger.info("Retrieval graded %s (%s)", grade.grade, grade.reason)

        if grade.needs_retry:
            # Retry with a DIFFERENT strategy rather than repeating what just failed:
            # a plain local re-query would return the same passages that were judged
            # irrelevant a moment ago. Lexical fusion is the cheapest genuinely
            # different signal available, and Phase 1 measured it as strongest on
            # exactly the specific-paper queries most likely to have missed.
            yield "First results looked off \u2014 retrying with a different search strategy"
            logger.info("Retrying retrieval with post-rerank lexical fusion")
            # Lexical fusion, deliberately WITHOUT the citation blend. The point of a
            # retry is to bring a genuinely different signal to bear, and BM25 is that
            # signal; stacking the fixed citation weight on top would reintroduce the
            # bias against recent and obscure papers that the learned ranker was
            # adopted to remove - on precisely the queries already judged a miss,
            # which skew toward papers the corpus knows least about.
            retry_chunks = retrieval.search(
                question,
                top_k=RETRY_TOP_K,
                fetch_k=CANDIDATE_POOL,
                fusion="post",
            )
            # Retry first, rejected originals after: synthesis truncates its passage
            # list, so ordering decides what actually reaches the prompt.
            unique_chunks = _dedupe(retry_chunks + unique_chunks)

    return unique_chunks


def _mark_identified(chunks: list[dict]) -> list[dict]:
    """Flags chunks that came from an exact-identity lookup."""
    for chunk in chunks:
        chunk[PRIORITY_KEY] = True
    return chunks


def _identified_first(chunks: list[dict]) -> list[dict]:
    """Stable sort putting exact-identity passages ahead of broad-search ones.

    Necessary because the synthesis prompt has a character budget and renders passages
    in order, so whatever sits at the front is what the answer is written from.

    The failure that forced this: asked for a specific paper by a garbled title, the
    agent located it correctly (calling get_paper_passages on the right id) but its
    broad arXiv fallback had already collected five unrelated papers - hexaquark decay,
    traffic models, graph recolouring - because "error weighted drift" matches those
    words scattered across physics abstracts. Those arrived first, filled the front of
    the prompt, and the answer discussed them while reporting that the requested paper
    could not be found. The right paper was in the prompt the whole time, at the back.

    Same principle as placing corrective-retry results first: with a truncating budget,
    ordering IS relevance weighting.
    """
    return sorted(chunks, key=lambda c: not c.get(PRIORITY_KEY, False))


def _dedupe(chunks: list[dict]) -> list[dict]:
    """Drops repeats, keeping first (best-ranked) occurrence - the same passage can
    surface across several query_index calls."""
    seen_ids = set()
    unique = []
    for chunk in chunks:
        if chunk["chunk_id"] not in seen_ids:
            seen_ids.add(chunk["chunk_id"])
            unique.append(chunk)
    return unique


def _cited_or_all(answer_text: str, chunks: list[dict]) -> list[dict]:
    # Retrieval often surfaces more chunks than end up cited (some are close
    # matches but not actually used in the answer). Report only the sources
    # the model actually cited - a "Sources" list that includes unused
    # passages would undercut the point of citation-grounding.
    #
    # Match bare arXiv ids too, not just [bracketed] ones: the model is asked
    # to always bracket citations but doesn't reliably do so inside markdown
    # tables in practice - relying purely on prompt compliance here would
    # make "cited" silently fall back to "everything retrieved" whenever the
    # model's formatting slips, which defeats the point of filtering at all.
    #
    # Compared on canonical ids and across every id form. Matching exact strings on
    # arXiv ids only meant an answer citing "1810.04805" did not match a passage stored
    # under "1810.04805v2", and a citation of a journal paper by its OpenAlex id did not
    # match at all - either way the filter found nothing and fell back to listing every
    # retrieved paper as a source, cited or not.
    #
    # Returned in order of first citation, so the Sources list reads in the same order
    # as the numbered citations in the answer.
    order: dict[str, int] = {}
    for match in re.findall(PAPER_ID_PATTERN, answer_text):
        order.setdefault(db.canonical_paper_id(match), len(order))
    cited_chunks = [c for c in chunks if db.canonical_paper_id(c["paper_id"]) in order]
    cited_chunks.sort(key=lambda c: order[db.canonical_paper_id(c["paper_id"])])
    return cited_chunks or chunks


def answer_query_stream(history: list[dict]):
    """history: [{"role": "user"|"assistant", "content": str}, ...], chronological,
    must end with a user message (the question being answered now).

    Yields ("activity", str) while the agent is choosing and running tools, then
    ("token", str) as the answer is written, then exactly one final
    ("done", list[dict] sources)."""
    if not history or history[-1]["role"] != "user":
        raise ValueError("history must be non-empty and end with a user message")

    # `_gather_chunks` is a generator: it yields a description of each step as it
    # runs and returns the chunks when done, so progress streams out of here without
    # a thread or a queue. StopIteration.value carries the return.
    #
    # Structure matters for correctness, not just tidiness. Running the orchestration
    # on a worker thread would swallow its exceptions - NoProviderAvailable raised
    # inside a tool loop would kill the thread silently, leave the chunk list empty,
    # and never reach the route's error handler, so a rate-limited request would
    # render as a confident answer with no sources instead of an error.
    gather = _gather_chunks(history)
    try:
        while True:
            yield "activity", next(gather)
    except StopIteration as finished:
        unique_chunks = finished.value or []

    synthesis_messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": _build_synthesis_prompt(history, unique_chunks)},
    ]

    answer_text = ""
    for delta in router.generate_stream(synthesis_messages, tier="strong"):
        answer_text += delta
        yield "token", delta

    yield "done", _cited_or_all(answer_text, unique_chunks)
