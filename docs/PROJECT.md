# Scientometric Paper Agent — Project Specification

**A citation-aware research assistant for academics.**

This document specifies the project: the idea, the problem it addresses, the planned
architecture, every technique with its source paper, the scientometric methodology in
detail, the phase plan, and the evaluation design.

---

## 1. The Idea

### One sentence

A research assistant that finds and explains academic papers, ranking them by
**citation evidence** rather than text similarity alone, and showing the researcher
*why* each paper matters with concrete quantitative indicators.

### The audience

University researchers and professors doing literature work: locating a specific
paper, understanding a topic, tracking what is new in a field, comparing approaches.

### What makes it different

Conventional academic search ranks papers by how closely their text matches the
query. That is one signal, and it is blind to everything else the scholarly record
knows about a paper — how often it is cited, whether it is foundational or derivative,
where it sits in the citation network, and whether the field is currently paying
attention to it.

This project adds that second signal and, critically, **learns when to apply it**.

### The analogy

In 1998, every search engine ranked web pages by the words on the page. Google won by
reading the **links between pages** — PageRank. They were not better at understanding
text; they used a structural signal everyone else ignored.

Citations are the link structure of science. Conventional paper search is AltaVista.
This project adds the PageRank.

---

## 2. The Problem

### The failure mode being targeted

Consider a researcher asking for *"the original transformer paper."*

A text-similarity system can return the wrong paper, because a recent survey whose
abstract contains the sentence *"the Transformer introduced by Vaswani et al."*
matches those query words at least as well as the actual paper does — whose abstract
never contains the phrase "Attention Is All You Need" at all.

On text alone, these two papers are near-indistinguishable. On citation evidence they
are not remotely comparable: the foundational paper carries on the order of 10⁵
citations and sits at the centre of the field's citation network, while the survey
carries a few hundred and sits at the periphery.

**Text similarity tells you what a paper is about. The citation network tells you what
a paper is.**

### Why this is a ranking problem, not a retrieval problem

The hypothesis this project tests is that in cases like the above, the correct paper
*is* retrieved — it simply ranks below several topically-similar competitors. If that
holds, then the intervention required is a better **ranking** signal, not better
recall. This distinction is important because it determines whether citation evidence
can help at all: no reranking method can promote a paper that was never retrieved.

Establishing whether this holds is the first measurement the project will make.

---

## 3. Scope and Constraints

### Hard constraints

| Constraint | Consequence |
|---|---|
| **Zero budget** | Free-tier LLM APIs only (Groq, Gemini, OpenRouter). Local embedding models. No paid bibliographic databases. |
| **No guaranteed GPU** | Everything must run on CPU. Rules out training large models. |
| **Single user, local-first** | No authentication, SQLite rather than a database server. |
| **Built from scratch** | Not a wrapper around an existing commercial product. |

### Domain scope: deep learning only

The corpus is restricted to arXiv categories `cs.LG`, `cs.CL`, `cs.CV`, `cs.AI`,
`stat.ML`.

**This is a methodological requirement, not an arbitrary narrowing.** Field-normalized
citation impact — the central scientometric instrument in this project — compares a
paper against a peer group. "The typical citation count for a 2023 cs.LG paper" is a
meaningful quantity computable from several hundred papers. "The typical citation
count for a 2023 paper" spans fields whose citation cultures differ by an order of
magnitude and therefore normalizes against nothing.

The scope decision is what makes the core technique valid.

---

## 4. System Architecture

```
                          USER QUERY
                              |
                    +---------v---------+
                    |  AGENT ORCHESTRA  |   ReAct loop, cheap LLM tier
                    +---------+---------+
                              |
        +---------------------+---------------------+
        |                     |                     |
   +----v-----+        +------v------+       +------v------+
   | DISCOVER |        |  RETRIEVE   |       |   VERIFY    |
   | arXiv +  |        | hybrid      |       | grade +     |
   | S2 live  |        | dense/BM25  |       | groundedness|
   +----+-----+        +------+------+       +------+------+
        |                     |                     |
        |              +------v------+              |
        |              |   RANKER    |  LambdaMART  |
        |              | text + cite |              |
        |              +------+------+              |
        |                     |                     |
        +---------------------+---------------------+
                              |
                    +---------v---------+
                    |    SYNTHESIS      |  strong LLM tier
                    | cited answer      |
                    +-------------------+

  DATA LAYER
  +------------------+  +------------------+  +------------------+
  |  SQLite          |  |  ChromaDB        |  |  Citation graph  |
  |  papers, chunks  |  |  vectors (live + |  |  networkx over   |
  |  metrics, cohorts|  |  frozen eval)    |  |  stored edges    |
  +------------------+  +------------------+  +------------------+
```

### Two-tier LLM usage

| Tier | Used for | Rationale |
|---|---|---|
| **Cheap** | Tool selection, intent classification, retrieval grading | These are most of an agentic loop's calls, and choosing a tool does not require a strong model. |
| **Strong** | Final answer synthesis | One call per query. The model writing the answer should not be occupied with tool bookkeeping. |

### Two-tier ingestion

| Tier | What it does | When |
|---|---|---|
| **Tier 1** | Metadata + abstract embedding | Every search result. Cheap, no PDF download. |
| **Tier 2** | PDF download, parse, chunk, embed with page numbers | Lazy — only for papers the user opens or compares. Cached. |

---

## 5. The Corpus

### Construction — staged and resumable

| Stage | Function |
|---|---|
| **seed** | Resolve ~40 landmark paper titles via arXiv title search, with string-similarity verification of the returned title |
| **arxiv** | Walk every (category, year) cohort, 2017–2026, storing metadata |
| **metrics** | Fetch citation counts, influential citations, venue from Semantic Scholar via its batch endpoint |
| **graph** | Fetch reference lists in batches; store edges where both endpoints are corpus papers |
| **cohorts** | Compute per-(category, year) citation medians; derive CNCI, PageRank, in-degree |
| **index** | Embed `title + abstract` into the vector store |
| **snapshot** | Freeze a copy of the vector store for evaluation |

### Why cohort-shaped sampling

The crawl samples a fixed number of papers per (category, year) pair rather than
"latest N in category." Field normalization compares a paper to others of its **own
category and year**, so the corpus must be constructed with that grouping in mind.
Crawling newest-first would accumulate recent papers and leave earlier years too thin
to yield a stable median.

### Why a seed list is also required

Cohort sampling draws a few hundred papers from years that published tens of
thousands, so it captures any individual famous paper only by chance. Two problems
follow: a deep-learning corpus missing ResNet is not credible, and an evaluation query
asking for a paper the corpus does not contain is unanswerable by construction — it
would measure the gap in the crawl rather than the quality of the system.

**Cohort sampling provides statistical depth; the seed list provides landmark
coverage.** Both are necessary.

### Target scale

Approximately 8,000–10,000 papers with citation metrics, a citation graph over those
papers, and cohort statistics across ten years and five categories.

---

## 6. Techniques — Retrieval Layer

### 6.1 Dense retrieval (semantic search)

Each paper's `title + abstract` is encoded as a vector. Queries are encoded the same
way; nearest vectors are returned.

- **Model:** `BAAI/bge-small-en-v1.5`, with a planned comparison against `bge-base`
  and SPECTER
- **Store:** ChromaDB (embedded, HNSW index internally)
- **Source:** Dense Passage Retrieval — Karpukhin et al., 2020, **arXiv:2004.04906**
- **Embedding model:** BGE / C-Pack — Xiao et al., 2023, **arXiv:2309.07597**

**Design point — index title together with abstract.** A paper's title is its most
identity-bearing text. Indexing the abstract alone means a query naming a paper cannot
match it, since a title frequently does not appear in its own abstract. Pairing title
with abstract is standard practice in scholarly retrieval and is what SPECTER does.

### 6.2 Lexical retrieval (BM25)

BM25 scores documents by term overlap, weighting rare terms more heavily and
normalizing for document length. It matches *literal words* where dense retrieval
matches *meaning*.

- **Library:** `rank_bm25` (Okapi BM25)
- **Source:** Robertson & Zaragoza, 2009, *The Probabilistic Relevance Framework:
  BM25 and Beyond*, Foundations and Trends in IR 3(4)

**Design point — no stemming or stopword removal.** Academic titles depend on function
words far more than general prose does; "Attention Is All You Need" is almost entirely
stopwords. Removing them would leave "attention" and destroy the exact-phrase signal
that motivates including BM25 at all.

### 6.3 Reciprocal Rank Fusion

Merges two ranked lists by position rather than by score:

```
RRF(d) = Σ_i  1 / (k + rank_i(d))          k = 60
```

- **Source:** Cormack, Clarke & Büttcher, 2009, SIGIR

**Why rank-based fusion.** A BM25 score and a cosine distance live on incomparable
scales, and any weighted sum of them requires calibration that shifts with corpus and
query. Using only positions removes the problem entirely.

### 6.4 Cross-encoder reranking

A model that consumes (query, document) **jointly** and emits a single relevance
score. More accurate than comparing independently-computed embeddings because it can
attend across both texts; too slow to apply to a whole corpus, so it reranks a
shortlist.

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Source:** Nogueira & Cho, 2019, *Passage Re-ranking with BERT*, **arXiv:1901.04085**

### 6.5 Where fusion is applied — an open design question

BM25 can be fused into the pipeline either **before** the cross-encoder rerank (as
candidate generation) or **after** it (as a final reordering). These are not
equivalent, because the reranker re-scores every candidate on its own terms and may
discard an ordering established upstream.

The project will measure both placements, and measure them **per query intent** rather
than only in aggregate — the expectation being that lexical matching helps queries that
name a paper and hurts queries phrased in vocabulary the paper does not use. If that
expectation holds, BM25 belongs as a conditional signal rather than a fixed stage.

### 6.6 Additional techniques to evaluate

| Technique | Source | Purpose |
|---|---|---|
| RAPTOR — recursive tree summarization | Sarthi et al., 2024, **arXiv:2401.18059** | Retrieve at multiple abstraction levels; suits papers' natural hierarchy |
| HyDE — hypothetical document embeddings | Gao et al., 2022, **arXiv:2212.10496** | Generate a hypothetical answer, search with that instead of the question |
| ColBERTv2 — late interaction | Santhanam et al., 2021, **arXiv:2112.01488** | Token-level multi-vector matching |
| Contextual retrieval | Anthropic engineering, 2024 (blog, not peer-reviewed) | Prepend chunk-specific context before embedding |
| SPECTER | Cohan et al., 2020, **arXiv:2004.07180** | Paper embeddings trained on citation links |
| SciNCL | Ostendorff et al., 2022, **arXiv:2202.06671** | Neighborhood contrastive scholarly embeddings |

---

## 7. Techniques — Agentic Layer

### 7.1 Base RAG pattern

Retrieve relevant passages, then generate an answer grounded in them.

- **Source:** Lewis et al., 2020, **arXiv:2005.11401**
- **Related:** REALM — Guu et al., 2020, **arXiv:2002.08909**

### 7.2 ReAct — the tool-calling loop

The agent interleaves reasoning and acting: choose a tool, call it, read the result,
choose again, up to a bounded number of iterations, then synthesize.

- **Source:** Yao et al., 2022, **arXiv:2210.03629**

**Tools:**

| Tool | Purpose |
|---|---|
| `search_papers` | Query arXiv and Semantic Scholar live; ingest results |
| `query_index` | Semantic search over already-ingested chunks |
| `ingest_full_text` | Download and index a paper's PDF (expensive, lazy, cached) |
| `get_paper_metrics` | Read a paper's scientometric indicators so the agent can reason about them in prose |

### 7.3 Corrective RAG — retrieval grading

After retrieval, a cheap model grades whether the passages actually answer the
question (`sufficient` / `partial` / `irrelevant`). On a clear miss, the system retries
with a **different** strategy rather than repeating the one that just failed.

- **Source:** Yan et al., 2024, **arXiv:2401.15884**

**Design point.** A naive fallback that fires only when retrieval returns *nothing*
cannot catch the far more common failure: many passages retrieved, none relevant.
Grading makes correctness a measurable property of the system rather than a property of
prompt wording.

### 7.4 Self-RAG — groundedness verification

Before returning an answer, verify that each factual claim traces to a retrieved
passage.

- **Source:** Asai et al., 2023, **arXiv:2310.11511**

**Why this exceeds citation-format checking.** A regex over paper identifiers verifies
*format*, not truth. An answer stating "achieves 94.2% accuracy and was trained on 512
TPUs [2005.11401]" passes a format check even when neither figure appears anywhere in
the cited paper. In a research tool a confident wrong citation is worse than a missing
one, and only groundedness verification detects it.

**Design point.** The verifier will be **advisory rather than enforcing** — surfacing
doubt instead of silently suppressing text — because the verifier is itself a
cheap-tier model and will sometimes be wrong.

### 7.5 Reflexion — self-correction

Verbal self-reflection retained across attempts, informing the retry.

- **Source:** Shinn et al., 2023, **arXiv:2303.11366**

### 7.6 Tool use in language models

- **Toolformer:** Schick et al., 2023, **arXiv:2302.04761**

### 7.7 Multi-provider LLM routing

Three free-tier providers attempted in order (Groq → Gemini → OpenRouter), each
exposing an OpenAI-compatible API so a single client abstraction covers all three. A
provider with no configured key is skipped; a rate-limited provider falls through.

**This is a reliability requirement, not an optimization.** Free tiers are individually
tight enough that a single provider will fail during a demonstration.

---

## 8. Scientometrics — The Core Methodology

This section specifies the project's distinguishing contribution in full.

### 8.0 What scientometrics is, and why it applies here

**Scientometrics** is the quantitative study of science itself — measuring research
output, impact, and the structure of scholarly communication. Its central insight is
that a paper is not only a document; it is a **node in a network**, produced at a point
in time, by authors with histories, in a venue with a reputation. All of that is
measurable data, and conventional text retrieval discards every bit of it.

The project applies four families of indicator:

| Family | Question answered | Indicators used |
|---|---|---|
| **Impact** | Does the field care about this paper? | Citation count, field-normalized impact, influential citations |
| **Network position** | Where does it sit in the structure of the field? | PageRank centrality, in-degree |
| **Relatedness** | Which papers does the field treat as belonging with it? | Co-citation, bibliographic coupling |
| **Temporal** | Is attention to it rising or falling? | Citation momentum, burst detection |

### 8.1 The age problem — why raw counts are unusable

Citation counts accumulate over time, so they measure age at least as much as quality.
Within a single field, the median citation count of a paper published eight years ago
will typically exceed that of a paper published last year by more than an order of
magnitude.

**Consequence: a ranking function that uses raw citation counts ranks by age.** It
would systematically bury recent work, which is precisely the work a researcher asking
"what is new in this area" needs to see.

Every impact indicator in this project is therefore normalized.

### 8.2 Field-Normalized Citation Impact (CNCI / FWCI)

The standard correction. A paper's citation count is divided by the typical citation
count for papers of the **same field and the same year**:

```
CNCI(p) = C(p) / median{ C(q) : q ∈ cohort(field(p), year(p)) }
```

Interpretation:

| Value | Meaning |
|---|---|
| 1.0 | Exactly typical for its field and year |
| 5.0 | Five times the typical paper of its cohort |
| 0.2 | One fifth of typical |

A 2025 paper with 40 citations and a 2017 paper with 4,000 can now be compared
directly, which raw counts make impossible.

- **Known as:** CNCI (Category Normalized Citation Impact — Clarivate / Web of
  Science) and FWCI (Field-Weighted Citation Impact — Elsevier / Scopus)
- **Source:** Waltman, van Eck, van Leeuwen, Visser & van Raan, 2011, *Towards a new
  crown indicator: an empirical analysis*, Scientometrics 87(3)

**Methodological decision 1 — median, not mean.** Citation distributions are extreme
right-skewed, approximately log-normal with a heavy tail. A field containing papers
with 200,000 citations alongside thousands with fewer than ten will have a mean far
above anything typical, so normalizing by the mean would make the overwhelming majority
of papers appear below average. The median describes the actual centre of the
distribution.

**Methodological decision 2 — refuse to normalize thin cohorts.** Below a minimum
cohort size (set at 30 papers), the median is too noisy to be a trustworthy
denominator. Papers in such cohorts receive a NULL indicator rather than a computed
one. *"Unknown"* and *"typical for its field"* are different claims, and a ranking model
handed a substitute value for the former would learn from a fabrication.

**Methodological decision 3 — cohort year comes from the submission date.** A paper
submitted on 31 December is announced the following January and receives an identifier
from the new year. Using the identifier would file it in the wrong cohort and distort
that year's median.

### 8.3 Influential citations

Not all citations are equal. "We build directly on the method of [X]" and "prior work
includes [X, Y, Z]" both count as one citation in a raw tally, but they represent
completely different degrees of intellectual debt.

Semantic Scholar runs a classifier that distinguishes substantive from perfunctory
citations. The ratio

```
influential_ratio(p) = influential_citations(p) / total_citations(p)
```

captures whether a paper is genuinely built upon or merely acknowledged.

- **Source:** Valenzuela, Ha & Etzioni, 2015, *Identifying Meaningful Citations*,
  AAAI Workshop on Scholarly Big Data

### 8.4 PageRank centrality

Citation counts measure how *often* a paper is cited. PageRank measures *by whom*.

```
PR(p) = (1 − d)/N  +  d · Σ_{q → p}  PR(q) / L(q)
```

where `q → p` denotes q citing p, `L(q)` is q's number of outgoing references, `N` is
the number of papers, and `d = 0.85` is the damping factor.

The definition is recursive: **a paper is important if important papers cite it.**
Twenty citations from foundational works therefore outweigh a hundred from
peripheral ones — a distinction no citation count can express, because counting treats
every citation as identical.

- **Source:** Page, Brin, Motwani & Winograd, 1999, *The PageRank Citation Ranking:
  Bringing Order to the Web*, Stanford InfoLab
- **Implementation:** `networkx.pagerank`

**Expected behaviour as a validity check.** If the citation graph is correctly
constructed, PageRank over a deep-learning corpus should recover recognizably
foundational work without any supervision beyond citation edges. Any substantial
divergence from that expectation indicates a defect in graph construction rather than
an interesting finding, and this will be used as a sanity check before the indicator is
trusted.

**Methodological decision — the induced subgraph.** An edge is retained only when
*both* endpoints are corpus papers. A typical paper cites forty or more works, most
outside any single field sample; retaining those would introduce hundreds of thousands
of leaf nodes that can never be ranked or returned, while diluting the centrality mass
of every paper that can. The resulting claim is deliberately narrow and defensible:
*central within deep learning as sampled here*, not *central within all of science*.

**Methodological decision — edge direction.** Edges run citing → cited, so importance
flows backward along citations toward the work being cited. Reversing this would rank
papers by how many references they contain, which measures bibliography length rather
than influence.

### 8.5 Co-citation analysis

Papers A and B are **co-cited** when some third paper cites both. Co-citation strength
is the count of such papers:

```
cocite(A, B) = |{ p : p → A  and  p → B }|
```

This measures *"the field treats these two papers as belonging together."* It is a
relatedness signal that text embeddings cannot produce, because two co-cited papers may
share no vocabulary at all — a theoretical result and an applied system that always
appear together in related-work sections, for instance.

- **Source:** Small, H., 1973, *Co-citation in the scientific literature: A new
  measure of the relationship between two documents*, Journal of the American Society
  for Information Science 24(4)

**Temporal direction: forward.** Co-citation requires other papers to have cited you,
so it accumulates only over years. A paper published last month has none.

### 8.6 Bibliographic coupling

Papers A and B are **coupled** when they cite the same earlier papers. Coupling
strength is the size of their shared reference set:

```
couple(A, B) = | R(A) ∩ R(B) |        where R(x) = x's reference list
```

This measures *shared intellectual ancestry* — two papers drawing on the same
foundations are likely addressing the same problem.

- **Source:** Kessler, M. M., 1963, *Bibliographic coupling between scientific
  papers*, American Documentation 14(1)

**Temporal direction: backward.** A paper's reference list exists on the day it is
published, so coupling is available immediately for brand-new work.

**Why both measures are included.** They are complements, each informative exactly
where the other is blind:

| | Co-citation | Bibliographic coupling |
|---|---|---|
| Direction | Forward (who cites us) | Backward (whom we cite) |
| Available for new papers | No | **Yes, immediately** |
| Stability over time | Grows and changes | **Fixed at publication** |
| Captures | Field's current grouping | Authors' intellectual basis |

### 8.7 Temporal dynamics — momentum and burst

Whether attention to a paper is accelerating:

```
momentum(p) = citations(p, last 12 months) / citations(p, previous 12 months)
```

A ratio above 1 indicates rising attention. This separates two genuinely different
recommendations: *canonical* (high impact, stable) versus *ascending* (moderate impact,
rapidly rising) — and researchers ask for both, at different times.

The rigorous form of this analysis is burst detection, which identifies statistically
significant periods of elevated activity rather than relying on a simple ratio.

- **Source:** Kleinberg, J., 2002, *Bursty and Hierarchical Structure in Streams*,
  KDD

### 8.8 Venue prestige

Publication venue is a weak but non-zero prior on quality, and — more practically — a
**legitimacy signal for the user interface**. A result list showing only arXiv preprint
identifiers reads as informal; the same list annotated "published at NeurIPS 2023"
reads as scholarly.

- **Ranking source:** CORE conference rankings (free; assigns A*/A/B/C, with
  NeurIPS, ICML, ICLR, CVPR and ACL rated A*)
- **Venue metadata source:** Semantic Scholar / OpenAlex, resolved through DOI

**Known difficulty:** venue name normalization is non-trivial. "NeurIPS", "NIPS" and
"Advances in Neural Information Processing Systems" denote one venue under three
strings, and reliable matching requires an alias table.

### 8.9 Author-level indicators

The **h-index** — an author has h-index *N* if they have *N* papers each cited at least
*N* times — measures sustained output and impact together. Semantic Scholar exposes it
directly.

- **Source:** Hirsch, J. E., 2005, *An index to quantify an individual's scientific
  research output*, PNAS 102(46)

Included as an available feature; expected to be weaker than paper-level indicators and
subject to well-documented criticisms (field dependence, career-stage bias, insensitivity
to author order).

### 8.10 How the indicators enter the system

The indicators serve three distinct purposes, and it is worth separating them:

**1. As ranking features.** Fed to the learned ranker (§9), which decides how and when
to weight them.

**2. As displayed evidence.** Shown in the interface beneath each result, so a
recommendation carries its justification:

> **Attention Is All You Need** (2017) — cited 193,000 times, **6,400× the median for
> its field and year**. Highest centrality among your results. Published at NeurIPS.

versus

> **[a recent paper]** — 47 citations, but **6× its 2025 cohort median**, and citations
> have **tripled in the last six months**. Shares 12 references with the paper you are
> reading.

The second is an actionable recommendation *with its reasoning exposed*, which is the
difference between a search tool and a research assistant.

**3. As agent-reasonable facts.** Exposed through the `get_paper_metrics` tool so the
agent can reason about them in prose — *"this is the foundational work, but this 2025
paper is where the field is currently moving"* — rather than merely displaying numbers.

### 8.11 Data sources

| Source | Provides | Access |
|---|---|---|
| **arXiv API** | Metadata, abstracts, PDFs, primary category, submission date | Free; 1 request / 3 seconds |
| **Semantic Scholar Graph API** | Citation counts, influential citations, venue, reference lists, author h-index | Free; batch endpoint accepts 500 identifiers per request |
| **OpenAlex** | ~250M works, venue metadata, broader citation coverage | Free, no key required |
| **CORE rankings** | Conference tier (A*/A/B/C) | Free, downloadable list |

**On Scopus and Web of Science.** Both are the canonical scientometric databases, and
both are paywalled institutional subscriptions — incompatible with the zero-budget
constraint, and their terms generally cover *reading* rather than *programmatic
mining*. **OpenAlex** is the open successor to Microsoft Academic Graph and covers most
of the same corpus, including everything published by IEEE, at no cost. The
methodology (CNCI, PageRank, co-citation, coupling) is database-independent; only the
data source changes.

### 8.12 Validity considerations

Scientometric indicators are widely criticized when misapplied, and the project must
anticipate those criticisms rather than be surprised by them:

| Concern | Mitigation in this design |
|---|---|
| **Citation counts measure attention, not quality** | Indicators are used as *ranking evidence* alongside text relevance, never as a quality verdict |
| **Age bias** | Field-and-year normalization (§8.2) |
| **Field bias** | Corpus restricted to one field, so cohorts are genuine peer groups (§3) |
| **Popularity reinforcement** | The central risk; addressed in §10.4 |
| **Thin cohorts give unstable denominators** | Minimum cohort size, with explicit NULL rather than a substitute value |
| **Self-citation and citation cartels** | Not addressed in this scope; acknowledged as a limitation |
| **Coverage gaps for very recent work** | Indexing services lag publication by weeks; recency queries rely on submission date rather than citation data |

---

## 9. Techniques — Learned Ranking

### 9.1 Why a learned model rather than a tuned weight

The naive approach is a hand-written formula:

```
score = w₁ · text_relevance + w₂ · citation_impact + w₃ · centrality + ...
```

Two problems make this inadequate.

**The weights would be guesses.** Whether citation impact should count for more or less
than text similarity is not knowable a priori, and hand-tuning against observed output
is not a defensible methodology.

**The correct weights differ by query.** For *"show me the original transformer
paper"*, centrality should dominate — the user wants the canonical work. For *"what is
new in diffusion models"*, momentum should dominate and high centrality is close to a
**disqualifier**, since high centrality generally indicates an older paper. No single
fixed weight vector can serve both, and choosing one necessarily trades accuracy on one
query type for accuracy on the other.

What is required is a **conditional** policy. A gradient-boosted decision tree can
express conditional rules — *"if the query appears to seek foundational work and
centrality is high, promote; if the query closely matches a title, trust the text score
and ignore citations"* — which a weighted sum structurally cannot.

### 9.2 LambdaMART

Gradient-boosted decision trees that optimize ranking quality directly: the objective
concerns where the correct answer lands in the ordering, not whether each candidate is
individually classified correctly.

- **Algorithm:** Burges, C., 2010, *From RankNet to LambdaRank to LambdaMART: An
  Overview*, Microsoft Research MSR-TR-2010-82
- **Implementation:** LightGBM with `objective='lambdarank'` — Ke et al., 2017, NeurIPS

**Why this is tractable under the constraints.** Gradient-boosted trees over a dozen
tabular features are highly sample-efficient — this is not a deep learning problem
requiring millions of examples. Training completes in minutes on CPU, which matters
given no guaranteed GPU.

**Planned regularization, and why.** Small trees (few leaves, minimum samples per leaf
enforced). With on the order of a thousand query groups and fifteen features, a deeper
forest will memorize individual papers rather than learn a ranking policy — and
memorizing "this specific paper is good" is the popularity shortcut in a different
form.

### 9.3 Feature set

| Group | Features |
|---|---|
| **Text relevance** | cross-encoder score, BM25 score, query/title token overlap |
| **Citation evidence** | log citation count, log CNCI, CNCI-availability flag, influential-citation ratio |
| **Graph centrality** | log PageRank, log in-degree within corpus |
| **Paper properties** | age in years, venue present |
| **Query shape** | query length, recency markers, comparison markers, foundational-intent markers |

**On query shape as an intent proxy.** The conditioning signal the model requires is
"what kind of query is this." An LLM intent classifier is more accurate, but invoking
it once per training example would consume thousands of free-tier API calls. Cheap
textual features recover much of the conditioning and cost nothing at inference time.
The LLM classifier remains available to substitute at inference.

**On log-scaling and missingness flags.** Citation counts span six orders of magnitude;
log-scaling makes decision thresholds meaningful. An availability flag accompanies CNCI
because otherwise the model cannot distinguish "no citations" from "never computed" —
which are different claims.

### 9.4 Training data — weak supervision

Human relevance annotation at the scale a ranker needs is infeasible for this project.
**Weak supervision** generates labels automatically from structure that already exists.

**Option A — citation contexts.** The sentence surrounding a citation becomes a query;
the cited paper is the known-correct answer. Semantic Scholar exposes these contexts.
This is the principle underlying SPECTER's training.

**Option B — synthetic queries.** A sentence drawn from a paper's abstract becomes a
query whose correct answer is that paper. Unlimited, free, and requires no API access.

**Selection criterion.** The two options must be assessed not only for label quality
but for **label distribution**, specifically the distribution of *how famous the correct
answers are*. Any training source in which correct answers are predominantly
highly-cited papers will teach the model to rank by fame — the failure mode described
in §10.4 — and will do so invisibly, since the resulting model scores well on any
benchmark sharing the same bias.

The project will measure this distribution for both sources before training, select
accordingly, and resample if necessary to obtain a spread across the fame range.

---

## 10. Evaluation Design

### 10.1 Metrics

| Metric | Question answered | Source |
|---|---|---|
| **nDCG@10** | Is the correct paper ranked near the top? | Järvelin & Kekäläinen, 2002, *Cumulated gain-based evaluation of IR techniques*, ACM TOIS 20(4) |
| **Recall@20** | Was it retrieved at all? (position-blind) | Standard IR |
| **MRR** | How high was the first correct result? | Standard IR |

**nDCG@10** is the primary metric. It discounts each result by the logarithm of its
position — so a correct answer at rank 1 counts for substantially more than the same
answer at rank 10 — and normalizes by the best achievable ordering, making scores
comparable across queries with different numbers of correct answers.

**Graded relevance** is used rather than binary: 3 = this is the paper requested,
2 = highly relevant, 1 = marginally relevant, 0 = irrelevant. In a research corpus,
"returned an excellent survey of the topic" and "returned the exact paper requested"
are both successes but not equal ones, and a binary metric cannot express the
difference.

### 10.2 The evaluation set

Hand-written queries with verified correct answers, spanning four intents that are
**scored separately** because they fail differently — an aggregate average can conceal
a large gain and a large regression occurring simultaneously:

| Intent | Example |
|---|---|
| `specific_paper` | "show me the original transformer paper" |
| `topical` | "how does retrieval improve factual accuracy in language models" |
| `recency` | "recent work on state space models for long sequences" |
| `comparison` | "compare BERT and RoBERTa pretraining approaches" |

Every judged paper is verified present in the corpus before use. A judgement pointing
at a paper the system never ingested is unanswerable, and would penalize the system for
failing to return something it could not possibly return.

### 10.3 Evaluation isolation

Two vector collections are maintained:

| Collection | Role |
|---|---|
| **Live** | The running application writes here; grows whenever the agent discovers a paper |
| **Frozen** | A deliberate snapshot; never written to by the application |

All measurement uses the frozen collection. Without this separation, a paper ingested
by an ordinary user search between two measurements could shift a score in a way
indistinguishable from the change under test. Comparing rows of a results table
requires every row to have seen an identical corpus.

### 10.4 The principal methodological risk: popularity collapse

**This is the risk most likely to invalidate the project, and it is worth stating
plainly.**

A ranker given citation features can learn a shortcut: *rank by fame*. Such a model
will score excellently on any benchmark whose correct answers happen to be famous
papers — and will fail completely on the queries researchers most need help with, where
the correct answer is a specific, obscure, or very recent paper.

The danger is that this failure is **invisible to ordinary validation**. The model
scores well, the ablation reports citation count as the most valuable feature, and every
number looks correct. A confident, well-measured, wrong result.

**Three sources of this bias must be independently checked:**

| Source | Why it may be biased | Guard |
|---|---|---|
| **The evaluation set** | Queries written about papers the authors know are, by definition, about famous papers | Report the citation percentile distribution of correct answers as a property of the eval set |
| **The training data** | Citation-derived labels inherit the citation graph's skew toward highly-cited targets | Measure the distribution before training; resample across fame buckets if skewed |
| **The candidate pool** | If retrieval only ever surfaces famous papers, ranking cannot fix it | Verify recall separately from ranking |

**Countermeasures built into the evaluation design:**

1. **Adversarial queries.** A deliberate subset whose correct answers sit in the middle
   of the citation distribution rather than the top percentile. A model exploiting the
   shortcut cannot solve these, so their score is a direct diagnostic.
2. **Separate reporting.** Scores reported separately for famous-answer and
   obscure-answer queries, never only in aggregate. A system that improves one while
   destroying the other must be visible as such.
3. **Recency queries.** Recent papers cannot be highly cited, making them naturally
   adversarial to the shortcut.
4. **Distributional reporting.** The citation percentile profile of the evaluation set
   is reported alongside the scores, so any residual bias is visible to a reader rather
   than hidden inside a number.

**The generalizable principle:** *validate the distribution of your labels, not only the
value of your score.*

### 10.5 The ablation

The ablation, not the model, is the research deliverable. Each feature group is removed,
the model retrained, and the system re-measured.

**Why ablation rather than feature importance.** Feature importance describes the model
that happened to be built; it cannot distinguish a feature that mattered from one that
merely correlated with a feature that mattered. Removing a group and retraining measures
what that group was actually *worth*. The two diverge whenever features are correlated —
and citation indicators are heavily correlated with one another, which is exactly why
they are ablated as a group rather than individually.

### 10.6 Statistical caution

A hand-built evaluation set is necessarily small. Differences of a few hundredths of a
point fall within noise and must not be reported as findings; per-intent breakdowns rest
on still fewer queries and should not be over-interpreted. Any small reported delta
requires either a larger evaluation set or a significance test before it can be
defended.

---

## 11. Phase Plan

| Phase | Deliverable | Output |
|---|---|---|
| **0 — Scope & Baseline** | A measured baseline | Corpus, evaluation set, metric implementations, evaluation harness |
| **1 — Retrieval Core** | Improved retrieval, measured | Title indexing, BM25, RRF, embedding model comparison |
| **2 — Agentic Robustness** | A self-checking agent loop | Intent classification, retrieval grading, corrective retry, groundedness verification |
| **3 — Scientometric Layer** | An indicator scorecard per paper | Citation graph, CNCI, PageRank, co-citation, coupling, momentum |
| **4 — Learned Ranker** | Trained model plus ablation | LambdaMART over the full feature set |
| **5 — Product Surface** | The demonstrable system | Indicators in the interface, `get_paper_metrics` tool, venue badges |

### Phase 0 is not optional

It ships no user-visible feature and it is the phase every later claim depends on.
Without a baseline, "we improved retrieval" is an opinion; with one, it is a result.

### Cut order, agreed in advance

Decided before deadline pressure, so that the wrong thing is not sacrificed in a panic:

1. Embedding model comparison — retain the current model
2. Co-citation and coupling — ship centrality alone
3. OpenAlex and venue tiers — fall back to Semantic Scholar only
4. Adaptive stopping and query decomposition — retain grading and groundedness
5. **Never cut:** the baseline and the ablation. Removing either converts a measured
   contribution back into an application demonstration.

---

## 12. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Backend | Python + FastAPI | Same language as the ML tooling |
| Frontend | Next.js (App Router) + Tailwind CSS v4 | — |
| Vector store | ChromaDB (embedded) | No separate server, zero operational overhead |
| Metadata store | SQLite | Single-user, local-first |
| Graph analysis | networkx | PageRank, co-citation, coupling |
| Embeddings | sentence-transformers (BGE) | Local, free, CPU-capable |
| Reranking | sentence-transformers CrossEncoder | Local, free |
| Lexical search | rank_bm25 | Pure Python, no index server |
| Learned ranking | LightGBM | CPU-only, fast, effective on small data |
| LLM access | Groq / Gemini / OpenRouter free tiers | Zero budget |
| PDF parsing | PyMuPDF | Page-number tracking for precise citations |
| Graph visualization | Cytoscape.js | Built for dense relational graphs |
| Math rendering | KaTeX (remark-math + rehype-katex) | — |

---

## 13. Reference List

### Retrieval and RAG
1. Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — **arXiv:2005.11401**
2. Karpukhin et al., 2020 — *Dense Passage Retrieval for Open-Domain Question Answering* — **arXiv:2004.04906**
3. Guu et al., 2020 — *REALM: Retrieval-Augmented Language Model Pre-Training* — **arXiv:2002.08909**
4. Robertson & Zaragoza, 2009 — *The Probabilistic Relevance Framework: BM25 and Beyond* — FnTIR 3(4)
5. Cormack, Clarke & Büttcher, 2009 — *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* — SIGIR
6. Nogueira & Cho, 2019 — *Passage Re-ranking with BERT* — **arXiv:1901.04085**
7. Santhanam et al., 2021 — *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction* — **arXiv:2112.01488**
8. Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)* — **arXiv:2212.10496**
9. Sarthi et al., 2024 — *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* — **arXiv:2401.18059**
10. Xiao et al., 2023 — *C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE)* — **arXiv:2309.07597**

### Agents
11. Yao et al., 2022 — *ReAct: Synergizing Reasoning and Acting in Language Models* — **arXiv:2210.03629**
12. Shinn et al., 2023 — *Reflexion: Language Agents with Verbal Reinforcement Learning* — **arXiv:2303.11366**
13. Asai et al., 2023 — *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* — **arXiv:2310.11511**
14. Yan et al., 2024 — *Corrective Retrieval Augmented Generation* — **arXiv:2401.15884**
15. Schick et al., 2023 — *Toolformer: Language Models Can Teach Themselves to Use Tools* — **arXiv:2302.04761**

### Scholarly document representation
16. Cohan et al., 2020 — *SPECTER: Document-level Representation Learning using Citation-informed Transformers* — **arXiv:2004.07180**
17. Ostendorff et al., 2022 — *Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings (SciNCL)* — **arXiv:2202.06671**

### Scientometrics
18. Page, Brin, Motwani & Winograd, 1999 — *The PageRank Citation Ranking: Bringing Order to the Web* — Stanford InfoLab Technical Report
19. Small, H., 1973 — *Co-citation in the scientific literature: A new measure of the relationship between two documents* — JASIS 24(4)
20. Kessler, M. M., 1963 — *Bibliographic coupling between scientific papers* — American Documentation 14(1)
21. Waltman, van Eck, van Leeuwen, Visser & van Raan, 2011 — *Towards a new crown indicator: an empirical analysis* — Scientometrics 87(3)
22. Kleinberg, J., 2002 — *Bursty and Hierarchical Structure in Streams* — KDD
23. Valenzuela, Ha & Etzioni, 2015 — *Identifying Meaningful Citations* — AAAI Workshop on Scholarly Big Data
24. Hirsch, J. E., 2005 — *An index to quantify an individual's scientific research output* — PNAS 102(46)

### Learning to rank and evaluation
25. Burges, C., 2010 — *From RankNet to LambdaRank to LambdaMART: An Overview* — Microsoft Research MSR-TR-2010-82
26. Ke et al., 2017 — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* — NeurIPS
27. Järvelin & Kekäläinen, 2002 — *Cumulated gain-based evaluation of IR techniques* — ACM TOIS 20(4)

### Verification note
The arXiv identifiers for references 1–3, 6–9 and 11–17 were resolved directly against
the arXiv API by title search with string-similarity verification of the returned
title, and are confirmed. References 4, 5 and 18–27 are non-arXiv publications
(journals, conferences, technical reports); **these should be verified against the
publisher before submission**, particularly volume, issue and page numbers.

---

## 14. Semester 2 — Planned Extension

A multi-agent research pipeline. The scientometric layer feeds it directly: impact and
centrality are how the system decides *which* paper is worth the effort of deeper
analysis.

**Planned capability:** given a paper the user selects, generate and execute code that
demonstrates its core mechanism — implement the equation, run it on toy data, and plot
the behaviour the paper claims.

**Three tiers, in order of feasibility:**

| Tier | Description | Feasibility |
|---|---|---|
| **1 — Visualization** | Code illustrating a concept: an attention heatmap, an optimizer's trajectory on a loss surface, a comparison of sampling methods | High. No training, runs in seconds on CPU |
| **2 — Core mechanism** | Implement the paper's equation as a function; verify shapes, gradients, and claimed properties against a reference | Moderate |
| **3 — Full reproduction** | Reproduce the paper's reported results table | **Out of scope.** A research project per paper |

**Infrastructure:** a Docker-sandboxed `run_python(code)` tool — no network access,
memory cap, single CPU, short timeout — combined with the same
write / run / read-traceback / fix loop the agent already uses for tool calls.

**Design principle: the agent must be able to decline.** If a paper's contribution is
an empirical scaling result requiring a compute cluster, the correct output is *"this
cannot be demonstrated at toy scale, and here is why."* A system that knows what it
cannot show is more trustworthy than one that fabricates a demonstration.

**Why the zero-budget constraint helps here.** Unable to train anything, the system is
forced toward small, illustrative, one-second demonstrations — which is precisely what
a researcher trying to understand a paper wants. Nobody reading a paper wants a
forty-hour training run; they want *"show me what this attention mask actually does."*
