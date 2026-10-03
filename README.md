# Paper Agent

A research assistant for academic literature. Ask for a paper by name, by author, or by
topic; get a citation-grounded answer with measured impact figures attached, not
adjectives.

Built as a year-long capstone project. Runs entirely on free API tiers and local
embedding models.

## What makes it different from a plain RAG pipeline

Text similarity cannot tell the difference between a paper and a paper *about* that
paper. Asked for "the original transformer paper", a cross-encoder reranker promotes a
survey whose abstract narrates the Transformer's existence over the paper itself, whose
abstract never contains its own title.

This project adds **citation evidence** to retrieval to resolve exactly that:

- **Field-normalized impact (CNCI)** - citations relative to the median paper of the
  same field and year, because a raw citation count mostly measures a paper's age.
- **PageRank over the citation graph** - being cited *by* influential papers, which is
  not the same as being cited often.
- **Co-citation and bibliographic coupling** - related work found through citation
  structure rather than wording.
- **A learned ranker (LambdaMART)** over text and citation features, because a fixed
  citation weight trades accuracy on famous papers against accuracy on everything
  else. Only a model that conditions on the query can have both.

Every figure shown to the user is measured, and anything that could not be computed is
reported as such rather than as zero.

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then run it:

```bash
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI expects the API on `http://localhost:8000`; override with
`NEXT_PUBLIC_API_URL`.

## API keys

All optional, all free. The LLM router tries providers in order and falls through on
rate limits, so one key is enough to start.

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | preferred LLM provider |
| `GEMINI_API_KEY` | fallback LLM provider |
| `OPENROUTER_API_KEY` | second fallback |
| `S2_API_KEY` | Semantic Scholar |

**`S2_API_KEY` is worth more than it looks.** Unauthenticated Semantic Scholar access is
currently limited to the batch endpoint; every search endpoint returns HTTP 429. Since
Semantic Scholar is the only source here covering IEEE, Springer, Elsevier and ACM, the
key is what makes non-arXiv papers findable by title. Without it the system still works,
but it can only discover papers that have an arXiv preprint. The key is free from the
Semantic Scholar API page.

## Building the corpus

The app works immediately, ingesting papers as it discovers them. The scientometric
features need a corpus with citation metrics behind them:

```bash
cd backend
python -m scripts.build_corpus --stage all
```

Stages run independently (`seed`, `arxiv`, `metrics`, `graph`, `cohorts`, `index`,
`snapshot`) so an interrupted run resumes rather than restarting. Expect a few hours,
most of it waiting on API rate limits.

Then train the ranker:

```bash
python -m scripts.build_training_set
python -m scripts.train_ranker
```

## Evaluation

```bash
python -m scripts.evaluate --systems dense-only learned   # score retrieval configurations
python -m scripts.evaluate --validate                     # check the eval set is well-formed
python -m scripts.ablate --seeds 8                        # feature ablation and noise floor
python -m scripts.regression                              # can we still find specific papers
```

Two separate things, deliberately:

- `scripts.evaluate` and `scripts.ablate` measure **ranking quality** against a frozen
  snapshot of the vector store, so numbers stay comparable over time.
- `scripts.regression` checks that **specific named papers** are still retrievable from
  the live index, using the wording a real person would type. It exits non-zero on a
  miss.

Always read `--seeds` output alongside any ablation number. Retraining the same
configuration under different train/validation splits moves the score by more than most
feature-group differences, so a difference smaller than that spread is not a result.

## Layout

```
backend/app/
  ingestion/        arXiv and Semantic Scholar clients, PDF parsing, embeddings,
                    hybrid dense + BM25 retrieval
  scientometrics/   CNCI cohorts, PageRank, co-citation, impact scorecards
  ranking/          feature extraction, LambdaMART model, inference path
  evaluation/       nDCG / recall / MRR, eval harness
  llm/              provider-fallback router, tool-calling agent, grading
  storage/          SQLite metadata, Chroma vectors
  routes/           FastAPI endpoints
backend/scripts/    corpus build, training, evaluation, ablation, regression
frontend/           Next.js app
docs/PROJECT.md     full specification
```
