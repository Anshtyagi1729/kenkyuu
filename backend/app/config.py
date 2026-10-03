import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
S2_API_KEY = os.getenv("S2_API_KEY", "")

CHROMA_DIR = str(BASE_DIR / os.getenv("CHROMA_DIR", "data/chroma"))
SQLITE_PATH = str(BASE_DIR / os.getenv("SQLITE_PATH", "data/papers.db"))
CACHE_DIR = str(BASE_DIR / os.getenv("CACHE_DIR", "data/cache"))

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# The corpus is deliberately scoped to deep learning rather than all of arXiv.
# Two reasons, and the second is the load-bearing one:
#   1. Retrieval quality is easier to evaluate honestly within one field.
#   2. Field-normalized citation impact needs a COHORT to normalize against.
#      "Average citations for a 2023 cs.LG paper" is a real, computable number;
#      "average citations for a 2023 paper" spans fields whose citation cultures
#      differ by an order of magnitude, and normalizing against it is meaningless.
# Set CORPUS_CATEGORIES="" in .env to disable scoping (searches all of arXiv).
_DEFAULT_CATEGORIES = "cs.LG,cs.CL,cs.CV,cs.AI,stat.ML"
CORPUS_CATEGORIES = tuple(
    c.strip() for c in os.getenv("CORPUS_CATEGORIES", _DEFAULT_CATEGORIES).split(",") if c.strip()
)

# All OpenAI-compatible endpoints, so one client abstraction covers all three.
# Model IDs on free tiers shift over time - these are current as of 2026-08-31,
# verified against provider docs, but double check before relying on them if
# this project sits untouched for a while.
GROQ_MODEL_CHEAP = os.getenv("GROQ_MODEL_CHEAP", "openai/gpt-oss-20b")
GROQ_MODEL_STRONG = os.getenv("GROQ_MODEL_STRONG", "openai/gpt-oss-120b")

GEMINI_MODEL_CHEAP = os.getenv("GEMINI_MODEL_CHEAP", "gemini-flash-lite-latest")
GEMINI_MODEL_STRONG = os.getenv("GEMINI_MODEL_STRONG", "gemini-flash-latest")

# OpenRouter's free-model pool rotates - no stable default. Set these in .env
# from https://openrouter.ai/models?max_price=0 (pick ones that support tool
# calling) if you want OpenRouter as a fallback; leave blank to skip it.
OPENROUTER_MODEL_CHEAP = os.getenv("OPENROUTER_MODEL_CHEAP", "")
OPENROUTER_MODEL_STRONG = os.getenv("OPENROUTER_MODEL_STRONG", "")

# LLM completions are cached for an hour by default - long enough to absorb
# repeat/duplicate queries during a demo or dev session without ever going
# stale in a way that matters for this use case.
LLM_CACHE_TTL_SECONDS = 3600
# API metadata (search results, paper details) changes slowly; cache longer.
API_CACHE_TTL_SECONDS = 3600
