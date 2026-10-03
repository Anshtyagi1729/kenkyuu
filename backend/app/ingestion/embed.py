"""Local, free embeddings via sentence-transformers. No API calls, no cost.

Models are registered with their own query/passage conventions rather than assumed
to share one. This matters more than it looks: BGE expects an instruction prefix on
queries and none on passages, E5 wants "query:"/"passage:" on both sides, and the
scholarly models (SPECTER) want neither because they were trained directly on
title+abstract text. Applying the wrong convention silently degrades a model rather
than failing, which would make a bake-off measure prompt handling instead of model
quality.
"""

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


@dataclass(frozen=True)
class EmbeddingSpec:
    """How to talk to one embedding model."""

    model_name: str
    query_prefix: str = ""
    passage_prefix: str = ""


SPECS: dict[str, EmbeddingSpec] = {
    # BGE: asymmetric - queries carry an instruction, passages go in bare.
    "BAAI/bge-small-en-v1.5": EmbeddingSpec(
        "BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "BAAI/bge-base-en-v1.5": EmbeddingSpec(
        "BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    # SPECTER: trained on citation-linked title+abstract pairs, so papers that cite
    # each other embed close together. No prefixes - the text goes in as-is.
    "allenai/specter": EmbeddingSpec("allenai/specter"),
    # E5: symmetric prefixes on BOTH sides.
    "intfloat/e5-base-v2": EmbeddingSpec(
        "intfloat/e5-base-v2", query_prefix="query: ", passage_prefix="passage: "
    ),
}


def spec_for(model_name: str | None = None) -> EmbeddingSpec:
    name = model_name or EMBEDDING_MODEL
    # An unregistered model is usable, just with no prefixes - better than refusing
    # to run, and the default is right for most non-instruction-tuned encoders.
    return SPECS.get(name, EmbeddingSpec(name))


@lru_cache(maxsize=4)
def _model(model_name: str) -> SentenceTransformer:
    """Cached per name so a bake-off can hold several models without reloading each
    time it switches between them."""
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    spec = spec_for(model_name)
    prefixed = [f"{spec.passage_prefix}{text}" for text in texts] if spec.passage_prefix else texts
    return _model(spec.model_name).encode(prefixed, normalize_embeddings=True).tolist()


def embed_query(query: str, model_name: str | None = None) -> list[float]:
    spec = spec_for(model_name)
    return _model(spec.model_name).encode(
        f"{spec.query_prefix}{query}", normalize_embeddings=True
    ).tolist()
