"""Provider-fallback LLM router: Groq -> Gemini -> OpenRouter free pool.

All three expose an OpenAI-compatible chat completions API, so one client
abstraction (just swapping base_url/api_key) covers all of them - no
provider-specific SDKs. A provider with no API key configured is skipped
silently, so this works with as few as one free-tier key set.

Two model tiers per provider: "cheap" for agent tool-routing steps (most of
an agentic loop's calls), "strong" reserved for final answer synthesis.

App-level response caching was tried and deliberately removed (see the note
above _call_provider) - Groq/Gemini's own automatic prompt-prefix caching
already protects the rate limits that matter, for free.
"""

import logging

import openai

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_CHEAP,
    GEMINI_MODEL_STRONG,
    GROQ_API_KEY,
    GROQ_MODEL_CHEAP,
    GROQ_MODEL_STRONG,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL_CHEAP,
    OPENROUTER_MODEL_STRONG,
)

logger = logging.getLogger(__name__)


class NoProviderAvailable(RuntimeError):
    pass


class _Provider:
    def __init__(self, name: str, base_url: str, api_key: str, cheap_model: str, strong_model: str):
        self.name = name
        self.api_key = api_key
        self.cheap_model = cheap_model
        self.strong_model = strong_model
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key) if api_key and cheap_model else None

    @property
    def available(self) -> bool:
        return self.client is not None

    def model_for(self, tier: str) -> str:
        return self.strong_model if tier == "strong" else self.cheap_model


_PROVIDERS = [
    _Provider("groq", "https://api.groq.com/openai/v1", GROQ_API_KEY, GROQ_MODEL_CHEAP, GROQ_MODEL_STRONG),
    _Provider(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        GEMINI_API_KEY,
        GEMINI_MODEL_CHEAP,
        GEMINI_MODEL_STRONG,
    ),
    _Provider(
        "openrouter",
        "https://openrouter.ai/api/v1",
        OPENROUTER_API_KEY,
        OPENROUTER_MODEL_CHEAP,
        OPENROUTER_MODEL_STRONG,
    ),
]
_PROVIDER_BY_NAME = {p.name: p for p in _PROVIDERS}


# Deliberately NOT caching LLM completions app-side (@cached(ttl_seconds=LLM_CACHE_TTL_SECONDS)
# above this def, if re-enabled). Exact-match response caching only hits on byte-identical
# repeat queries (~30% hit rate industry-wide) and adds no value here: Groq auto-caches the
# repeated prefix (system prompt + tool schemas) for our gpt-oss models for free, and those
# cached tokens don't even count against the rate limit; Gemini 2.5+ does the same on its free
# tier. That provider-side caching is what actually protects our rate limits - this app-level
# layer was solving a problem that's already solved upstream. Left commented, not deleted, in
# case a provider/model swap ever loses prompt-caching support.
def _call_provider(provider_name: str, model: str, messages: list[dict], tools: list[dict] | None) -> dict:
    provider = _PROVIDER_BY_NAME[provider_name]
    kwargs = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    completion = provider.client.chat.completions.create(**kwargs)
    return completion.model_dump()


def generate(messages: list[dict], tools: list[dict] | None = None, tier: str = "cheap") -> dict:
    """Returns the OpenAI-format completion as a plain dict, e.g.
    result["choices"][0]["message"]["content"] / ["tool_calls"]."""
    last_error: Exception | None = None
    for provider in _PROVIDERS:
        if not provider.available:
            continue
        model = provider.model_for(tier)
        try:
            return _call_provider(provider.name, model, messages, tools)
        except openai.APIError as exc:
            logger.warning("Provider %s failed (%s) - falling through", provider.name, exc)
            last_error = exc
            continue

    raise NoProviderAvailable(
        "No LLM provider available. Check that at least one of GROQ_API_KEY / "
        f"GEMINI_API_KEY / OPENROUTER_API_KEY is set in .env. Last error: {last_error}"
    )


def _call_provider_stream(provider_name: str, model: str, messages: list[dict]):
    provider = _PROVIDER_BY_NAME[provider_name]
    # The openai client makes the request (and would raise here on an auth/rate-limit
    # error) inside create() itself - iteration below just parses already-arriving
    # chunks, so a provider failing after this line means it failed mid-stream, not
    # before sending anything.
    stream = provider.client.chat.completions.create(model=model, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_stream(messages: list[dict], tier: str = "cheap"):
    """No tool-calling support (synthesis-only, doesn't need it). Yields text deltas.
    Provider fallback only covers failures before the first token - once a provider
    starts streaming, this doesn't retry elsewhere mid-answer (see comment above)."""
    last_error: Exception | None = None
    for provider in _PROVIDERS:
        if not provider.available:
            continue
        model = provider.model_for(tier)
        try:
            yield from _call_provider_stream(provider.name, model, messages)
            return
        except openai.APIError as exc:
            logger.warning("Provider %s failed (%s) - falling through", provider.name, exc)
            last_error = exc
            continue

    raise NoProviderAvailable(
        "No LLM provider available. Check that at least one of GROQ_API_KEY / "
        f"GEMINI_API_KEY / OPENROUTER_API_KEY is set in .env. Last error: {last_error}"
    )
