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


def _configure_retries() -> None:
    """Lets a rate limit reach the router when there is somewhere else to go.

    The OpenAI SDK retries a 429 internally, sleeping for the provider's Retry-After -
    measured on Groq's free tier at 12 to 46 seconds per call. While it sleeps, the
    router never sees the error, so the fall-through to Gemini or OpenRouter that this
    module exists for never happens: a question that should have moved to the next
    provider in a second instead waited out Groq's quota, and a four-call answer took
    nearly two minutes.

    With more than one provider configured, the SDK's own retries are turned off and the
    router falls through immediately. With only one, they stay on - there is nowhere
    else to go, and waiting out the limit beats failing the request.
    """
    available = [p for p in _PROVIDERS if p.available]
    if len(available) > 1:
        for provider in available:
            provider.client = provider.client.with_options(max_retries=0)
    logger.info(
        "LLM providers configured: %s%s",
        ", ".join(p.name for p in available) or "none",
        " (immediate fall-through on rate limits)" if len(available) > 1 else
        " (single provider: rate limits are waited out - add another key to avoid this)",
    )


_configure_retries()


# Deliberately NOT caching LLM completions app-side (@cached(ttl_seconds=LLM_CACHE_TTL_SECONDS)
# above this def, if re-enabled). Exact-match response caching only hits on byte-identical
# repeat queries (~30% hit rate industry-wide) and adds no value here: Groq auto-caches the
# repeated prefix (system prompt + tool schemas) for our gpt-oss models for free, and those
# cached tokens don't even count against the rate limit; Gemini 2.5+ does the same on its free
# tier. That provider-side caching is what actually protects our rate limits - this app-level
# layer was solving a problem that's already solved upstream. Left commented, not deleted, in
# case a provider/model swap ever loses prompt-caching support.
# Google's documented placeholder for a function call whose signature is unavailable
# because it was produced by a different model.
_GEMINI_PLACEHOLDER_SIGNATURE = "skip_thought_signature_validator"


def _prepare_messages(provider_name: str, messages: list[dict]) -> list[dict]:
    """Adapts the tool-call history to what each provider accepts.

    Gemini's thinking models attach a "thought signature" to every function call they
    emit and reject any request whose history contains a function call without one
    (HTTP 400, "Function call is missing a thought_signature"). Falling back between
    providers mid-answer produces exactly that history: the orchestrator's first tool
    calls come from Groq, Groq rate-limits, and the conversation moves to Gemini
    carrying Groq's unsigned calls. Observed live once a Gemini key was added - the
    fallback that was meant to rescue a rate-limited answer failed it instead.

    So for Gemini every unsigned call gets the placeholder Google documents for
    conversations that began on another model, and signatures Gemini produced itself
    are passed through untouched. For other providers the Gemini-specific field is
    removed, since Groq rejects unfamiliar fields echoed back in messages.
    """
    prepared = []
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not calls:
            prepared.append(message)
            continue
        new_calls = []
        for call in calls:
            call = {k: v for k, v in call.items() if k != "extra_content" or provider_name == "gemini"}
            if provider_name == "gemini":
                google = dict((call.get("extra_content") or {}).get("google") or {})
                google.setdefault("thought_signature", _GEMINI_PLACEHOLDER_SIGNATURE)
                call["extra_content"] = {**(call.get("extra_content") or {}), "google": google}
            new_calls.append(call)
        prepared.append({**message, "tool_calls": new_calls})
    return prepared


def _call_provider(provider_name: str, model: str, messages: list[dict], tools: list[dict] | None) -> dict:
    provider = _PROVIDER_BY_NAME[provider_name]
    kwargs = {"model": model, "messages": _prepare_messages(provider_name, messages)}
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
    stream = provider.client.chat.completions.create(
        model=model, messages=_prepare_messages(provider_name, messages), stream=True
    )
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
