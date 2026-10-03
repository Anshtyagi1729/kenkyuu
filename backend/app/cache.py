"""Generic content-addressed cache for external calls (API requests, LLM
completions). Keyed by a hash of the wrapped function's actual inputs, so a
repeated call with the same arguments never hits the network - critical
given how tight the free-tier rate limits are (Groq, Gemini, S2 all cap
requests/minute) and how easy it is to burn them on duplicate work.
"""

import functools
import hashlib
import json

import diskcache

from app.config import CACHE_DIR

_cache = diskcache.Cache(CACHE_DIR)
_MISS = object()


def _make_key(prefix: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"{prefix}:{digest}"


def cached(ttl_seconds: int):
    """Decorator: cache a function's return value by a hash of its arguments.

    Arguments must be JSON-serializable (or at least stable under str()) -
    stick to primitives/lists/dicts, not live client/connection objects.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = _make_key(fn.__qualname__, args, kwargs)
            hit = _cache.get(key, default=_MISS)
            if hit is not _MISS:
                return hit
            result = fn(*args, **kwargs)
            _cache.set(key, result, expire=ttl_seconds)
            return result

        wrapper.cache_clear = lambda: _cache.clear()
        return wrapper

    return decorator
