"""Minimal Server-Sent Events formatting, shared by streaming routes."""

import json


def sse_event(event: str, data: str | list | dict) -> str:
    # Always JSON-encode, even plain strings: SSE frames data line-by-line, so a raw
    # token containing a literal newline (very possible mid-answer) would otherwise
    # split across an unprefixed continuation line and get silently dropped by any
    # parser that only reads lines starting with "data:".
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
