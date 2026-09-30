"""Copy evidence summaries while removing values stored under sensitive keys."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "<redacted>"
SENSITIVE_KEY_TOKENS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "api-key",
    "api_key",
    "token",
    "secret",
    "password",
    "raw_response",
    "response_text",
    "content",
    "body",
    "document",
    "vector",
    "matched_value",
}


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower()
    return any(token in normalized for token in SENSITIVE_KEY_TOKENS)


def _is_safe_metadata(key: str, value: Any) -> bool:
    normalized = key.lower()
    if normalized.endswith("_length"):
        return type(value) is int and value >= 0
    if normalized.endswith("_sha256"):
        return type(value) is str and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
    return False


def redact(value: Any, key: str = "") -> Any:
    """Return a detached, recursively redacted copy of a summary value."""
    if type(key) is not str:
        raise ValueError("summary keys must be built-in strings")
    if _is_sensitive_key(key):
        return value if _is_safe_metadata(key, value) else REDACTED
    if isinstance(value, dict):
        result = {}
        for item_key, item_value in value.items():
            if type(item_key) is not str:
                raise ValueError("summary keys must be built-in strings")
            result[item_key] = redact(item_value, item_key)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if type(value) in (str, int, float, bool, type(None)):
        return value
    return REDACTED
