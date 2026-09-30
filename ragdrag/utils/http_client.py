"""Shared HTTP client configuration for RAGdrag."""

from __future__ import annotations

import httpx

from ragdrag import __version__
from ragdrag.engine.models import RequestBudget
from ragdrag.engine.profile import TargetProfile
from ragdrag.engine.transport import OriginBoundClient

DEFAULT_TIMEOUT = 30.0
DEFAULT_HEADERS = {
    "User-Agent": f"ragdrag/{__version__}",
}


def build_client(
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    verify_ssl: bool = True,
    *,
    target: str | None = None,
    cookie: str | None = None,
) -> OriginBoundClient:
    """Build a bounded client with any supplied credentials bound to target."""
    if target is None:
        if headers or cookie:
            raise ValueError("target is required when scoped headers or cookies are supplied")
        target = "http://localhost"
    profile = TargetProfile.from_cli(
        target,
        headers=headers,
        cookie=cookie,
        verify_ssl=verify_ssl,
        budget=RequestBudget(timeout_seconds=timeout),
    )
    return OriginBoundClient(profile)


def build_async_client(
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    verify_ssl: bool = True,
) -> httpx.AsyncClient:
    """Deprecated: build a legacy async client for uncredentialed requests."""
    if headers:
        raise ValueError("unscoped headers are unsupported by the deprecated async builder")
    return httpx.AsyncClient(
        timeout=timeout,
        headers=DEFAULT_HEADERS,
        verify=verify_ssl,
        follow_redirects=True,
    )
