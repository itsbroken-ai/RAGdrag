"""Target normalization and origin-scoped request configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlparse, urlunparse

from ragdrag.engine.models import ImpactLevel, RequestBudget


def canonical_origin(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid HTTP target: {url}")
    port = parsed.port
    if port == 0:
        raise ValueError("HTTP target port zero is invalid")
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}:{port}"


def normalize_target(url: str) -> str:
    parsed = urlparse(url)
    canonical_origin(url)
    path = parsed.path.rstrip("/") if parsed.path not in {"", "/"} else parsed.path
    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, parsed.params, parsed.query, ""))


@dataclass(frozen=True)
class TargetProfile:
    target_url: str
    approved_origins: frozenset[str]
    headers_by_origin: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    query_field: str = "query"
    response_field: str | None = None
    history_field: str | None = None
    session_field: str | None = None
    session_id: str | None = None
    verify_ssl: bool = True
    budget: RequestBudget = field(default_factory=RequestBudget)
    impact_ceiling: ImpactLevel = ImpactLevel.ACTIVE

    def __post_init__(self) -> None:
        target = normalize_target(self.target_url)
        primary = canonical_origin(target)
        approved = frozenset(canonical_origin(origin) for origin in self.approved_origins) | {primary}
        scoped = {
            canonical_origin(origin): MappingProxyType(dict(headers))
            for origin, headers in self.headers_by_origin.items()
        }
        unknown = set(scoped) - set(approved)
        if unknown:
            raise ValueError(f"Header origin is not approved: {sorted(unknown)}")
        object.__setattr__(self, "target_url", target)
        object.__setattr__(self, "approved_origins", approved)
        object.__setattr__(self, "headers_by_origin", MappingProxyType(scoped))

    def headers_for(self, url: str) -> dict[str, str]:
        return dict(self.headers_by_origin.get(canonical_origin(url), {}))

    @classmethod
    def from_cli(
        cls,
        target_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        cookie: str | None = None,
        additional_origin_headers: Mapping[str, Mapping[str, str]] | None = None,
        **kwargs: object,
    ) -> TargetProfile:
        origin = canonical_origin(target_url)
        scoped = dict(headers or {})
        if cookie:
            scoped["Cookie"] = cookie
        additional = dict(additional_origin_headers or {})
        headers_by_origin = {origin: scoped, **additional}
        return cls(
            target_url=target_url,
            approved_origins=frozenset(headers_by_origin),
            headers_by_origin=headers_by_origin,
            **kwargs,
        )
