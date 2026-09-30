"""Strict chat response extraction and explicit conversation state."""

from __future__ import annotations

import json
from copy import copy, deepcopy
from dataclasses import dataclass, field
from http.cookiejar import Cookie, CookieJar
from urllib.request import Request

import httpx

from ragdrag.engine.models import OutcomeCode
from ragdrag.engine.profile import TargetProfile, canonical_origin
from ragdrag.engine.transport import HTTPClient, OriginBoundClient


@dataclass(frozen=True)
class ChatConfig:
    target_url: str
    query_field: str = "query"
    response_field: str | None = None
    history_field: str | None = None
    session_field: str | None = None
    session_id: str | None = None

    @classmethod
    def from_profile(cls, profile: TargetProfile) -> ChatConfig:
        return cls(
            target_url=profile.target_url,
            query_field=profile.query_field,
            response_field=profile.response_field,
            history_field=profile.history_field,
            session_field=profile.session_field,
            session_id=profile.session_id,
        )


@dataclass
class ConversationState:
    messages: list[dict[str, str]] = field(default_factory=list)
    session_id: str | None = None
    mechanism: str | None = None
    _cookie_keys: set[tuple[str, str, str, str]] = field(default_factory=set, repr=False, compare=False)


class ChatResponseError(RuntimeError):
    def __init__(self, outcome: OutcomeCode, detail: str) -> None:
        super().__init__(detail)
        self.outcome = outcome


@dataclass(frozen=True)
class ChatReply:
    text: str
    status_code: int
    state_mechanism: str | None


def extract_response_text(response: httpx.Response, field: str | None) -> str:
    """Return configured text or a typed error for an unsupported response."""
    if field is None:
        return response.text
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        raise ChatResponseError(OutcomeCode.UNSUPPORTED_RESPONSE, "response is not JSON") from error
    if not isinstance(payload, dict) or field not in payload:
        raise ChatResponseError(
            OutcomeCode.UNSUPPORTED_RESPONSE, f"response field {field!r} is unavailable"
        )
    value = payload[field]
    if not isinstance(value, str):
        raise ChatResponseError(
            OutcomeCode.UNSUPPORTED_RESPONSE, f"response field {field!r} is not text"
        )
    return value


def _cookie_snapshot(cookies: httpx.Cookies) -> dict[tuple[str, str, str], dict[str, object]]:
    """Copy complete cookie attributes without collapsing names across scopes."""
    return {
        (cookie.domain, cookie.path, cookie.name): deepcopy(vars(cookie))
        for cookie in cookies.jar
    }


def _cookie_applies(cookie: Cookie, target_url: str) -> bool:
    """Use the cookie jar's request rules for domain, path, scheme, and expiry."""
    jar = CookieJar()
    jar.set_cookie(copy(cookie))
    request = Request(target_url)
    jar.add_cookie_header(request)
    return request.has_header("Cookie")


def _cookie_jars(client: HTTPClient) -> dict[str, httpx.Cookies]:
    if isinstance(client, OriginBoundClient):
        # The public cookies property exposes only the primary origin's jar.
        return client._origin_cookies
    return {"": client.cookies}


def _cookie_header_overrides_jar(client: HTTPClient, response: httpx.Response) -> bool:
    if isinstance(client, OriginBoundClient):
        headers = client.profile.headers_for(str(response.url))
    else:
        # HTTPX strips/rebuilds Cookie on redirects, but auth retries also
        # appear in history and can retain the configured header. Require a
        # redirect status and Location, not merely a preceding response.
        # OriginBoundClient instead reapplies scoped headers at each hop.
        if any(hop.has_redirect_location for hop in response.history):
            return False
        headers = getattr(client, "headers", None)
    return "cookie" in httpx.Headers(headers)


class ChatAdapter:
    def __init__(self, client: HTTPClient, config: ChatConfig) -> None:
        self.client = client
        self.config = config

    def send(self, query: str, state: ConversationState) -> ChatReply:
        payload: dict[str, object] = {self.config.query_field: query}
        if self.config.history_field:
            payload[self.config.history_field] = [
                *(dict(message) for message in state.messages),
                {"role": "user", "content": query},
            ]
        session_id = state.session_id or self.config.session_id
        if self.config.session_field and session_id:
            payload[self.config.session_field] = session_id
        cookies_before = {
            origin: _cookie_snapshot(jar)
            for origin, jar in _cookie_jars(self.client).items()
        }
        response = self.client.post(self.config.target_url, json=payload)
        text = extract_response_text(response, self.config.response_field)
        try:
            final_url = str(response.url)
        except RuntimeError:
            final_url = None
        cookie_origin = (
            canonical_origin(final_url)
            if final_url is not None and isinstance(self.client, OriginBoundClient)
            else ""
        )
        final_jar = _cookie_jars(self.client).get(cookie_origin) if final_url is not None else None
        cookies_after = _cookie_snapshot(final_jar) if final_jar is not None else {}
        if (
            final_jar is not None
            and final_url is not None
            and not _cookie_header_overrides_jar(self.client, response)
        ):
            usable_cookies = {
                (cookie_origin, cookie.domain, cookie.path, cookie.name):
                    cookies_after[(cookie.domain, cookie.path, cookie.name)]
                for cookie in final_jar.jar
                if _cookie_applies(cookie, final_url)
            }
        else:
            usable_cookies = {}
        established = {
            key for key, attributes in usable_cookies.items()
            if cookies_before.get(cookie_origin, {}).get(key[1:]) != attributes
        }
        cookie_keys = (state._cookie_keys & usable_cookies.keys()) | established
        if self.config.history_field:
            mechanism = "history"
        elif self.config.session_field and session_id:
            mechanism = "session"
        else:
            mechanism = "cookie" if cookie_keys else None
        state.messages.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": text},
        ])
        state._cookie_keys = cookie_keys
        state.mechanism = mechanism
        return ChatReply(text, response.status_code, mechanism)
