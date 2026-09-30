"""Bounded HTTP transport with credentials scoped to approved origins."""

from __future__ import annotations

import zlib
from contextlib import contextmanager
from collections.abc import Iterator
from threading import Lock
from typing import Any, Protocol
from urllib.parse import urljoin

import httpx

from ragdrag import __version__
from ragdrag.engine.models import OutcomeCode
from ragdrag.engine.profile import TargetProfile, canonical_origin


class RequestBudgetExceeded(RuntimeError):
    """The engagement has reached its request limit."""


class ResponseTooLarge(RuntimeError):
    """A response body exceeded the configured byte limit."""


_EXACT_TRANSPORT_ERRORS = (
    httpx.TransportError, httpx.TimeoutException,
    httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
    httpx.NetworkError, httpx.ConnectError, httpx.ReadError, httpx.WriteError,
    httpx.CloseError, httpx.ProtocolError, httpx.LocalProtocolError,
    httpx.RemoteProtocolError, httpx.ProxyError, httpx.UnsupportedProtocol,
)


def _request_exception_outcome(exc: Exception) -> OutcomeCode:
    """Classify only known exact classes; never inspect exception instances."""
    error_type = type(exc)
    if any(error_type is known for known in _EXACT_TRANSPORT_ERRORS):
        return OutcomeCode.UNREACHABLE
    if error_type is ResponseTooLarge or error_type is httpx.DecodingError:
        return OutcomeCode.UNSUPPORTED_RESPONSE
    return OutcomeCode.INDETERMINATE


class PhaseObservation:
    """A request-bound, bounded collection of distinct phase failures."""

    def __init__(self, response_field: str | None) -> None:
        self._response_field = response_field
        self._outcomes: dict[OutcomeCode, None] = {}
        self._lock = Lock()
        self._in_flight = 0

    @property
    def response_field(self) -> str | None:
        return self._response_field

    def record(self, outcome: OutcomeCode | None) -> None:
        if type(outcome) is OutcomeCode:
            with self._lock:
                self._outcomes.setdefault(outcome, None)

    def snapshot(self) -> tuple[OutcomeCode, ...]:
        with self._lock:
            return tuple(self._outcomes)

    def begin_request(self) -> None:
        with self._lock:
            self._in_flight += 1

    def finish_request(self) -> None:
        with self._lock:
            self._in_flight -= 1

    def has_in_flight_requests(self) -> bool:
        with self._lock:
            return self._in_flight != 0

    def __iter__(self) -> Iterator[OutcomeCode]:
        return iter(self.snapshot())


def _bounded_zlib_chunks(chunks: Iterator[bytes], encoding: str, limit: int) -> Iterator[bytes]:
    """Decode one content-encoding layer without producing over limit + 1 bytes."""
    wbits = zlib.MAX_WBITS | 16 if encoding == "gzip" else zlib.MAX_WBITS
    decoder = zlib.decompressobj(wbits)
    size = 0
    first = True
    saw_input = False

    for raw in chunks:
        saw_input = True
        if decoder.eof and raw:
            raise httpx.DecodingError("trailing compressed data")
        # A transport may yield a large raw chunk. Feed at most 64 KiB to
        # zlib at once so even its unused_data cannot retain an unbounded tail.
        for offset in range(0, len(raw), 65_536):
            piece = raw[offset : offset + 65_536]
            pending = piece
            while pending:
                if decoder.eof:
                    raise httpx.DecodingError("trailing compressed data")
                try:
                    output_limit = min(65_536, limit - size + 1)
                    decoded = decoder.decompress(pending, output_limit)
                except zlib.error as exc:
                    if encoding != "deflate" or not first:
                        raise httpx.DecodingError(str(exc)) from exc
                    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
                    try:
                        decoded = decoder.decompress(pending, output_limit)
                    except zlib.error as raw_exc:
                        raise httpx.DecodingError(str(raw_exc)) from raw_exc
                first = False
                remainder = decoder.unconsumed_tail
                size += len(decoded)
                if size > limit:
                    raise ResponseTooLarge(f"response exceeded {limit} bytes")
                if decoder.unused_data or (decoder.eof and remainder):
                    raise httpx.DecodingError("trailing compressed data")
                if decoder.eof and offset + len(piece) < len(raw):
                    raise httpx.DecodingError("trailing compressed data")
                if decoded:
                    yield decoded
                if remainder == pending and not decoded:
                    raise httpx.DecodingError("content decoder made no progress")
                pending = remainder

        # Drain decoder output before asking the source for another raw chunk.
        # An excess byte already decoded must win over a later read failure.
        while not decoder.eof:
            try:
                decoded = decoder.decompress(b"", min(65_536, limit - size + 1))
            except zlib.error as exc:
                raise httpx.DecodingError(str(exc)) from exc
            if not decoded:
                break
            size += len(decoded)
            if size > limit:
                raise ResponseTooLarge(f"response exceeded {limit} bytes")
            if decoder.unused_data:
                raise httpx.DecodingError("trailing compressed data")
            yield decoded
    if saw_input and not decoder.eof:
        raise httpx.DecodingError("incomplete compressed response")


def _bounded_body_chunks(response: httpx.Response, limit: int) -> Iterator[bytes]:
    # MockTransport can return an already-buffered Response(content=...). Its
    # content is decoded by HTTPX before send() gives it to us, and iter_raw()
    # is no longer available. Real streamed responses take the bounded path.
    if hasattr(response, "_content"):
        content = response.content
        if len(content) > limit:
            raise ResponseTooLarge(f"response exceeded {limit} bytes")
        if content:
            yield content
        return

    encodings = [
        item.strip().lower()
        for item in response.headers.get("content-encoding", "").split(",")
        if item.strip()
    ]
    if any(encoding not in {"identity", "gzip", "deflate"} for encoding in encodings):
        raise httpx.DecodingError("unsupported content encoding")

    layers = [encoding for encoding in reversed(encodings) if encoding != "identity"]
    if len(layers) > 4:
        raise httpx.DecodingError("unsupported content encoding stack")
    chunks: Iterator[bytes] = response.iter_raw()
    layer_overhead = 1_024 + limit // 1_024
    for index, encoding in enumerate(layers):
        remaining_layers = len(layers) - index - 1
        layer_limit = limit + layer_overhead * remaining_layers
        chunks = _bounded_zlib_chunks(chunks, encoding, layer_limit)

    size = 0
    for chunk in chunks:
        size += len(chunk)
        if size > limit:
            raise ResponseTooLarge(f"response exceeded {limit} bytes")
        if chunk:
            yield chunk


class HTTPClient(Protocol):
    @property
    def timeout(self) -> httpx.Timeout: ...

    @property
    def cookies(self) -> httpx.Cookies: ...

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response: ...

    def get(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def head(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def options(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def delete(self, url: str, **kwargs: Any) -> httpx.Response: ...

    def close(self) -> None: ...


class OriginBoundClient:
    """Synchronous HTTP client sharing request and response limits across calls."""

    _REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
    _FRAMING_HEADERS = frozenset(
        {"content-type", "content-length", "host", "accept-encoding", "transfer-encoding"}
    )

    def __init__(
        self,
        profile: TargetProfile,
        *,
        verify_ssl: bool | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.profile = profile
        self._requests = 0
        self._request_lock = Lock()
        self._observation_lock = Lock()
        self._phase_observer: PhaseObservation | None = None
        self._origin_cookies = {
            origin: httpx.Cookies() for origin in profile.approved_origins
        }
        self._operator_cookies = self._origin_cookies[canonical_origin(profile.target_url)]
        self._client = httpx.Client(
            timeout=profile.budget.timeout_seconds,
            verify=profile.verify_ssl if verify_ssl is None else verify_ssl,
            follow_redirects=False,
            transport=transport,
            headers={"User-Agent": f"ragdrag/{__version__}"},
        )

    @property
    def timeout(self) -> httpx.Timeout:
        return self._client.timeout

    @property
    def cookies(self) -> httpx.Cookies:
        return self._operator_cookies

    @property
    def requests_used(self) -> int:
        return self._requests

    @contextmanager
    def observe_phase(self, response_field: str | None) -> Iterator[PhaseObservation]:
        """Collect safe chat-request outcomes without retaining response bodies."""
        with self._observation_lock:
            if self._phase_observer is not None:
                raise RuntimeError("phase observation is already active")
            observer = PhaseObservation(response_field)
            self._phase_observer = observer
        completed = False
        try:
            yield observer
            completed = True
        finally:
            with self._observation_lock:
                if self._phase_observer is observer:
                    self._phase_observer = None
                in_flight = observer.has_in_flight_requests()
            if completed and in_flight:
                raise RuntimeError("phase observation has in-flight requests")

    def _observe_chat_response(
        self, response: httpx.Response, *, validate_field: bool, response_field: str | None,
    ) -> OutcomeCode | None:
        status = response.status_code
        if status in (401, 403):
            return OutcomeCode.AUTHENTICATION_REQUIRED
        if status == 404:
            return OutcomeCode.INVALID_TARGET
        if status == 429:
            return OutcomeCode.RATE_LIMITED
        if not 200 <= status < 300:
            return OutcomeCode.INDETERMINATE
        field = response_field if validate_field else None
        if field is None:
            return None
        if type(field) is not str or not field:
            return OutcomeCode.UNSUPPORTED_RESPONSE
        try:
            data = response.json()
        except (ValueError, TypeError):
            return OutcomeCode.UNSUPPORTED_RESPONSE
        if type(data) is not dict or type(data.get(field)) is not str:
            return OutcomeCode.UNSUPPORTED_RESPONSE
        return None

    def _bounded_timeout(self, value: Any) -> httpx.Timeout:
        cap = self.profile.budget.timeout_seconds
        if value is None:
            return httpx.Timeout(cap)
        timeout = value if isinstance(value, httpx.Timeout) else httpx.Timeout(value)

        def bounded(component: float | None) -> float:
            return min(component, cap) if component is not None else cap

        return httpx.Timeout(
            connect=bounded(timeout.connect),
            read=bounded(timeout.read),
            write=bounded(timeout.write),
            pool=bounded(timeout.pool),
        )

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        # Capture ownership before waiting on the request lock. An earlier
        # request may finish after its scope exits or a later scope begins.
        with self._observation_lock:
            observer = self._phase_observer if url == self.profile.target_url else None
            response_field = observer.response_field if observer is not None else None
            if observer is not None:
                observer.begin_request()
        try:
            return self._request_observed(method, url, observer, response_field, **kwargs)
        finally:
            if observer is not None:
                observer.finish_request()

    def _request_observed(
        self, method: str, url: str, observer: PhaseObservation | None,
        response_field: str | None, **kwargs: Any,
    ) -> httpx.Response:
        # Keep the request counter and port-specific cookie jars atomic across
        # callers sharing this engagement client.
        with self._request_lock:
            try:
                response = self._request_unlocked(method, url, **kwargs)
            except Exception as exc:
                if observer is not None:
                    outcome = OutcomeCode.INDETERMINATE
                    try:
                        candidate = _request_exception_outcome(exc)
                        if type(candidate) is OutcomeCode:
                            outcome = candidate
                    except Exception:
                        pass
                    finally:
                        observer.record(outcome)
                raise
            if observer is not None:
                observer.record(
                    self._observe_chat_response(
                        response, validate_field=method.upper() == "POST",
                        response_field=response_field,
                    )
                )
            return response

    def _request_unlocked(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        current_method = method.upper()
        current_url = url
        request_origin = canonical_origin(url)
        body_kwargs = dict(kwargs)
        caller_headers = httpx.Headers(body_kwargs.pop("headers", None))
        caller_cookies = body_kwargs.pop("cookies", None)
        caller_auth = body_kwargs.pop("auth", None)
        follow_redirects = body_kwargs.pop("follow_redirects", True)
        if caller_auth is not None:
            raise ValueError("auth flows are unsupported; pass scoped Authorization headers")
        timeout = self._bounded_timeout(body_kwargs.pop("timeout", None))
        extensions = dict(body_kwargs.pop("extensions", None) or {})
        extensions.pop("timeout", None)
        history: list[httpx.Response] = []

        for redirect_count in range(self.profile.budget.max_redirects + 1):
            if self._requests >= self.profile.budget.max_requests:
                raise RequestBudgetExceeded("request budget exhausted")
            self._requests += 1

            current_origin = canonical_origin(current_url)
            original_approved = (
                current_origin == request_origin
                and current_origin in self.profile.approved_origins
            )
            headers = httpx.Headers(self.profile.headers_for(current_url))
            if original_approved:
                headers.update(caller_headers)
            if redirect_count:
                for name in self._FRAMING_HEADERS:
                    headers.pop(name, None)

            # HTTPX's ordinary cookie jar is host-scoped and ignores ports. Keep it
            # empty; caller cookies are attached only to the approved request origin.
            self._client.cookies.clear()
            cookies = self._origin_cookies.get(current_origin)
            if original_approved:
                if caller_cookies is not None:
                    cookies = caller_cookies
            request_url = httpx.URL(current_url)
            if current_origin not in self.profile.approved_origins:
                request_url = request_url.copy_with(username="", password="")
            request = self._client.build_request(
                current_method,
                request_url,
                headers=headers,
                cookies=cookies,
                timeout=timeout,
                extensions=extensions,
                **body_kwargs,
            )
            streamed = self._client.send(
                request, stream=True, follow_redirects=False,
            )
            chunks: list[bytes] = []
            try:
                for chunk in _bounded_body_chunks(
                    streamed, self.profile.budget.max_response_bytes
                ):
                    chunks.append(chunk)
            finally:
                streamed.close()
                if current_origin in self._origin_cookies:
                    self._origin_cookies[current_origin].extract_cookies(streamed)
                self._client.cookies.clear()

            # The original response retains its wire headers, request, timing,
            # extensions, and downloaded-byte count. Cache the bounded decoded
            # body so HTTPX will not decode it a second time on later access.
            streamed._content = b"".join(chunks)
            response = streamed
            if (
                not follow_redirects
                or response.status_code not in self._REDIRECT_STATUSES
                or "location" not in response.headers
            ):
                response.history = history
                return response
            if redirect_count == self.profile.budget.max_redirects:
                raise httpx.TooManyRedirects("redirect budget exhausted", request=response.request)
            history.append(response)
            current_url = urljoin(str(response.url), response.headers["location"])
            redirected_method = "GET" if (
                (response.status_code in {302, 303} and current_method != "HEAD")
                or (response.status_code == 301 and current_method == "POST")
            ) else current_method
            if redirected_method == "GET" and redirected_method != current_method:
                for key in ("json", "content", "data", "files"):
                    body_kwargs.pop(key, None)
            current_method = redirected_method
            body_kwargs.pop("params", None)
        raise AssertionError("redirect loop escaped budget")

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("HEAD", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def options(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("OPTIONS", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OriginBoundClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
