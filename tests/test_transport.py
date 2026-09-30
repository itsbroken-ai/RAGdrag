"""Origin scoping and bounded HTTP transport behavior."""

import gzip
import zlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError, Event

import httpx
import pytest
from click.testing import CliRunner

from ragdrag.cli import cli
from ragdrag.engine.models import ExitCode, OutcomeCode, RequestBudget, RunResult
from ragdrag.engine.phases import EngagementOutcome
from ragdrag.engine.profile import TargetProfile
from ragdrag.engine.transport import OriginBoundClient, RequestBudgetExceeded, ResponseTooLarge
from ragdrag.utils.http_client import build_async_client, build_client


class ControlledStream(httpx.SyncByteStream):
    """Record reads and closure without relying on HTTPX's byte chunker."""

    def __init__(self, steps):
        self.steps = steps
        self.read_count = 0
        self.closed = False

    def __iter__(self):
        for step in self.steps:
            self.read_count += 1
            if isinstance(step, Exception):
                raise step
            yield step

    def close(self):
        self.closed = True


def test_alternate_port_drops_scoped_headers_and_cookie():
    seen = []
    transport = httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200, json={}))
    profile = TargetProfile.from_cli(
        "http://example.test/chat",
        headers={"Authorization": "Bearer secret", "X-Api-Key": "key"},
        cookie="sid=secret",
    )
    with OriginBoundClient(profile, transport=transport) as client:
        client.get("http://example.test:6333/collections")
    assert "authorization" not in seen[0].headers
    assert "x-api-key" not in seen[0].headers
    assert "cookie" not in seen[0].headers


def test_cross_origin_redirect_drops_scoped_headers():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "example.test":
            return httpx.Response(302, headers={"Location": "https://other.test/final"})
        return httpx.Response(200, json={"ok": True})

    profile = TargetProfile.from_cli("https://example.test/chat", headers={"Authorization": "Bearer secret"})
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.get(profile.target_url, headers={"X-CSRF-Token": "request-secret"})
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert seen[0].headers["x-csrf-token"] == "request-secret"
    assert "authorization" not in seen[1].headers
    assert "x-csrf-token" not in seen[1].headers


def test_redirect_to_independently_approved_origin_uses_only_its_credentials():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "chat.test":
            return httpx.Response(302, headers={"Location": "https://vectors.test:6333/data"})
        return httpx.Response(200, text="ok")

    profile = TargetProfile.from_cli(
        "https://chat.test/chat",
        headers={"Authorization": "Bearer chat"},
        additional_origin_headers={"https://vectors.test:6333": {"X-Api-Key": "vector-key"}},
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.get(profile.target_url, headers={"X-CSRF-Token": "chat-only"})
    assert seen[1].headers["x-api-key"] == "vector-key"
    assert "authorization" not in seen[1].headers
    assert "x-csrf-token" not in seen[1].headers


def test_request_budget_stops_before_extra_request():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, text="ok")

    profile = TargetProfile.from_cli("https://example.test", budget=RequestBudget(max_requests=1))
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.get(profile.target_url)
        with pytest.raises(RequestBudgetExceeded):
            client.get(profile.target_url)
    assert calls == 1


def test_redirect_consumes_shared_request_budget():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"Location": "/next"})

    profile = TargetProfile.from_cli("https://example.test", budget=RequestBudget(max_requests=1))
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RequestBudgetExceeded):
            client.get(profile.target_url)
    assert len(seen) == 1


def test_concurrent_callers_cannot_spend_same_final_request_slot():
    barrier = Barrier(2)
    seen = []

    class RacingBudget:
        timeout_seconds = 2.0
        max_response_bytes = 100
        max_redirects = 0
        max_concurrency = 2

        @property
        def max_requests(self):
            try:
                barrier.wait(timeout=0.2)
            except BrokenBarrierError:
                pass
            return 1

    profile = TargetProfile.from_cli("https://example.test", budget=RacingBudget())
    transport = httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200))
    with OriginBoundClient(profile, transport=transport) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = [
                future.exception()
                for future in [pool.submit(client.get, profile.target_url) for _ in range(2)]
            ]
    assert len(seen) == 1
    assert sum(isinstance(outcome, RequestBudgetExceeded) for outcome in outcomes) == 1


def test_inflight_request_reports_to_its_starting_observer_not_next_phase():
    entered, release = Event(), Event()
    calls = []

    def respond(request):
        calls.append(request)
        entered.set()
        assert release.wait(2)
        return httpx.Response(404)

    profile = TargetProfile.from_cli("https://example.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(respond)) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="phase observation has in-flight requests"):
                with client.observe_phase(None) as first:
                    future = pool.submit(client.get, profile.target_url)
                    assert entered.wait(2)
            try:
                with client.observe_phase(None) as second:
                    release.set()
                    assert future.result(timeout=2).status_code == 404
            finally:
                release.set()
    assert len(calls) == 1
    assert list(first) == [OutcomeCode.INVALID_TARGET]
    assert list(second) == []


def test_inflight_response_uses_its_starting_response_field():
    entered, release = Event(), Event()

    def respond(request):
        entered.set()
        assert release.wait(2)
        return httpx.Response(200, json={"answer": "safe"})

    profile = TargetProfile.from_cli("https://example.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(respond)) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="phase observation has in-flight requests"):
                with client.observe_phase("answer") as first:
                    future = pool.submit(client.post, profile.target_url, json={"query": "safe"})
                    assert entered.wait(2)
            try:
                with client.observe_phase("different") as second:
                    release.set()
                    assert future.result(timeout=2).status_code == 200
            finally:
                release.set()
    assert list(first) == []
    assert list(second) == []


def test_request_captures_response_field_before_completion():
    entered, release = Event(), Event()

    def respond(request):
        entered.set()
        assert release.wait(2)
        return httpx.Response(200, json={"answer": "safe"})

    profile = TargetProfile.from_cli("https://example.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(respond)) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="phase observation has in-flight requests"):
                with client.observe_phase("answer") as observation:
                    future = pool.submit(client.post, profile.target_url, json={"query": "safe"})
                    assert entered.wait(2)
                    observation._response_field = "different"
            try:
                release.set()
                assert future.result(timeout=2).status_code == 200
            finally:
                release.set()
    assert list(observation) == []


def test_phase_observations_store_only_distinct_failures_after_budget_exhaustion():
    calls = []
    profile = TargetProfile.from_cli(
        "https://example.test/chat", budget=RequestBudget(max_requests=1),
    )
    with OriginBoundClient(
        profile, transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200)),
    ) as client:
        with client.observe_phase(None) as observations:
            client.get(profile.target_url)
            for _ in range(128):
                with pytest.raises(RequestBudgetExceeded):
                    client.get(profile.target_url)
    assert len(calls) == 1
    assert client.requests_used == 1
    assert list(observations) == [OutcomeCode.INDETERMINATE]


def test_phase_observation_rejects_nested_scope_and_resets_sequentially():
    profile = TargetProfile.from_cli("https://example.test/chat")
    responses = iter([httpx.Response(401), httpx.Response(200)])
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: next(responses))) as client:
        with client.observe_phase(None) as first:
            with pytest.raises(RuntimeError, match="phase observation is already active"):
                with client.observe_phase(None):
                    pass
            client.get(profile.target_url)
        with client.observe_phase(None) as second:
            client.get(profile.target_url)
    assert list(first) == [OutcomeCode.AUTHENTICATION_REQUIRED]
    assert list(second) == []


@pytest.mark.parametrize("error_type,expected", [
    (httpx.ConnectError, OutcomeCode.UNREACHABLE),
    (httpx.ReadTimeout, OutcomeCode.UNREACHABLE),
    (httpx.RemoteProtocolError, OutcomeCode.UNREACHABLE),
    (httpx.ProxyError, OutcomeCode.UNREACHABLE),
    (httpx.DecodingError, OutcomeCode.UNSUPPORTED_RESPONSE),
    (httpx.TooManyRedirects, OutcomeCode.INDETERMINATE),
])
def test_observed_exact_httpx_exception_mapping_is_preserved(error_type, expected):
    profile = TargetProfile.from_cli("https://example.test/chat")

    def respond(request):
        raise error_type("synthetic request failure", request=request)

    with OriginBoundClient(profile, transport=httpx.MockTransport(respond)) as client:
        with client.observe_phase(None) as observation:
            with pytest.raises(error_type):
                client.get(profile.target_url)
    assert client.requests_used == 1
    assert list(observation) == [expected]


@pytest.mark.parametrize("classifier_failure", ["raises", "returns-untrusted"])
def test_failed_request_keeps_canonical_outcome_if_classifier_fails(monkeypatch, classifier_failure):
    original = httpx.HTTPError("synthetic original failure")

    class UntrustedOutcome:
        def __str__(self):
            raise AssertionError("untrusted outcome rendered")

        def __repr__(self):
            raise AssertionError("untrusted outcome represented")

    def classify(exc):
        assert exc is original
        if classifier_failure == "raises":
            raise httpx.HTTPError("synthetic nested classifier failure")
        return UntrustedOutcome()

    monkeypatch.setattr("ragdrag.engine.transport._request_exception_outcome", classify)
    profile = TargetProfile.from_cli("https://example.test/chat")
    with OriginBoundClient(
        profile, transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(original)),
    ) as client:
        with client.observe_phase(None) as observation:
            with pytest.raises(httpx.HTTPError) as caught:
                client.get(profile.target_url)
    assert caught.value is original
    assert client.requests_used == 1
    assert list(observation) == [OutcomeCode.INDETERMINATE]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_classifier_interruption_is_not_swallowed_and_request_is_observed(monkeypatch, interrupt):
    def classify(exc):
        raise interrupt()

    monkeypatch.setattr("ragdrag.engine.transport._request_exception_outcome", classify)
    profile = TargetProfile.from_cli("https://example.test/chat")
    with OriginBoundClient(
        profile,
        transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.HTTPError("synthetic"))),
    ) as client:
        with client.observe_phase(None) as observation:
            with pytest.raises(interrupt):
                client.get(profile.target_url)
    assert client.requests_used == 1
    assert list(observation) == [OutcomeCode.INDETERMINATE]


def test_response_limit_raises_typed_error():
    profile = TargetProfile.from_cli(
        "https://example.test",
        budget=RequestBudget(max_response_bytes=4),
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"12345"))
    with OriginBoundClient(profile, transport=transport) as client:
        with pytest.raises(ResponseTooLarge):
            client.get(profile.target_url)


def test_response_at_exact_limit_succeeds():
    profile = TargetProfile.from_cli(
        "https://example.test", budget=RequestBudget(max_response_bytes=4)
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"1234"))
    with OriginBoundClient(profile, transport=transport) as client:
        assert client.get(profile.target_url).content == b"1234"


def test_compat_builder_requires_target_for_scoped_headers():
    with pytest.raises(ValueError, match="target is required"):
        build_client(headers={"Authorization": "Bearer secret"})


def test_compat_builder_requires_target_for_cookie():
    with pytest.raises(ValueError, match="target is required"):
        build_client(cookie="sid=secret")


def test_deprecated_async_builder_rejects_unscoped_headers():
    with pytest.raises(ValueError, match="unscoped headers"):
        build_async_client(headers={"Authorization": "Bearer secret"})


def test_request_level_cookie_and_headers_do_not_follow_alternate_port_redirect():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"Location": "https://example.test:8443/next"})
        return httpx.Response(200, text="ok")

    profile = TargetProfile.from_cli("https://example.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.get(
            profile.target_url,
            headers={"X-CSRF-Token": "secret"},
            cookies={"sid": "secret"},
        )
    assert seen[0].headers["x-csrf-token"] == "secret"
    assert "sid=secret" in seen[0].headers["cookie"]
    assert "x-csrf-token" not in seen[1].headers
    assert "cookie" not in seen[1].headers


def test_operator_cookie_jar_is_scoped_to_primary_origin():
    seen = []
    profile = TargetProfile.from_cli("https://example.test/chat")
    transport = httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200))
    with OriginBoundClient(profile, transport=transport) as client:
        client.cookies.set("sid", "secret")
        client.get("https://example.test:8443/data")
        client.get(profile.target_url)
    assert "cookie" not in seen[0].headers
    assert seen[1].headers["cookie"] == "sid=secret"


def test_ipv6_credentials_follow_only_effective_port():
    seen = []
    profile = TargetProfile.from_cli("http://[::1]/", headers={"Authorization": "Bearer v6"})
    transport = httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200))
    with OriginBoundClient(profile, transport=transport) as client:
        client.get("http://[::1]:8080/data")
        client.get("http://[::1]:80/data")
    assert "authorization" not in seen[0].headers
    assert seen[1].headers["authorization"] == "Bearer v6"


def test_redirect_limit_stops_after_allowed_count():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"Location": "/next"})

    profile = TargetProfile.from_cli(
        "https://example.test", budget=RequestBudget(max_redirects=1)
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.TooManyRedirects):
            client.get(profile.target_url)
    assert len(seen) == 2


def test_per_call_redirect_opt_out_preserves_httpx_behavior():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"Location": "/next"})

    profile = TargetProfile.from_cli("https://example.test")
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        response = client.get(profile.target_url, follow_redirects=False)
    assert response.status_code == 302
    assert len(seen) == 1


def test_caller_timeout_cannot_exceed_profile_cap():
    observed = []

    def handler(request):
        observed.append(request.extensions["timeout"])
        return httpx.Response(200)

    profile = TargetProfile.from_cli(
        "https://example.test", budget=RequestBudget(timeout_seconds=2)
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.get(profile.target_url, timeout=30)
        client.get(profile.target_url, timeout=None)
    assert all(all(value <= 2 for value in timeout.values()) for timeout in observed)


def test_post_redirect_does_not_copy_transport_framing_headers():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"Location": "/next"})
        return httpx.Response(200)

    profile = TargetProfile.from_cli("https://example.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.post(
            profile.target_url,
            content=b"data",
            headers={
                "Content-Type": "text/custom",
                "Content-Length": "4",
                "Host": "custom.test",
                "Accept-Encoding": "identity",
            },
        )
    assert seen[1].method == "GET"
    assert "content-type" not in seen[1].headers
    assert "content-length" not in seen[1].headers
    assert seen[1].headers["host"] == "example.test"
    assert seen[1].headers["accept-encoding"] != "identity"


def test_auth_flow_cannot_send_unbudgeted_challenge_retry():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(401, headers={"WWW-Authenticate": 'Digest realm="test", nonce="abc", qop="auth"'})
        return httpx.Response(200)

    profile = TargetProfile.from_cli(
        "https://example.test", budget=RequestBudget(max_requests=1)
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="auth"):
            client.get(profile.target_url, auth=httpx.DigestAuth("name", "secret"))
    assert seen == []


def test_response_limit_stops_streaming_and_closes_source():
    class Source(httpx.SyncByteStream):
        def __init__(self):
            self.read_count = 0
            self.closed = False

        def __iter__(self):
            for chunk in (b"1234", b"5", b"unread"):
                self.read_count += 1
                yield chunk

        def close(self):
            self.closed = True

    source = Source()
    profile = TargetProfile.from_cli(
        "https://example.test", budget=RequestBudget(max_response_bytes=4)
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=source))
    with OriginBoundClient(profile, transport=transport) as client:
        with pytest.raises(ResponseTooLarge):
            client.get(profile.target_url)
    assert source.read_count == 2
    assert source.closed


def test_timeout_extension_cannot_override_profile_cap():
    observed = []

    def handler(request):
        observed.append(request.extensions["timeout"])
        return httpx.Response(200)

    profile = TargetProfile.from_cli(
        "https://example.test", budget=RequestBudget(timeout_seconds=2)
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.get(
            profile.target_url,
            extensions={"timeout": {"connect": 30, "read": 30, "write": 30, "pool": 30}},
        )
    assert all(value <= 2 for value in observed[0].values())


def test_url_userinfo_cannot_attach_auth_to_unapproved_origin():
    seen = []
    profile = TargetProfile.from_cli("https://example.test")
    transport = httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200))
    with OriginBoundClient(profile, transport=transport) as client:
        client.get("https://name:secret@other.test/data")
    assert "authorization" not in seen[0].headers


def test_response_cookie_stays_on_its_exact_approved_origin():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(200, headers={"Set-Cookie": "sid=server; Path=/"})
        return httpx.Response(200)

    profile = TargetProfile.from_cli("https://example.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.get(profile.target_url)
        client.get("https://example.test:8443/data")
        client.get(profile.target_url)
    assert "cookie" not in seen[1].headers
    assert seen[2].headers["cookie"] == "sid=server"


@pytest.mark.parametrize("command", ["fingerprint", "exfiltrate"])
def test_cli_scoped_headers_supply_target_to_engine(monkeypatch, command):
    observed = {}

    def capture_run(profile, phases, options):
        observed.update(profile=profile, phases=phases)
        now = "2026-09-29T12:00:00+00:00"
        return EngagementOutcome(
            RunResult("run-1", profile.target_url, now, now, [], ExitCode.CLEAN), [], [],
        )

    monkeypatch.setattr("ragdrag.cli.run_engagement", capture_run)
    result = CliRunner().invoke(
        cli, [command, "--target", "https://example.test/chat", "--header", "X-CSRF: secret"]
    )
    assert result.exit_code == 0
    assert observed["profile"].target_url == "https://example.test/chat"
    assert observed["profile"].headers_for("https://example.test/chat")["X-CSRF"] == "secret"
    assert observed["profile"].headers_for("https://other.test/chat") == {}
    assert observed["phases"] == ["R1" if command == "fingerprint" else "R3"]
    assert "secret" not in result.output


@pytest.mark.parametrize("encoding,compress", [("gzip", gzip.compress), ("deflate", zlib.compress)])
def test_compressed_json_at_exact_decoded_limit_is_buffered_once(encoding, compress):
    payload = b'{"ok": true}'
    wire = compress(payload)
    source = ControlledStream([wire[:3], wire[3:]])
    profile = TargetProfile.from_cli(
        "https://example.test/data", budget=RequestBudget(max_response_bytes=len(payload))
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Encoding": encoding, "Content-Type": "application/json"},
            stream=source,
            extensions={"http_version": b"HTTP/2"},
        )
    )
    with OriginBoundClient(profile, transport=transport) as client:
        response = client.get(profile.target_url)
    assert response.content == payload
    assert response.json() == {"ok": True}
    assert response.request.url == httpx.URL(profile.target_url)
    assert response.http_version == "HTTP/2"
    assert response.num_bytes_downloaded == len(wire)
    assert response.elapsed.total_seconds() >= 0
    assert source.closed


def test_stacked_gzip_then_deflate_decodes_small_json():
    payload = b'{"stacked": true}'
    wire = zlib.compress(gzip.compress(payload))
    source = ControlledStream([wire])
    profile = TargetProfile.from_cli(
        "https://example.test/data", budget=RequestBudget(max_response_bytes=len(payload))
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"Content-Encoding": "gzip, deflate"}, stream=source
        )
    )
    with OriginBoundClient(profile, transport=transport) as client:
        response = client.get(profile.target_url)
    assert response.json() == {"stacked": True}
    assert source.closed


@pytest.mark.parametrize("encoding,compress", [("gzip", gzip.compress), ("deflate", zlib.compress)])
def test_compressed_overflow_caps_decoder_output_before_materialization(monkeypatch, encoding, compress):
    wire = compress(b"A" * 1_000_000)
    source = ControlledStream([wire, AssertionError("read after oversized decoded body")])
    observed_limits = []
    original_factory = zlib.decompressobj

    class BoundedProbe:
        def __init__(self, inner):
            self.inner = inner

        def decompress(self, data, max_length=0):
            observed_limits.append(max_length)
            assert 0 < max_length <= 65, "decoder output was not bounded"
            return self.inner.decompress(data, max_length)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    monkeypatch.setattr(
        zlib, "decompressobj", lambda *args, **kwargs: BoundedProbe(original_factory(*args, **kwargs))
    )
    profile = TargetProfile.from_cli(
        "https://example.test/data", budget=RequestBudget(max_response_bytes=64)
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"Content-Encoding": encoding}, stream=source
        )
    )
    with OriginBoundClient(profile, transport=transport) as client:
        with pytest.raises(ResponseTooLarge):
            client.get(profile.target_url)
    assert observed_limits
    assert source.read_count == 1
    assert source.closed


def test_stacked_encoding_overflow_bounds_each_decoder(monkeypatch):
    wire = zlib.compress(gzip.compress(b"A" * 1_000_000))
    source = ControlledStream([wire, AssertionError("read after oversized decoded body")])
    observed_limits = []
    original_factory = zlib.decompressobj

    class BoundedProbe:
        def __init__(self, inner):
            self.inner = inner

        def decompress(self, data, max_length=0):
            observed_limits.append(max_length)
            assert 0 < max_length <= 2_226, "stacked decoder output was not bounded"
            return self.inner.decompress(data, max_length)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    monkeypatch.setattr(
        zlib, "decompressobj", lambda *args, **kwargs: BoundedProbe(original_factory(*args, **kwargs))
    )
    profile = TargetProfile.from_cli(
        "https://example.test/data", budget=RequestBudget(max_response_bytes=1_200)
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"Content-Encoding": "gzip, deflate"}, stream=source
        )
    )
    with OriginBoundClient(profile, transport=transport) as client:
        with pytest.raises(ResponseTooLarge):
            client.get(profile.target_url)
    assert len(observed_limits) >= 2
    assert any(length <= 1_201 for length in observed_limits)
    assert source.read_count == 1
    assert source.closed


def test_unsupported_encoding_fails_closed_before_reading_body():
    source = ControlledStream([b"anything"])
    profile = TargetProfile.from_cli("https://example.test/data")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"Content-Encoding": "br"}, stream=source)
    )
    with OriginBoundClient(profile, transport=transport) as client:
        with pytest.raises(httpx.DecodingError, match="unsupported"):
            client.get(profile.target_url)
    assert source.read_count == 0
    assert source.closed


@pytest.mark.parametrize("limit", [65_536, 65_537])
def test_first_excess_byte_wins_over_later_source_error(limit):
    exact_chunks = [b"A" * 65_536]
    if limit == 65_537:
        exact_chunks.append(b"B")
    source = ControlledStream(
        [*exact_chunks, b"X", httpx.ReadTimeout("source failed after first excess byte")]
    )
    profile = TargetProfile.from_cli(
        "https://example.test/data", budget=RequestBudget(max_response_bytes=limit)
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=source))
    with OriginBoundClient(profile, transport=transport) as client:
        with pytest.raises(ResponseTooLarge):
            client.get(profile.target_url)
    assert source.read_count == len(exact_chunks) + 1
    assert source.closed


def test_redirected_compressed_response_retains_request_and_history():
    payload = b'{"ok": true}'
    source = ControlledStream([gzip.compress(payload)])
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"Location": "/final"})
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=source)

    profile = TargetProfile.from_cli("https://example.test/start")
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        response = client.get(profile.target_url)
    assert response.json() == {"ok": True}
    assert response.request is seen[1]
    assert len(response.history) == 1
    assert response.history[0].request is seen[0]
    assert response.history[0].status_code == 302
    assert source.closed


@pytest.mark.parametrize(
    "status,method,expected",
    [
        (301, "GET", "GET"), (301, "HEAD", "HEAD"), (301, "POST", "GET"),
        (301, "PUT", "PUT"), (301, "DELETE", "DELETE"), (301, "OPTIONS", "OPTIONS"),
        (302, "GET", "GET"), (302, "HEAD", "HEAD"), (302, "POST", "GET"),
        (302, "PUT", "GET"), (302, "DELETE", "GET"), (302, "OPTIONS", "GET"),
        (303, "GET", "GET"), (303, "HEAD", "HEAD"), (303, "POST", "GET"),
        (303, "PUT", "GET"), (303, "DELETE", "GET"), (303, "OPTIONS", "GET"),
        (307, "GET", "GET"), (307, "HEAD", "HEAD"), (307, "POST", "POST"),
        (307, "PUT", "PUT"), (307, "DELETE", "DELETE"), (307, "OPTIONS", "OPTIONS"),
        (308, "GET", "GET"), (308, "HEAD", "HEAD"), (308, "POST", "POST"),
        (308, "PUT", "PUT"), (308, "DELETE", "DELETE"), (308, "OPTIONS", "OPTIONS"),
    ],
)
def test_redirect_method_and_body_match_httpx_policy(status, method, expected):
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(status, headers={"Location": "/next"})
        return httpx.Response(200)

    profile = TargetProfile.from_cli("https://example.test/start")
    body = b"body" if method not in {"GET", "HEAD"} else b""
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.request(method, profile.target_url, content=body)
    assert [request.method for request in seen] == [method, expected]
    assert seen[1].content == (body if expected == method else b"")
    if expected != method:
        assert "content-length" not in seen[1].headers


@pytest.mark.parametrize(
    "encoding,wire",
    [
        ("gzip", gzip.compress(b"A")),
        ("deflate", zlib.compress(b"A")),
        ("gzip, deflate", zlib.compress(gzip.compress(b"A"))),
    ],
    ids=["gzip", "deflate", "stacked-outer"],
)
@pytest.mark.parametrize("later_chunk", [False, True], ids=["same-chunk", "later-chunk"])
def test_compressed_trailing_data_precedes_later_source_error(encoding, wire, later_chunk):
    if later_chunk:
        steps = [wire, b"!", httpx.ReadTimeout("read after trailing data")]
        expected_reads = 2
    else:
        steps = [wire + b"!", httpx.ReadTimeout("read after trailing data")]
        expected_reads = 1
    source = ControlledStream(steps)
    profile = TargetProfile.from_cli(
        "https://example.test/data", budget=RequestBudget(max_response_bytes=1)
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"Content-Encoding": encoding}, stream=source)
    )
    with OriginBoundClient(profile, transport=transport) as client:
        with pytest.raises(httpx.DecodingError, match="trailing"):
            client.get(profile.target_url)
    assert source.read_count == expected_reads
    assert source.closed


def test_stacked_inner_trailing_chunk_precedes_later_source_error():
    compressor = zlib.compressobj()
    first = compressor.compress(gzip.compress(b"A")) + compressor.flush(zlib.Z_SYNC_FLUSH)
    second = compressor.compress(b"!") + compressor.flush()
    source = ControlledStream([first, second, httpx.ReadTimeout("read after inner trailing data")])
    profile = TargetProfile.from_cli(
        "https://example.test/data", budget=RequestBudget(max_response_bytes=1)
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"Content-Encoding": "gzip, deflate"}, stream=source
        )
    )
    with OriginBoundClient(profile, transport=transport) as client:
        with pytest.raises(httpx.DecodingError, match="trailing"):
            client.get(profile.target_url)
    assert source.read_count == 2
    assert source.closed


def test_same_chunk_trailing_input_does_not_grow_decoder_unused_data(monkeypatch):
    wire = gzip.compress(b"A") + b"!" * 100_000
    source = ControlledStream([wire, httpx.ReadTimeout("read after trailing data")])
    unused_lengths = []
    original_factory = zlib.decompressobj

    class TrackingDecoder:
        def __init__(self, inner):
            self.inner = inner

        def decompress(self, data, max_length=0):
            result = self.inner.decompress(data, max_length)
            unused_lengths.append(len(self.inner.unused_data))
            return result

        def __getattr__(self, name):
            return getattr(self.inner, name)

    monkeypatch.setattr(
        zlib, "decompressobj", lambda *args, **kwargs: TrackingDecoder(original_factory(*args, **kwargs))
    )
    profile = TargetProfile.from_cli("https://example.test/data")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=source)
    )
    with OriginBoundClient(profile, transport=transport) as client:
        with pytest.raises(httpx.DecodingError, match="trailing"):
            client.get(profile.target_url)
    assert max(unused_lengths) <= 65_536
    assert source.read_count == 1
    assert source.closed


@pytest.mark.parametrize("status", [302, 303])
def test_get_body_survives_redirect_with_httpx_framing(status):
    def capture(client_factory):
        seen = []

        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                return httpx.Response(status, headers={"Location": "/next"})
            return httpx.Response(200)

        with client_factory(httpx.MockTransport(handler)) as client:
            client.request("GET", "https://example.test/start", content=b"body")
        return seen

    reference = capture(lambda transport: httpx.Client(transport=transport, follow_redirects=True))
    profile = TargetProfile.from_cli("https://example.test/start")
    observed = capture(lambda transport: OriginBoundClient(profile, transport=transport))
    assert [(r.method, r.content, r.headers.get("content-length")) for r in observed] == [
        (r.method, r.content, r.headers.get("content-length")) for r in reference
    ]
