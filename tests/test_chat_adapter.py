"""Chat response and explicit conversation-state contracts."""

import json
from copy import deepcopy

import httpx
import pytest

from ragdrag.adapters.chat import (
    ChatAdapter,
    ChatConfig,
    ChatResponseError,
    ConversationState,
    extract_response_text,
)
from ragdrag.engine.models import OutcomeCode
from ragdrag.engine.profile import TargetProfile
from ragdrag.engine.transport import OriginBoundClient


@pytest.mark.parametrize("response", [
    httpx.Response(200, text="not-json"),
    httpx.Response(200, json=["not", "an", "object"]),
    httpx.Response(200, json={"other": "missing"}),
    httpx.Response(200, json={"answer": 42}),
    httpx.Response(200, json={"answer": None}),
    httpx.Response(200, json={"answer": {"text": "nested"}}),
])
def test_configured_response_field_rejects_unsupported_shapes(response):
    with pytest.raises(ChatResponseError) as error:
        extract_response_text(response, "answer")
    assert error.value.outcome is OutcomeCode.UNSUPPORTED_RESPONSE


def test_unconfigured_response_field_returns_raw_text():
    assert extract_response_text(httpx.Response(200, text="plain reply"), None) == "plain reply"


def test_second_turn_contains_prior_user_and_assistant_messages():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": f"reply-{len(bodies)}"})

    profile = TargetProfile.from_cli(
        "https://example.test/chat", history_field="messages", response_field="answer",
    )
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig.from_profile(profile))
        adapter.send("first", state)
        adapter.send("second", state)
    assert bodies[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply-1"},
        {"role": "user", "content": "second"},
    ]
    assert state.mechanism == "history"


def test_state_is_detached_from_request_body_and_failed_reply_does_not_advance_it():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(200, json={"answer": "first reply"})
        return httpx.Response(200, text="invalid")

    profile = TargetProfile.from_cli(
        "https://example.test/chat", history_field="messages", response_field="answer",
    )
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig.from_profile(profile))
        adapter.send("first", state)
        with pytest.raises(ChatResponseError):
            adapter.send("second", state)
    assert bodies[0]["messages"] == [{"role": "user", "content": "first"}]
    assert state.messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
    ]


def test_history_wins_over_session_and_cookie_without_copying_headers():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "reply"}, headers={"set-cookie": "sid=new"})

    profile = TargetProfile.from_cli(
        "https://example.test/chat", headers={"Authorization": "Bearer secret"},
        response_field="answer", history_field="messages", session_field="session",
        session_id="operator-session",
    )
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        reply = ChatAdapter(client, ChatConfig.from_profile(profile)).send("first", state)
    assert reply.state_mechanism == state.mechanism == "history"
    assert bodies == [{
        "query": "first", "messages": [{"role": "user", "content": "first"}],
        "session": "operator-session",
    }]
    assert "secret" not in repr(state)


def test_session_wins_over_new_cookie():
    profile = TargetProfile.from_cli(
        "https://example.test/chat", response_field="answer",
        session_field="session", session_id="one",
    )
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"answer": "reply"}, headers={"set-cookie": "sid=new"})
    )) as client:
        reply = ChatAdapter(client, ChatConfig.from_profile(profile)).send("first", state)
    assert reply.state_mechanism == "session"


def test_no_history_session_or_cookie_reports_no_state_mechanism():
    profile = TargetProfile.from_cli("https://example.test/chat", response_field="answer")
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"answer": "reply"})
    )) as client:
        ChatAdapter(client, ChatConfig.from_profile(profile)).send("first", state)
    assert state.mechanism is None


@pytest.mark.parametrize("state_mode", ["history", "session"])
def test_duplicate_cookie_names_already_present_do_not_block_configured_state(state_mode):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "reply"})

    profile = TargetProfile.from_cli(
        "https://example.test/chat", response_field="answer",
        history_field="messages" if state_mode == "history" else None,
        session_field="session" if state_mode == "session" else None,
        session_id="configured-session" if state_mode == "session" else None,
    )
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.cookies.set("sid", "root-cookie-secret", domain="example.test", path="/")
        client.cookies.set("sid", "chat-cookie-secret", domain="example.test", path="/chat")
        reply = ChatAdapter(client, ChatConfig.from_profile(profile)).send("first", state)
    assert reply.state_mechanism == state_mode
    assert state.messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    assert "cookie-secret" not in json.dumps(bodies)
    assert "cookie-secret" not in repr(state)


@pytest.mark.parametrize("state_mode", [None, "history", "session"])
def test_duplicate_cookie_name_introduced_by_response_commits_state_once(state_mode):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "reply"}, headers={
            "set-cookie": "sid=chat-cookie-secret; Path=/chat",
        })

    profile = TargetProfile.from_cli(
        "https://example.test/chat", response_field="answer",
        history_field="messages" if state_mode == "history" else None,
        session_field="session" if state_mode == "session" else None,
        session_id="configured-session" if state_mode == "session" else None,
    )
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        client.cookies.set("sid", "root-cookie-secret", domain="example.test", path="/")
        reply = ChatAdapter(client, ChatConfig.from_profile(profile)).send("first", state)
    assert reply.state_mechanism == (state_mode or "cookie")
    assert state.messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    assert "cookie-secret" not in json.dumps(bodies)
    assert "cookie-secret" not in repr(state)


@pytest.mark.parametrize("set_cookie", [
    "sid=; Max-Age=0; Path=/",
    "sid=unused-cookie-secret; Path=/unused",
    "sid=secure-cookie-secret; Secure; Path=/",
])
def test_unusable_cookie_changes_do_not_establish_state(set_cookie):
    profile = TargetProfile.from_cli("http://example.test/chat", response_field="answer")
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"answer": "reply"}, headers={
            "set-cookie": set_cookie,
        })
    )) as client:
        if "Max-Age=0" in set_cookie:
            client.cookies.set("sid", "old-cookie-secret", domain="example.test", path="/")
        reply = ChatAdapter(client, ChatConfig.from_profile(profile)).send("first", state)
    assert reply.state_mechanism is None
    assert state.mechanism is None
    assert "cookie-secret" not in repr(state)


def test_cookie_for_unrelated_domain_does_not_establish_state():
    profile = TargetProfile.from_cli("https://example.test/chat", response_field="answer")
    state = ConversationState()
    def handler(request):
        client.cookies.set("sid", "other-cookie-secret", domain="other.test", path="/")
        return httpx.Response(200, json={"answer": "reply"})

    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        reply = ChatAdapter(client, ChatConfig.from_profile(profile)).send("first", state)
    assert reply.state_mechanism is None


def test_cookie_from_primary_origin_cannot_establish_state_for_other_transport_origin():
    profile = TargetProfile.from_cli(
        "https://example.test/chat", response_field="answer",
        additional_origin_headers={"https://other.test": {}},
    )
    state = ConversationState()

    def handler(request):
        client.cookies.set("sid", "primary-cookie-secret", domain="other.test", path="/")
        return httpx.Response(200, json={"answer": "reply"})

    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig("https://other.test/chat", response_field="answer"))
        reply = adapter.send("first", state)
    assert reply.state_mechanism is None
    assert "cookie-secret" not in repr(state)


def test_cookie_is_reused_then_later_loss_clears_mechanism():
    seen_cookies = []

    def handler(request):
        seen_cookies.append(request.headers.get("cookie"))
        headers = {}
        if len(seen_cookies) == 1:
            headers["set-cookie"] = "sid=reusable-cookie-secret; Path=/"
        if len(seen_cookies) == 3:
            headers["set-cookie"] = "sid=; Max-Age=0; Path=/"
        return httpx.Response(200, json={"answer": "reply"}, headers=headers)

    profile = TargetProfile.from_cli("https://example.test/chat", response_field="answer")
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig.from_profile(profile))
        first = adapter.send("first", state)
        second = adapter.send("second", state)
        third = adapter.send("third", state)
        fourth = adapter.send("fourth", state)
    assert first.state_mechanism == "cookie"
    assert seen_cookies[1] == "sid=reusable-cookie-secret"
    assert second.state_mechanism == "cookie"
    assert seen_cookies[2] == "sid=reusable-cookie-secret"
    assert third.state_mechanism is None
    assert fourth.state_mechanism is None
    assert "cookie-secret" not in repr(state)


@pytest.mark.parametrize("final_url", [
    "https://other.test/reply",
    "https://example.test/reply",
    "http://example.test/chat",
])
def test_redirected_reply_without_usable_final_cookie_has_no_cookie_mechanism(final_url):
    final_cookies = []
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if str(request.url) == "https://example.test/chat":
            return httpx.Response(307, headers={
                "location": final_url,
                "set-cookie": "sid=entry-cookie-secret; Secure; Path=/chat",
            })
        final_cookies.append(request.headers.get("cookie"))
        return httpx.Response(200, json={"answer": "reply"})

    profile = TargetProfile.from_cli("https://example.test/chat", response_field="answer")
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig.from_profile(profile))
        first = adapter.send("first", state)
        second = adapter.send("second", state)
    assert first.state_mechanism is None
    assert second.state_mechanism is None
    assert final_cookies == [None, None]
    assert "cookie-secret" not in json.dumps(bodies)
    assert "cookie-secret" not in repr(state)


def test_later_cross_origin_redirect_clears_prior_cookie_mechanism():
    final_cookies = []
    turns = 0

    def handler(request):
        nonlocal turns
        if str(request.url) == "https://example.test/chat":
            turns += 1
            if turns == 1:
                return httpx.Response(200, json={"answer": "reply"}, headers={
                    "set-cookie": "sid=valid-cookie-secret; Secure; Path=/",
                })
            return httpx.Response(307, headers={"location": "https://other.test/reply"})
        final_cookies.append(request.headers.get("cookie"))
        return httpx.Response(200, json={"answer": "reply"})

    profile = TargetProfile.from_cli("https://example.test/chat", response_field="answer")
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig.from_profile(profile))
        first = adapter.send("first", state)
        second = adapter.send("second", state)
    assert first.state_mechanism == "cookie"
    assert second.state_mechanism is None
    assert final_cookies == [None]
    assert "cookie-secret" not in repr(state)


def test_same_origin_redirect_with_cookie_for_entry_and_final_retains_mechanism():
    seen_cookies = []

    def handler(request):
        seen_cookies.append((str(request.url), request.headers.get("cookie")))
        if request.url.path == "/chat":
            return httpx.Response(307, headers={
                "location": "/reply",
                "set-cookie": "sid=shared-cookie-secret; Secure; Path=/",
            })
        return httpx.Response(200, json={"answer": "reply"})

    profile = TargetProfile.from_cli("https://example.test/chat", response_field="answer")
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig.from_profile(profile))
        first = adapter.send("first", state)
        second = adapter.send("second", state)
    assert first.state_mechanism == "cookie"
    assert second.state_mechanism == "cookie"
    assert seen_cookies[1] == ("https://example.test/reply", "sid=shared-cookie-secret")
    assert seen_cookies[2] == ("https://example.test/chat", "sid=shared-cookie-secret")
    assert seen_cookies[3] == ("https://example.test/reply", "sid=shared-cookie-secret")
    assert "cookie-secret" not in repr(state)


@pytest.mark.parametrize("secondary_origin", [False, True])
def test_configured_cookie_header_suppresses_new_jar_cookie_at_final_reply(secondary_origin):
    entry = "https://entry.test/chat" if secondary_origin else "https://example.test/chat"
    final = "https://final.test/reply" if secondary_origin else "https://example.test/reply"
    seen_final_cookies = []
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if str(request.url) == entry:
            return httpx.Response(307, headers={"location": final})
        seen_final_cookies.append(request.headers.get("cookie"))
        return httpx.Response(200, json={"answer": "reply"}, headers={
            "set-cookie": "sid=shadowed-cookie-secret; Secure; Path=/",
        })

    if secondary_origin:
        profile = TargetProfile.from_cli(
            entry, response_field="answer",
            additional_origin_headers={"https://final.test": {"Cookie": "operator=static"}},
        )
    else:
        profile = TargetProfile.from_cli(entry, response_field="answer", cookie="operator=static")
    state = ConversationState()
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig.from_profile(profile))
        first = adapter.send("first", state)
        second = adapter.send("second", state)
    assert seen_final_cookies == ["operator=static", "operator=static"]
    assert first.state_mechanism is None
    assert second.state_mechanism is None
    assert "cookie-secret" not in json.dumps(bodies)
    assert "cookie-secret" not in repr(state)


def test_configured_cookie_header_clears_previously_tracked_jar_cookie():
    target = "https://example.test/chat"
    state = ConversationState()
    first_profile = TargetProfile.from_cli(target, response_field="answer")
    with OriginBoundClient(first_profile, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"answer": "reply"}, headers={
            "set-cookie": "sid=tracked-cookie-secret; Secure; Path=/",
        })
    )) as client:
        first = ChatAdapter(client, ChatConfig.from_profile(first_profile)).send("first", state)
    assert first.state_mechanism == "cookie"

    seen_cookies = []

    def handler(request):
        seen_cookies.append(request.headers.get("cookie"))
        return httpx.Response(200, json={"answer": "reply"})

    second_profile = TargetProfile.from_cli(target, response_field="answer", cookie="operator=static")
    with OriginBoundClient(second_profile, transport=httpx.MockTransport(handler)) as client:
        client.cookies.set("sid", "tracked-cookie-secret", domain="example.test", path="/")
        second = ChatAdapter(client, ChatConfig.from_profile(second_profile)).send("second", state)
    assert seen_cookies == ["operator=static"]
    assert second.state_mechanism is None
    assert state.mechanism is None
    assert "cookie-secret" not in repr(state)


COOKIE_ROUTES = [
    pytest.param(("https://entry.test/reply",), id="direct"),
    pytest.param(("https://entry.test/chat", "https://entry.test/reply"), id="same-origin"),
    pytest.param(("https://entry.test/chat", "https://final.test/reply"), id="cross-origin"),
    pytest.param((
        "https://entry.test/chat", "https://middle.test/hop", "https://final.test/reply",
    ), id="multi-hop-secondary"),
    pytest.param((
        "https://entry.test/chat", "https://middle.test/hop", "https://entry.test/reply",
    ), id="multi-hop-primary"),
]


@pytest.mark.parametrize("route", COOKIE_ROUTES)
@pytest.mark.parametrize("header_name", ["Cookie", "cookie", "cOoKiE"])
@pytest.mark.parametrize("configured_cookie", [None, "", "operator=static"])
def test_generic_cookie_state_uses_final_request_after_redirects(
    route, header_name, configured_cookie,
):
    receipts = []
    bodies = []
    final_receipts = []

    def handler(request):
        receipts.append((str(request.url), request.headers.get("cookie")))
        bodies.append(json.loads(request.content))
        hop = route.index(str(request.url))
        if hop < len(route) - 1:
            return httpx.Response(307, headers={"location": route[hop + 1]})
        final_receipts.append(request.headers.get("cookie"))
        headers = {"set-cookie": "sid=final-cookie-secret; Secure; Path=/"} if (
            len(final_receipts) == 1
        ) else {}
        return httpx.Response(200, json={"answer": "reply"}, headers=headers)

    headers = {} if configured_cookie is None else {header_name: configured_cookie}
    state = ConversationState()
    with httpx.Client(
        headers=headers, follow_redirects=True, transport=httpx.MockTransport(handler),
    ) as client:
        headers_before = client.headers.copy()
        adapter = ChatAdapter(client, ChatConfig(route[0], response_field="answer"))
        replies = [adapter.send(query, state) for query in ("first", "second", "third")]
        assert client.headers == headers_before

    overridden = configured_cookie is not None and len(route) == 1
    expected_receipts = [configured_cookie] * 3 if overridden else [
        None, "sid=final-cookie-secret", "sid=final-cookie-secret",
    ]
    assert final_receipts == expected_receipts
    if configured_cookie is not None:
        assert [cookie for url, cookie in receipts if url == route[0]] == [configured_cookie] * 3
    expected_mechanism = None if overridden else "cookie"
    assert [reply.state_mechanism for reply in replies] == [expected_mechanism] * 3
    assert state.mechanism == expected_mechanism
    assert bool(state._cookie_keys) is (not overridden)
    assert len(state.messages) == 6
    assert "cookie-secret" not in json.dumps(bodies)
    assert "cookie-secret" not in repr(replies)
    assert "cookie-secret" not in repr(vars(state))


@pytest.mark.parametrize("route", COOKIE_ROUTES)
@pytest.mark.parametrize("header_name, configured_cookie", [
    ("Cookie", "operator=static"),
    ("cookie", "operator=static"),
    ("cOoKiE", "operator=static"),
    ("Cookie", ""),
    ("Cookie", "sid=tracked-cookie-secret"),
])
def test_generic_cookie_header_retains_only_state_delivered_by_final_request(
    route, header_name, configured_cookie,
):
    final_receipts = []
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        hop = route.index(str(request.url))
        if hop < len(route) - 1:
            return httpx.Response(307, headers={"location": route[hop + 1]})
        final_receipts.append(request.headers.get("cookie"))
        headers = {"set-cookie": "sid=tracked-cookie-secret; Secure; Path=/"} if (
            len(final_receipts) == 1
        ) else {}
        return httpx.Response(200, json={"answer": "reply"}, headers=headers)

    state = ConversationState()
    with httpx.Client(follow_redirects=True, transport=httpx.MockTransport(handler)) as client:
        first = ChatAdapter(client, ChatConfig(route[-1], response_field="answer")).send(
            "first", state,
        )
        assert first.state_mechanism == "cookie"
        tracked_keys = state._cookie_keys.copy()
        assert tracked_keys
        client.headers[header_name] = configured_cookie
        headers_before = client.headers.copy()
        adapter = ChatAdapter(client, ChatConfig(route[0], response_field="answer"))
        replies = [adapter.send(query, state) for query in ("second", "third")]
        assert client.headers == headers_before

    redirected = len(route) > 1
    expected_cookie = "sid=tracked-cookie-secret" if redirected else configured_cookie
    assert final_receipts == [None, expected_cookie, expected_cookie]
    assert [reply.state_mechanism for reply in replies] == ["cookie" if redirected else None] * 2
    assert state._cookie_keys == (tracked_keys if redirected else set())
    assert len(state.messages) == 6
    assert "cookie-secret" not in json.dumps(bodies)
    assert "cookie-secret" not in repr(replies)
    assert "cookie-secret" not in repr(vars(state))


@pytest.mark.parametrize("failure", ["transport", "response", "precedence"])
def test_generic_redirect_failure_preserves_complete_conversation_state(failure, monkeypatch):
    final_turns = 0

    def handler(request):
        nonlocal final_turns
        if request.url.path == "/chat":
            return httpx.Response(307, headers={"location": "/reply"})
        final_turns += 1
        if final_turns > 1 and failure == "transport":
            raise httpx.ReadError("local failure", request=request)
        answer = 42 if final_turns > 1 and failure == "response" else "reply"
        return httpx.Response(200, json={"answer": answer}, headers={
            "set-cookie": "sid=atomic-cookie-secret; Secure; Path=/",
        })

    state = ConversationState(session_id="existing-session")
    with httpx.Client(follow_redirects=True, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig("https://entry.test/chat", response_field="answer"))
        assert adapter.send("first", state).state_mechanism == "cookie"
        before = deepcopy(vars(state))
        client.headers["Cookie"] = "operator=static"
        if failure == "precedence":
            def fail_precedence(*args):
                raise RuntimeError("local precedence failure")
            monkeypatch.setattr("ragdrag.adapters.chat._cookie_header_overrides_jar", fail_precedence)
        expected_error = {
            "transport": httpx.ReadError, "response": ChatResponseError, "precedence": RuntimeError,
        }[failure]
        with pytest.raises(expected_error):
            adapter.send("second", state)
    assert vars(state) == before
    assert "cookie-secret" not in repr(vars(state))


DIGEST_CHALLENGE = 'Digest realm="local", nonce="local-nonce", algorithm=MD5, qop="auth"'
DIGEST_COOKIE = "sid=digest-cookie-secret"


@pytest.mark.parametrize("prior_state", [False, True], ids=["new", "tracked"])
@pytest.mark.parametrize("fresh_auth", [False, True], ids=["cached-auth", "fresh-auth"])
@pytest.mark.parametrize("header_name", ["Cookie", "cookie", "cOoKiE"])
@pytest.mark.parametrize("configured_cookie", [None, "", "operator=static", DIGEST_COOKIE])
def test_digest_challenge_without_redirect_preserves_configured_cookie_precedence(
    prior_state, fresh_auth, header_name, configured_cookie,
):
    receipts = []
    responses = []
    bodies = []
    successes = 0

    def handler(request):
        nonlocal successes
        body = json.loads(request.content)
        bodies.append(body)
        receipts.append((str(request.url), request.headers.get("cookie")))
        if body["query"] != "bootstrap" and "authorization" not in request.headers:
            # Location on a 401 does not cause HTTPX redirect handling.
            return httpx.Response(401, headers={
                "www-authenticate": DIGEST_CHALLENGE, "location": "/not-a-redirect",
            })
        successes += 1
        response = httpx.Response(200, json={"answer": "reply"}, headers={
            "set-cookie": f"{DIGEST_COOKIE}; Secure; Path=/",
        } if successes == 1 else {})
        responses.append(response)
        return response

    state = ConversationState()
    target = "https://entry.test/reply"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig(target, response_field="answer"))
        if prior_state:
            assert adapter.send("bootstrap", state).state_mechanism == "cookie"
        if configured_cookie is not None:
            client.headers[header_name] = configured_cookie
        headers_before = client.headers.copy()
        client.auth = httpx.DigestAuth("digest-user-secret", "digest-password-secret")
        for turn in range(3):
            if fresh_auth:
                client.auth = httpx.DigestAuth("digest-user-secret", "digest-password-secret")
            receipt_start = len(receipts)
            reply = adapter.send(f"turn-{turn}", state)
            challenged = fresh_auth or turn == 0
            expected_cookie = configured_cookie if configured_cookie is not None else (
                DIGEST_COOKIE if prior_state or turn > 0 else None
            )
            assert receipts[receipt_start:] == [(target, expected_cookie)] * (2 if challenged else 1)
            assert [item.status_code for item in responses[-1].history] == ([401] if challenged else [])
            assert str(responses[-1].url) == target
            assert client.headers == headers_before
            assert len(state.messages) == 2 * (turn + 1 + prior_state)
            for secret in ("cookie-secret", "digest-user-secret", "digest-password-secret"):
                assert secret not in json.dumps(bodies)
                assert secret not in repr(reply)
                assert secret not in repr(vars(state))
            expected = "cookie" if configured_cookie is None else None
            assert reply.state_mechanism == state.mechanism == expected
            assert state._cookie_keys == (
                {("", "entry.test", "/", "sid")} if expected else set()
            )


@pytest.mark.parametrize("route", [
    "redirect", "auth-redirect", "redirect-auth", "redirect-auth-direct",
])
@pytest.mark.parametrize("redirect_status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("prior_state", [False, True], ids=["new", "tracked"])
@pytest.mark.parametrize("fresh_auth", [False, True], ids=["cached-auth", "fresh-auth"])
@pytest.mark.parametrize("configured_cookie", [None, "", "operator=static", DIGEST_COOKIE])
def test_digest_and_redirect_cookie_state_follows_effective_final_leg(
    route, redirect_status, prior_state, fresh_auth, configured_cookie,
):
    receipts = []
    responses = []
    bodies = []
    successes = 0

    def handler(request):
        nonlocal successes
        bodies.append(json.loads(request.content) if request.content else None)
        receipts.append((request.url.path, request.method, request.headers.get("cookie")))
        authenticated = "authorization" in request.headers
        challenge = (
            route == "auth-redirect" and request.url.path == "/chat" and not authenticated
        ) or (
            route in ("redirect-auth", "redirect-auth-direct")
            and request.url.path == "/reply" and not authenticated
        )
        if challenge:
            return httpx.Response(401, headers={"www-authenticate": DIGEST_CHALLENGE})
        if request.url.path == "/chat" and not (
            route == "redirect-auth-direct" and authenticated
        ):
            return httpx.Response(redirect_status, headers={"location": "/reply"})
        successes += 1
        response = httpx.Response(200, json={"answer": "reply"}, headers={
            "set-cookie": f"{DIGEST_COOKIE}; Secure; Path=/",
        } if successes == 1 else {})
        responses.append(response)
        return response

    state = ConversationState()
    with httpx.Client(follow_redirects=True, transport=httpx.MockTransport(handler)) as client:
        if prior_state:
            bootstrap = ChatAdapter(client, ChatConfig("https://entry.test/seed", response_field="answer"))
            assert bootstrap.send("bootstrap", state).state_mechanism == "cookie"
        if configured_cookie is not None:
            client.headers["cOoKiE"] = configured_cookie
        headers_before = client.headers.copy()
        client.auth = httpx.DigestAuth("digest-user-secret", "digest-password-secret")
        adapter = ChatAdapter(client, ChatConfig("https://entry.test/chat", response_field="answer"))
        for turn in range(3):
            if fresh_auth:
                client.auth = httpx.DigestAuth("digest-user-secret", "digest-password-secret")
            receipt_start = len(receipts)
            reply = adapter.send(f"turn-{turn}", state)
            jar_cookie = DIGEST_COOKIE if prior_state or turn > 0 else None
            entry_cookie = configured_cookie if configured_cookie is not None else jar_cookie
            entry = ("/chat", "POST", entry_cookie)
            final = ("/reply", "GET" if redirect_status in (301, 302, 303) else "POST", jar_cookie)
            challenged = route != "redirect" and (fresh_auth or turn == 0)
            if route == "redirect-auth-direct":
                expected_receipts = [entry, final, entry] if challenged else [entry]
                expected_history = [401] if challenged else []
            elif route == "auth-redirect" and challenged:
                expected_receipts = [entry, entry, final]
                expected_history = [401, redirect_status]
            elif route == "redirect-auth" and challenged:
                expected_receipts = [entry, final, entry, final]
                expected_history = [401, redirect_status]
            else:
                expected_receipts = [entry, final]
                expected_history = [redirect_status]
            assert receipts[receipt_start:] == expected_receipts
            assert [item.status_code for item in responses[-1].history] == expected_history
            expected_final_path = "/chat" if route == "redirect-auth-direct" else "/reply"
            assert responses[-1].url.path == expected_final_path
            assert client.headers == headers_before
            assert len(state.messages) == 2 * (turn + 1 + prior_state)
            for secret in ("cookie-secret", "digest-user-secret", "digest-password-secret"):
                assert secret not in json.dumps(bodies)
                assert secret not in repr(reply)
                assert secret not in repr(vars(state))
            overridden = route == "redirect-auth-direct" and configured_cookie is not None
            assert reply.state_mechanism == state.mechanism == (None if overridden else "cookie")
            assert state._cookie_keys == (
                set() if overridden else {("", "entry.test", "/", "sid")}
            )


@pytest.mark.parametrize("redirect", [False, True], ids=["direct", "redirect"])
@pytest.mark.parametrize("failure", ["transport", "response", "precedence"])
def test_digest_failure_preserves_complete_conversation_state(redirect, failure, monkeypatch):
    receipts = []
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        receipts.append((request.url.path, request.headers.get("cookie")))
        if body["query"] != "bootstrap":
            if "authorization" not in request.headers:
                return httpx.Response(401, headers={"www-authenticate": DIGEST_CHALLENGE})
            if redirect and request.url.path == "/chat":
                return httpx.Response(307, headers={"location": "/reply"})
            if failure == "transport":
                raise httpx.ReadError("local failure", request=request)
        answer = 42 if body["query"] != "bootstrap" and failure == "response" else "reply"
        return httpx.Response(200, json={"answer": answer}, headers={
            "set-cookie": f"{DIGEST_COOKIE}; Secure; Path=/",
        })

    state = ConversationState(session_id="existing-session")
    with httpx.Client(follow_redirects=True, transport=httpx.MockTransport(handler)) as client:
        adapter = ChatAdapter(client, ChatConfig("https://entry.test/chat", response_field="answer"))
        assert adapter.send("bootstrap", state).state_mechanism == "cookie"
        before = deepcopy(vars(state))
        client.headers["Cookie"] = "operator=static"
        headers_before = client.headers.copy()
        client.auth = httpx.DigestAuth("digest-user-secret", "digest-password-secret")
        if failure == "precedence":
            def fail_precedence(*args):
                raise RuntimeError("local precedence failure")
            monkeypatch.setattr("ragdrag.adapters.chat._cookie_header_overrides_jar", fail_precedence)
        expected_error = {
            "transport": httpx.ReadError, "response": ChatResponseError, "precedence": RuntimeError,
        }[failure]
        with pytest.raises(expected_error):
            adapter.send("failing", state)
        assert client.headers == headers_before
    assert receipts == [("/chat", None), ("/chat", "operator=static"), ("/chat", "operator=static")] + (
        [("/reply", DIGEST_COOKIE)] if redirect else []
    )
    assert vars(state) == before
    for secret in ("cookie-secret", "digest-user-secret", "digest-password-secret"):
        assert secret not in json.dumps(bodies)
        assert secret not in repr(vars(state))
