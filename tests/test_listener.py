"""Tests for the credential capture listener and TLS helper."""

import http.client
import errno
import json
import os
import socket
import stat
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import ragdrag.core.listener as listener
from ragdrag.core.listener import (
    CapturedRequest, Credential, OpenSSLUnavailable, format_request_output,
    generate_self_signed_cert, write_capture,
)


def secret_capture(value="real-secret"):
    return CapturedRequest(
        timestamp="2026-09-29T12:00:00+00:00", source_ip="127.0.0.1",
        method="POST", path="/capture", query_string=f"token={value}&page=2",
        headers={"Authorization": f"Bearer {value}", "User-Agent": "safe-agent"},
        body=f'{{"password":"{value}"}}',
        credentials=[Credential("header", "Authorization", f"Bearer {value}")],
    )


def start_test_server(tmp_path, **kwargs):
    server = listener.create_listener_server("127.0.0.1", 0, str(tmp_path / "capture.jsonl"), **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_test_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, method="POST", headers=None, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    connection.putrequest(method, "/capture?token=wire-secret")
    for key, value in (headers or {}).items():
        connection.putheader(key, value)
    connection.endheaders(body)
    response = connection.getresponse()
    status = response.status
    response.read()
    connection.close()
    return status


def test_default_capture_and_terminal_withhold_secrets_and_controls():
    capture = secret_capture()
    capture.path = "/\x1b[2J\rforged\nline\x7f\x85"
    capture.headers["User-Agent"] = "agent\x1b[31m\rforged"
    data = capture.to_dict()
    rendered = format_request_output(capture)
    assert "real-secret" not in json.dumps(data)
    assert "real-secret" not in rendered
    assert data["headers"]["Authorization"] == "<redacted>"
    assert data["headers"]["User-Agent"] == "<redacted>"
    assert data["body"] == "<redacted>"
    assert data["credentials"][0]["value"] == "<redacted>"
    assert "token=real-secret" not in data["query_string"]
    untrusted_rendered = rendered
    for trusted_color in ("\x1b[91m", "\x1b[1m", "\x1b[0m", "\x1b[2m"):
        untrusted_rendered = untrusted_rendered.replace(trusted_color, "")
    for control in ("\x1b", "\r", "\x7f", "\x85"):
        assert control not in untrusted_rendered
    assert "\\x1b[2J" in rendered
    assert "\\x0dforged\\x0aline" in rendered


def test_unrecognized_header_value_is_withheld_by_default():
    capture = secret_capture()
    capture.headers["User-Agent"] = "opaque-secret-no-pattern"
    capture.credentials.clear()
    assert "opaque-secret-no-pattern" not in json.dumps(capture.to_dict())
    assert "opaque-secret-no-pattern" not in format_request_output(capture)


def test_known_secret_in_header_names_is_redacted_without_key_loss(tmp_path):
    capture = secret_capture("known-secret")
    capture.credentials.append(Credential("pattern", "token", "known-secret"))
    capture.headers = {
        "X-known-secret": "one",
        "X-<redacted>": "two",
        "Other-known-secret": "three",
    }
    first = capture.to_dict()
    second = capture.to_dict()
    assert first == second
    assert len(first["headers"]) == 3
    assert len(set(first["headers"])) == 3
    assert "known-secret" not in json.dumps(first)
    path = tmp_path / "capture.jsonl"
    write_capture(capture, path)
    assert "known-secret" not in path.read_text()
    write_capture(capture, path, store_raw=True)
    assert "X-known-secret" in json.loads(path.read_text().splitlines()[1])["headers"]


@pytest.mark.parametrize("secret", ["<redacted>", "redacted", "safe-key-1"])
def test_default_capture_marker_never_reintroduces_known_secret(secret):
    capture = secret_capture(secret)
    capture.credentials.append(Credential("pattern", "token", secret))
    capture.headers = {f"X-{secret}": "opaque"}
    data = capture.to_dict()
    assert secret not in json.dumps(data)
    assert len(data["headers"]) == 1


@pytest.mark.parametrize("secret", ["<redacted>", "redacted"])
def test_known_marker_secret_stays_out_of_terminal_and_default_jsonl(tmp_path, secret):
    capture = secret_capture(secret)
    capture.credentials.append(Credential("pattern", "token", secret))
    rendered = format_request_output(capture)
    assert secret not in rendered
    path = tmp_path / "capture.jsonl"
    write_capture(capture, path)
    assert secret not in path.read_text()
    write_capture(capture, path, store_raw=True)
    assert secret in path.read_text().splitlines()[1]


def test_compact_json_separator_credential_fails_closed_before_output(tmp_path):
    secret = '\",\"'
    capture = secret_capture(secret)
    capture.credentials.append(Credential("pattern", "token", secret))
    path = tmp_path / "capture.jsonl"
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        capture.to_dict()
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        format_request_output(capture)
    with pytest.raises(OSError, match="^Capture storage failed$"):
        write_capture(capture, path)
    assert not path.exists()
    write_capture(capture, path, store_raw=True)
    assert secret in path.read_text()


def test_terminal_trusted_label_collision_fails_closed_before_render():
    capture = secret_capture("CAPTURE")
    capture.credentials.append(Credential("pattern", "token", "CAPTURE"))
    assert "CAPTURE" not in json.dumps(capture.to_dict())
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        format_request_output(capture)


def test_listener_rejects_unrenderable_capture_before_file_or_terminal_output(tmp_path, capsys):
    server, thread = start_test_server(tmp_path)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/capture?token=CAPTURE")
        response = connection.getresponse()
        assert response.status == 500
        response.read()
        connection.close()
        assert not (tmp_path / "capture.jsonl").exists()
        output = capsys.readouterr()
        assert "CAPTURE" not in output.out
        assert "Traceback" not in output.err
    finally:
        stop_test_server(server, thread)


@pytest.mark.parametrize("character,escaped", [
    ("\u202a", "\\u202a"), ("\u202b", "\\u202b"),
    ("\u202c", "\\u202c"), ("\u202d", "\\u202d"),
    ("\u202e", "\\u202e"), ("\u2066", "\\u2066"),
    ("\u2067", "\\u2067"), ("\u2068", "\\u2068"),
    ("\u2069", "\\u2069"), ("\u200e", "\\u200e"),
    ("\u200f", "\\u200f"), ("\u061c", "\\u061c"),
    ("\u2028", "\\u2028"), ("\u2029", "\\u2029"),
    ("\u2060", "\\u2060"), ("\ud800", "\\ud800"),
    ("\U000e0001", "\\U000e0001"),
    ("\t", "\\x09"), ("\r", "\\x0d"), ("\n", "\\x0a"),
])
def test_terminal_safe_escapes_unicode_format_and_line_controls(character, escaped):
    assert listener._terminal_safe("A" + character + "B") == "A" + escaped + "B"


def test_terminal_output_keeps_only_application_ansi_with_bidi_path():
    capture = secret_capture()
    capture.path = "/before\u202eafter\u2028line"
    rendered = format_request_output(capture)
    assert "\u202e" not in rendered
    assert "\u2028" not in rendered
    assert "\\u202eafter\\u2028line" in rendered


def test_public_output_rejects_hostile_credential_without_property_access(tmp_path):
    seen = []

    class HostileCredential:
        @property
        def value(self):
            seen.append("called")
            return "secret"

    capture = secret_capture()
    capture.credentials = [HostileCredential()]
    for output in (capture.to_dict, lambda: format_request_output(capture)):
        with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
            output()
    path = tmp_path / "capture.jsonl"
    with pytest.raises(OSError, match="^Capture storage failed$"):
        write_capture(capture, path)
    assert seen == []
    assert not path.exists()


@pytest.mark.parametrize("field,value", [
    ("path", b"/bytes"), ("headers", []),
    ("headers", {"X": object()}), ("credentials", ()),
    ("credentials", [object()]),
])
def test_public_output_rejects_malformed_shape_with_fixed_error(field, value):
    capture = secret_capture()
    setattr(capture, field, value)
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        capture.to_dict()
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        format_request_output(capture)


def test_public_output_rejects_subclass_and_container_hooks_without_calling_them():
    seen = []

    class HostileCapture(CapturedRequest):
        def __getattribute__(self, name):
            seen.append(name)
            raise RuntimeError("caller hook")

    base = secret_capture()
    hostile = HostileCapture(**vars(base))
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        CapturedRequest.to_dict(hostile)
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        format_request_output(hostile)

    class HostileHeaders(dict):
        def items(self):
            seen.append("items")
            raise RuntimeError("caller hook")

    base.headers = HostileHeaders(base.headers)
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        base.to_dict()
    assert seen == []


def test_exact_capture_instance_shadow_cannot_replace_trusted_serializer(tmp_path):
    capture = secret_capture()
    calls = []

    def hostile_serializer(*args, **kwargs):
        calls.append("called")
        raise RuntimeError("sensitive hook detail")

    capture.to_dict = hostile_serializer
    assert "real-secret" not in format_request_output(capture)
    path = tmp_path / "capture.jsonl"
    write_capture(capture, path)
    assert "real-secret" not in path.read_text()
    assert calls == []


@pytest.mark.parametrize("field", [
    "timestamp", "source_ip", "method", "path", "query_string",
    "headers", "body", "credentials",
])
def test_missing_capture_attribute_has_fixed_public_errors(tmp_path, field):
    capture = secret_capture()
    delattr(capture, field)
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        CapturedRequest.to_dict(capture)
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        format_request_output(capture)
    path = tmp_path / "capture.jsonl"
    with pytest.raises(OSError, match="^Capture storage failed$"):
        write_capture(capture, path)
    assert not path.exists()


@pytest.mark.parametrize("field", ["source", "key", "value"])
def test_missing_credential_attribute_has_fixed_public_errors(tmp_path, field):
    capture = secret_capture()
    delattr(capture.credentials[0], field)
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        CapturedRequest.to_dict(capture)
    with pytest.raises(ValueError, match="^Invalid captured request\\.$"):
        format_request_output(capture)
    path = tmp_path / "capture.jsonl"
    with pytest.raises(OSError, match="^Capture storage failed$"):
        write_capture(capture, path)
    assert not path.exists()


def test_raw_storage_requires_exact_boolean_opt_in(tmp_path):
    path = tmp_path / "capture.jsonl"
    with pytest.raises(OSError, match="^Capture storage failed$"):
        write_capture(secret_capture(), path, store_raw=1)
    assert not path.exists()


def test_hostile_path_conversion_cannot_reflect_internal_error():
    class HostilePath:
        def __fspath__(self):
            raise RuntimeError("sensitive filesystem detail")

    with pytest.raises(OSError, match="^Capture storage failed$"):
        write_capture(secret_capture(), HostilePath())


def test_raw_opt_in_and_owner_only_append(tmp_path):
    path = tmp_path / "capture.jsonl"
    capture = secret_capture()
    write_capture(capture, path)
    write_capture(capture, path, store_raw=True)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert "real-secret" not in json.dumps(lines[0])
    assert lines[1]["body"] == capture.body
    assert lines[1]["query_string"] == capture.query_string
    assert lines[1]["credentials"][0]["value"] == "Bearer real-secret"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "directory"])
def test_capture_append_rejects_aliases_and_nonregular_targets(tmp_path, alias):
    original = tmp_path / "original"
    original.write_text("untouched")
    target = tmp_path / "capture.jsonl"
    if alias == "symlink":
        target.symlink_to(original)
    elif alias == "hardlink":
        os.link(original, target)
    else:
        target.mkdir()
    with pytest.raises(OSError, match="Capture storage failed"):
        write_capture(secret_capture(), target, store_raw=True)
    assert original.read_text() == "untouched"


def test_capture_append_rejects_parent_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError, match="Capture storage failed"):
        write_capture(secret_capture(), alias / "capture.jsonl")
    assert not (real / "capture.jsonl").exists()


def test_capture_fifo_fails_without_waiting_for_a_reader(tmp_path):
    target = tmp_path / "capture.jsonl"
    os.mkfifo(target)
    errors = []

    def attempt():
        try:
            write_capture(secret_capture(), target)
        except OSError as error:
            errors.append(str(error))

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(timeout=0.2)
    assert not worker.is_alive(), "FIFO capture blocked waiting for a reader"
    assert errors == ["Capture storage failed"]


def test_concurrent_capture_append_keeps_complete_lines(tmp_path):
    path = tmp_path / "capture.jsonl"
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda i: write_capture(secret_capture(f"secret-{i}"), path, store_raw=True), range(80)))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 80
    assert {line["body"] for line in lines} == {
        f'{{"password":"secret-{i}"}}' for i in range(80)
    }


@pytest.mark.parametrize("store_raw", [False, True])
def test_malformed_capture_fails_with_fixed_error_before_creating_file(tmp_path, store_raw):
    path = tmp_path / "capture.jsonl"
    capture = secret_capture()
    capture.headers["X-Bad"] = object()
    with pytest.raises(OSError, match="^Capture storage failed$"):
        write_capture(capture, path, store_raw=store_raw)
    assert not path.exists()


@pytest.mark.parametrize("method,headers,want", [
    ("POST", {}, 411),
    ("POST", {"Content-Length": "bad"}, 400),
    ("POST", {"Content-Length": "-1"}, 400),
    ("POST", {"Transfer-Encoding": "chunked"}, 400),
    ("POST", {"Content-Length": "5"}, 413),
    ("GET", {"Content-Length": "5"}, 413),
])
def test_request_framing_rejects_before_body_read(tmp_path, method, headers, want):
    server, thread = start_test_server(tmp_path, max_body_bytes=4, max_concurrency=2)
    try:
        assert request(server, method, headers) == want
        assert not (tmp_path / "capture.jsonl").exists()
    finally:
        stop_test_server(server, thread)


def test_duplicate_content_length_is_rejected(tmp_path):
    server, thread = start_test_server(tmp_path, max_body_bytes=4, max_concurrency=2)
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.putrequest("POST", "/")
        connection.putheader("Content-Length", "0")
        connection.putheader("Content-Length", "0")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 400
        assert response.getheader("Connection") == "close"
        response.read()
        connection.close()
    finally:
        stop_test_server(server, thread)


def test_truncated_declared_body_is_not_captured(tmp_path):
    server, thread = start_test_server(tmp_path, max_body_bytes=4)
    try:
        sock = socket.create_connection(("127.0.0.1", server.server_port), timeout=2)
        sock.sendall(b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 4\r\n\r\nx")
        sock.shutdown(socket.SHUT_WR)
        response = sock.recv(1024)
        assert b"400 Bad Request" in response
        assert not (tmp_path / "capture.jsonl").exists()
        sock.close()
    finally:
        stop_test_server(server, thread)


def test_saturation_returns_503_without_worker(tmp_path):
    server = listener.create_listener_server("127.0.0.1", 0, str(tmp_path / "capture.jsonl"), max_concurrency=1)
    assert server._slots.acquire(blocking=False)
    server_side, client_side = socket.socketpair()
    try:
        server.process_request(server_side, ("127.0.0.1", 12345))
        assert b"503 Service Unavailable" in client_side.recv(1024)
        assert not server._slots.acquire(blocking=False)
    finally:
        server._slots.release()
        server_side.close()
        client_side.close()
        server.server_close()


def test_worker_slot_returns_after_normal_requests(tmp_path):
    server, thread = start_test_server(tmp_path, max_concurrency=1)
    try:
        assert request(server, headers={"Content-Length": "0"}) == 200
        assert request(server, headers={"Content-Length": "0"}) == 200
    finally:
        stop_test_server(server, thread)


def test_worker_slot_returns_after_handler_baseexception(tmp_path):
    server = listener.create_listener_server("127.0.0.1", 0, str(tmp_path / "capture.jsonl"), max_concurrency=1)
    server_side, client_side = socket.socketpair()
    assert server._slots.acquire(blocking=False)
    try:
        with patch.object(server, "finish_request", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                server.process_request_thread(server_side, ("127.0.0.1", 12345))
        assert server._slots.acquire(blocking=False)
        server._slots.release()
    finally:
        server_side.close()
        client_side.close()
        server.server_close()


def test_worker_slot_returns_when_thread_cannot_start(tmp_path):
    server = listener.create_listener_server("127.0.0.1", 0, str(tmp_path / "capture.jsonl"), max_concurrency=1)
    server_side, client_side = socket.socketpair()
    try:
        with patch("threading.Thread.start", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                server.process_request(server_side, ("127.0.0.1", 12345))
        assert server._slots.acquire(blocking=False)
        server._slots.release()
    finally:
        server_side.close()
        client_side.close()
        server.server_close()


def test_direct_listener_api_refuses_public_bind_before_socket_creation():
    with pytest.raises(ValueError, match="--allow-public"):
        listener.create_listener_server("0.0.0.0", 0)


@pytest.mark.parametrize("ack", [1, "false", object(), None])
def test_direct_listener_api_requires_exact_boolean_ack_before_constructor(ack):
    with patch.object(listener.BoundedThreadingHTTPServer, "__init__", side_effect=AssertionError("constructed")) as init:
        with pytest.raises(ValueError, match="invalid listener acknowledgement"):
            listener.create_listener_server("0.0.0.0", 0, allow_public=ack)
    init.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "4", 65])
def test_direct_bounded_server_constructor_rejects_invalid_limit_before_socket(limit):
    with patch.object(listener.ThreadingHTTPServer, "__init__", side_effect=AssertionError("socket created")) as init:
        with pytest.raises(ValueError, match="invalid listener limits"):
            listener.BoundedThreadingHTTPServer(("127.0.0.1", 0), listener._CaptureHandler,
                                                max_concurrency=limit)
    init.assert_not_called()


def test_direct_bounded_server_constructor_accepts_valid_boundary_limits():
    for limit in (1, 64):
        server = listener.BoundedThreadingHTTPServer(
            ("127.0.0.1", 0), listener._CaptureHandler, max_concurrency=limit,
        )
        try:
            assert server._slots.acquire(blocking=False)
            server._slots.release()
        finally:
            server.server_close()


def test_initial_capture_fstat_failure_closes_descriptor_and_allows_later_append(tmp_path):
    path = tmp_path / "capture.jsonl"
    path.write_text("prior\n")
    real_open = os.open
    real_fstat = os.fstat
    opened = []
    tripped = False

    def tracked_open(target, *args, **kwargs):
        fd = real_open(target, *args, **kwargs)
        if target == path.name:
            opened.append(fd)
        return fd

    def fail_initial_stat(fd):
        nonlocal tripped
        if opened and fd == opened[-1] and not tripped:
            tripped = True
            raise OSError(errno.EIO, "synthetic stat failure")
        return real_fstat(fd)

    with patch.object(listener.os, "open", side_effect=tracked_open):
        with patch.object(listener.os, "fstat", side_effect=fail_initial_stat):
            with pytest.raises(OSError, match="^Capture storage failed$"):
                write_capture(secret_capture(), path)
    assert opened
    with pytest.raises(OSError) as error:
        real_fstat(opened[-1])
    assert error.value.errno == errno.EBADF
    assert path.read_text() == "prior\n"
    write_capture(secret_capture(), path)
    assert len(path.read_text().splitlines()) == 2


@pytest.mark.parametrize("failure", [
    "initial-fstat", "nonregular", "fchmod", "final-directory-close",
])
def test_capture_descriptor_failure_closes_each_owned_fd_once(failure):
    closed = []
    regular = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_nlink=1)
    nonregular = SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_nlink=1)

    def inspected(fd):
        if failure == "initial-fstat":
            raise OSError(errno.EIO, "synthetic stat failure")
        return nonregular if failure == "nonregular" else regular

    def chmod(fd, mode):
        if failure == "fchmod":
            raise OSError(errno.EIO, "synthetic chmod failure")

    def close(fd):
        closed.append(fd)
        if failure == "final-directory-close" and fd == 100:
            raise OSError(errno.EIO, "close released fd then reported failure")

    with patch.object(listener.os, "open", side_effect=[100, 101]):
        with patch.object(listener.os, "fstat", side_effect=inspected):
            with patch.object(listener.os, "fchmod", side_effect=chmod):
                with patch.object(listener.os, "close", side_effect=close):
                    with pytest.raises(OSError, match="^Capture storage failed$"):
                        write_capture(secret_capture(), "capture.jsonl")
    assert sorted(closed) == [100, 101]


def test_capture_append_close_failure_attempts_descriptor_once():
    closed = []

    def close(fd):
        closed.append(fd)
        raise OSError(errno.EIO, "close released fd then reported failure")

    with patch.object(listener, "_open_capture_file", return_value=101):
        with patch.object(listener.fcntl, "flock"):
            with patch.object(listener.os, "fstat", return_value=SimpleNamespace(st_size=0)):
                with patch.object(listener.os, "write", side_effect=lambda fd, data: len(data)):
                    with patch.object(listener.os, "close", side_effect=close):
                        with pytest.raises(OSError, match="^Capture storage failed$"):
                            write_capture(secret_capture(), "capture.jsonl")
    assert closed == [101]


def test_tls_still_wraps_local_listener_socket(tmp_path):
    with patch("ragdrag.core.listener.generate_self_signed_cert", return_value=("cert", "key")):
        with patch("ragdrag.core.listener.ssl.SSLContext") as context_type:
            with patch.object(listener.BoundedThreadingHTTPServer, "serve_forever"):
                context = context_type.return_value
                context.wrap_socket.side_effect = lambda original, server_side: original
                listener.start_listener("127.0.0.1", 0, str(tmp_path / "capture.jsonl"), tls=True)
    context.load_cert_chain.assert_called_once_with("cert", "key")
    assert context.wrap_socket.call_args.kwargs["server_side"] is True


def test_tls_setup_failure_closes_bound_listener(tmp_path):
    with patch.object(listener.BoundedThreadingHTTPServer, "server_close", autospec=True) as close:
        with patch("ragdrag.core.listener.generate_self_signed_cert", side_effect=OpenSSLUnavailable("missing")):
            with pytest.raises(OpenSSLUnavailable, match="missing"):
                listener.start_listener("127.0.0.1", 0, str(tmp_path / "capture.jsonl"), tls=True)
    assert close.call_count == 1


def test_listener_banner_escapes_untrusted_output_path(capsys):
    with patch.object(listener.BoundedThreadingHTTPServer, "serve_forever"):
        listener.start_listener("127.0.0.1", 0, "capture\x1b[2J\rforged.jsonl")
    shown = capsys.readouterr().out
    assert "\x1b" not in shown
    assert "\r" not in shown
    assert "capture\\x1b[2J\\x0dforged.jsonl" in shown


class TestOpenSSLWrapping:
    def test_missing_openssl_raises_friendly_error(self, tmp_path):
        """If openssl is not on PATH, the error should guide the operator, not raise FileNotFoundError."""
        with patch("ragdrag.core.listener.subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(OpenSSLUnavailable, match="openssl not found"):
                generate_self_signed_cert(cert_dir=str(tmp_path))

    def test_openssl_failure_surfaces_stderr(self, tmp_path):
        """If openssl fails (e.g., permission denied), stderr should be surfaced."""
        err = subprocess.CalledProcessError(returncode=1, cmd="openssl", stderr=b"no write permission")
        with patch("ragdrag.core.listener.subprocess.run", side_effect=err):
            with pytest.raises(OpenSSLUnavailable, match="no write permission"):
                generate_self_signed_cert(cert_dir=str(tmp_path))

    def test_openssl_success_returns_paths(self, tmp_path):
        """Happy path: successful openssl run returns cert/key paths in cert_dir."""
        with patch("ragdrag.core.listener.subprocess.run") as m:
            m.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            cert_path, key_path = generate_self_signed_cert(cert_dir=str(tmp_path))
            assert cert_path.endswith("cert.pem")
            assert key_path.endswith("key.pem")
            assert str(tmp_path) in cert_path
            assert str(tmp_path) in key_path
