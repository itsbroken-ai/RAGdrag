"""Tests for CLI helpers, especially URL validation and normalization."""

import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import h11
import pytest
from click import BadParameter
from click.testing import CliRunner

from ragdrag.cli import _validate_url, cli
from ragdrag.engine.capability import InvalidConfiguration
from ragdrag.engine.models import CleanupState, ExitCode, ImpactLevel, MutationRecord, RunResult
from ragdrag.engine.phases import EngagementOutcome
from ragdrag.engine.runner import EngagementInterrupted
from ragdrag.engine.transport import OriginBoundClient


TARGET = "https://target.test/chat"
CLEANUP = "https://target.test/documents/{id}"
CONTROLS = [
    "--established-control", "baseline",
    "--established-control", "negative-control",
    "--established-control", "cleanup-verification",
]


def outcome(exit_code=ExitCode.CLEAN, status="completed"):
    return EngagementOutcome(
        RunResult("run-1", TARGET, "2026-09-29T12:00:00+00:00",
                  "2026-09-29T12:00:01+00:00", [], exit_code, status=status),
        [], [],
    )


def invoke_with_engine(args, *, result=None):
    calls = []

    def fake_run(profile, phases, options):
        calls.append((profile, phases, options))
        return result or outcome()

    with patch("ragdrag.cli.run_engagement", side_effect=fake_run, create=True):
        invocation = CliRunner().invoke(cli, ["--quiet", *args])
    return invocation, calls


def test_scan_defaults_to_r1_r2_r3_once():
    invocation, calls = invoke_with_engine(["scan", "-t", TARGET])
    assert invocation.exit_code == 0
    assert len(calls) == 1
    assert calls[0][1] == ["R1", "R2", "R3"]
    assert calls[0][2].established_controls == frozenset()


def test_scan_help_and_version_keep_click_behavior():
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--quiet", "scan", "--help"])
    assert help_result.exit_code == 0
    assert "R1,R2,R3" in help_result.output
    assert "--allow-write" in help_result.output
    assert runner.invoke(cli, ["--version"]).exit_code == 0


@pytest.mark.parametrize("phases", ["", ", ", "R7", "R1,R7", "R1,", ",R1", "R1,,R2"])
def test_empty_or_unknown_phases_fail_before_engine(phases):
    invocation, calls = invoke_with_engine(["scan", "-t", TARGET, "-p", phases])
    assert invocation.exit_code == 3
    assert calls == []


@pytest.mark.parametrize("extra,expected", [
    ([], "--allow-write"),
    (["--allow-write"], "--cleanup-url"),
    (["--allow-write", "--cleanup-url", CLEANUP], "--established-control"),
    (["--allow-write", "--cleanup-url", CLEANUP, *CONTROLS[:4]], "--established-control"),
])
def test_mutating_scan_requires_all_explicit_gates(extra, expected):
    invocation, calls = invoke_with_engine(["scan", "-t", TARGET, "-p", "R4", *extra])
    assert invocation.exit_code == 3
    assert expected in invocation.output
    assert calls == []


def test_mutating_scan_passes_exact_asserted_controls():
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "-p", "R4,R5", "--allow-write",
        "--cleanup-url", CLEANUP, *CONTROLS,
    ])
    assert invocation.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0].impact_ceiling is ImpactLevel.MUTATING
    assert calls[0][1] == ["R4", "R5"]
    assert calls[0][2].established_controls == frozenset({
        "baseline", "negative-control", "cleanup-verification",
    })


def test_duplicate_phases_are_dispatched_once_in_requested_order():
    invocation, calls = invoke_with_engine(["scan", "-t", TARGET,
                                            "-p", "R2,R1,R2,R3,R1"])
    assert invocation.exit_code == 0
    assert len(calls) == 1
    assert calls[0][1] == ["R2", "R1", "R3"]


@pytest.mark.parametrize("cleanup", [
    "https://target.test/documents", "https://target.test/documents/{id}?x=1",
    "https://evil.test/documents/{id}", "https://target.test/{id}/documents",
])
def test_unsafe_cleanup_fails_before_engine(cleanup):
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "-p", "R4", "--allow-write",
        "--cleanup-url", cleanup, *CONTROLS,
    ])
    assert invocation.exit_code == 3
    assert calls == []


@pytest.mark.parametrize("command,write_flag", [
    ("scan", ["-p", "R4", "--allow-write"]),
    ("poison", []), ("hijack", []),
])
def test_cleanup_template_2048_is_accepted_and_2049_rejected_before_engagement(
    tmp_path, command, write_flag,
):
    prefix = "https://target.test/docs/"
    suffix = "/{id}"
    valid = prefix + "a" * (2048 - len(prefix) - len(suffix)) + suffix
    invalid = prefix + "a" * (2049 - len(prefix) - len(suffix)) + suffix
    assert len(valid) == 2048
    assert len(invalid) == 2049
    output = tmp_path / "existing.json"
    output.write_text("original")
    base = [command, "-t", TARGET, *write_flag, *CONTROLS, "-o", str(output)]
    rejected, calls = invoke_with_engine([*base, "--cleanup-url", invalid])
    assert rejected.exit_code == 3
    assert calls == []
    assert output.read_text() == "original"
    accepted, calls = invoke_with_engine([*base, "--cleanup-url", valid])
    assert accepted.exit_code == 0
    assert len(calls) == 1
    assert calls[0][2].cleanup_url == valid


@pytest.mark.parametrize("option", [
    ["-H", "Authorization: Bearer secret\r\nX-Other: yes"],
    ["-H", "Cookie: one", "--cookie", "two"],
    ["--cookie", "sid=secret\x1b[31m"],
    ["--timeout", "nan"],
    ["--timeout", "0"],
    ["-t", "https://name:password@target.test/chat"],
])
def test_malformed_options_never_reach_engine_or_echo_secrets(option):
    args = ["scan", "-t", TARGET, *option]
    invocation, calls = invoke_with_engine(args)
    assert invocation.exit_code == 3
    assert calls == []
    assert "secret" not in invocation.output
    assert "password" not in invocation.output
    assert "\x1b" not in invocation.output


@pytest.mark.parametrize("raw", [
    "X-Test: value\r\n", "X-Test: \r\nvalue", "X-Test: \vvalue",
    "X-Test: value\f", "X-Test: value\x00", "X-Test: value\x7f",
    "X-Test: value\x80", "X-Test: café", "X-Test\r: value",
])
def test_raw_header_controls_and_non_ascii_fail_before_engagement(tmp_path, raw):
    output = tmp_path / "existing.json"
    output.write_text("original")
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "-H", raw, "-o", str(output),
    ])
    assert invocation.exit_code == 3
    assert calls == []
    assert output.read_text() == "original"
    assert "\r" not in invocation.output
    assert "\v" not in invocation.output
    assert "\f" not in invocation.output


@pytest.mark.parametrize("option,value", [
    ("--cookie", "sid=café"), ("--cookie", "sid=ok\r\n"),
    ("--cookie", "sid=ok\x7f"), ("--api-key", "café"),
    ("--api-key", "key\r\n"), ("--api-key", "key\x80"),
])
def test_cookie_and_api_key_encoding_rejected_before_engagement(tmp_path, option, value):
    output = tmp_path / "existing.json"
    output.write_text("original")
    command = "poison" if option == "--api-key" else "scan"
    extras = ["--cleanup-url", CLEANUP, *CONTROLS] if command == "poison" else []
    invocation, calls = invoke_with_engine([
        command, "-t", TARGET, option, value, "-o", str(output), *extras,
    ])
    assert invocation.exit_code == 3
    assert calls == []
    assert output.read_text() == "original"


@pytest.mark.parametrize("option,value", [
    ("--cookie", " "), ("--cookie", "\t \t"),
    ("--api-key", " "), ("--api-key", "\t \t"),
])
def test_whitespace_only_cookie_and_api_key_reject_before_engagement(
    tmp_path, option, value,
):
    output = tmp_path / "existing.json"
    output.write_text("original")
    command = "poison" if option == "--api-key" else "scan"
    extras = ["--cleanup-url", CLEANUP, *CONTROLS] if command == "poison" else []
    invocation, calls = invoke_with_engine([
        command, "-t", TARGET, option, value, "-o", str(output), *extras,
    ])
    assert invocation.exit_code == 3
    assert calls == []
    assert output.read_text() == "original"


def test_padded_cookie_and_api_key_are_normalized_for_wire_and_origin_scope():
    invocation, calls = invoke_with_engine([
        "poison", "-t", TARGET, "--ingest-url", "https://ingest.test/documents",
        "--cookie", " \tsid=one two\tthree \t",
        "--api-key", " \tkey one\ttwo \t",
        "--cleanup-url", "https://ingest.test/documents/{id}", *CONTROLS,
    ])
    assert invocation.exit_code == 0
    assert len(calls) == 1
    profile = calls[0][0]
    assert profile.headers_for(TARGET)["Cookie"] == "sid=one two\tthree"
    assert profile.headers_for("https://ingest.test/documents")["X-Api-Key"] == "key one\ttwo"
    seen = []

    def respond(request):
        h11.Request(method=request.method, target=request.url.raw_path,
                    headers=request.headers.raw)
        seen.append((str(request.url), request.headers.get("cookie"),
                     request.headers.get("x-api-key")))
        if request.url.host == "ingest.test":
            return httpx.Response(302, headers={"Location": TARGET})
        return httpx.Response(200)

    with OriginBoundClient(profile, transport=httpx.MockTransport(respond)) as client:
        response = client.get("https://ingest.test/documents", follow_redirects=True)
    assert response.status_code == 200
    assert seen == [
        ("https://ingest.test/documents", None, "key one\ttwo"),
        (TARGET, "sid=one two\tthree", None),
    ]


def test_legal_horizontal_whitespace_and_ascii_header_value_are_preserved():
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "-H", "\tX-Test\t: \tvalue~\t",
    ])
    assert invocation.exit_code == 0
    assert calls[0][0].headers_for(TARGET)["X-Test"] == "value~"


def test_network_options_create_scoped_profile_and_chat_state(tmp_path):
    output = tmp_path / "run.json"
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "-H", "Authorization: Bearer secret",
        "--cookie", "sid=secret", "--timeout", "7", "--max-requests", "250",
        "--no-verify-ssl",
        "--query-field", "prompt", "--response-field", "answer",
        "--history-field", "messages", "--session-field", "conversation_id",
        "--session-id", "session-1", "-o", str(output),
    ], result=outcome(ExitCode.PARTIAL, "partial"))
    assert invocation.exit_code == 2
    assert len(calls) == 1
    profile, _, options = calls[0]
    assert profile.headers_for(TARGET)["Authorization"] == "Bearer secret"
    assert profile.headers_for(TARGET)["Cookie"] == "sid=secret"
    assert profile.headers_for("https://elsewhere.test/chat") == {}
    assert profile.budget.timeout_seconds == 7
    assert profile.budget.max_requests == 250
    assert profile.verify_ssl is False
    assert (profile.query_field, profile.response_field, profile.history_field,
            profile.session_field, profile.session_id) == (
                "prompt", "answer", "messages", "conversation_id", "session-1")
    assert (options.history_field, options.session_field, options.session_id) == (
        "messages", "conversation_id", "session-1")
    assert json.loads(output.read_text())["schema_version"] == "1.0"
    assert "secret" not in invocation.output


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_invalid_max_requests_fails_before_engine(value):
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "--max-requests", value,
    ])
    assert invocation.exit_code == 3
    assert calls == []


@pytest.mark.parametrize("field_name", ["input-text", "answer.text", "résultat", "input text"])
@pytest.mark.parametrize("option,attribute", [
    ("--query-field", "query_field"), ("--response-field", "response_field"),
    ("--history-field", "history_field"), ("--session-field", "session_field"),
])
def test_literal_json_field_names_reach_profile_and_phase_options_unchanged(
    option, attribute, field_name,
):
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, option, field_name,
    ])
    assert invocation.exit_code == 0
    assert len(calls) == 1
    assert getattr(calls[0][0], attribute) == field_name
    assert getattr(calls[0][2], attribute) == field_name


def test_json_field_name_256_utf8_bytes_is_accepted():
    field_name = "é" * 128
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "--query-field", field_name,
    ])
    assert invocation.exit_code == 0
    assert calls[0][0].query_field == field_name


@pytest.mark.parametrize("field_name", ["", "x\x00", "x\x1b", "x\x80", "\ud800", "é" * 128 + "a"])
def test_invalid_or_oversized_json_field_names_reject_before_engagement(tmp_path, field_name):
    output = tmp_path / "existing.json"
    output.write_text("original")
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "--query-field", field_name, "-o", str(output),
    ])
    assert invocation.exit_code == 3
    assert calls == []
    assert output.read_text() == "original"


@pytest.mark.parametrize("command,extra,phase,option,value", [
    ("fingerprint", ["--no-port-scan"], "R1", "scan_ports", False),
    ("probe", ["--depth", "full"], "R2", "deep", True),
    ("exfiltrate", ["--deep"], "R3", "deep", True),
    ("evade", [], "R6", "deep", True),
])
def test_read_only_command_mappings(command, extra, phase, option, value):
    invocation, calls = invoke_with_engine([command, "-t", TARGET, *extra])
    assert invocation.exit_code == 0
    assert len(calls) == 1
    assert calls[0][1] == [phase]
    assert getattr(calls[0][2], option) == value
    assert calls[0][0].impact_ceiling is ImpactLevel.ACTIVE


@pytest.mark.parametrize("command,phase", [
    ("fingerprint", "R1"), ("probe", "R2"), ("exfiltrate", "R3"),
    ("evade", "R6"),
])
def test_every_read_only_command_accepts_scoped_credentials(command, phase):
    invocation, calls = invoke_with_engine([
        command, "-t", TARGET, "-H", "Authorization: Bearer secret",
        "--cookie", "sid=secret", "--history-field", "messages",
        "--session-field", "conversation_id", "--session-id", "one",
    ])
    assert invocation.exit_code == 0
    assert calls[0][1] == [phase]
    assert calls[0][0].headers_for(TARGET)["Authorization"] == "Bearer secret"
    assert calls[0][0].headers_for("https://other.test/") == {}
    assert calls[0][2].session_id == "one"
    assert "secret" not in invocation.output


@pytest.mark.parametrize("command,extra,phase", [
    ("poison", ["--listener", "listener.test"], "R4"),
    ("hijack", ["--callback", "https://callback.test", "--camouflage"], "R5"),
])
def test_write_commands_require_cleanup_and_controls(command, extra, phase):
    rejected, calls = invoke_with_engine([command, "-t", TARGET, *extra])
    assert rejected.exit_code == 3
    assert calls == []
    accepted, calls = invoke_with_engine([
        command, "-t", TARGET, *extra, "--cleanup-url", CLEANUP, *CONTROLS,
    ])
    assert accepted.exit_code == 0
    assert len(calls) == 1
    assert calls[0][1] == [phase]
    assert calls[0][0].impact_ceiling is ImpactLevel.MUTATING


def test_api_key_scopes_to_ingestion_origin_and_preserves_target_auth():
    invocation, calls = invoke_with_engine([
        "poison", "-t", TARGET, "--ingest-url", "https://ingest.test/documents",
        "--api-key", "secret-key", "-H", "Authorization: Bearer target-secret",
        "--cleanup-url", "https://ingest.test/documents/{id}", *CONTROLS,
    ])
    assert invocation.exit_code == 0
    profile, _, options = calls[0]
    assert profile.headers_for(TARGET) == {"Authorization": "Bearer target-secret"}
    assert profile.headers_for("https://ingest.test/documents") == {"X-Api-Key": "secret-key"}
    assert options.ingest_url == "https://ingest.test/documents"
    assert not hasattr(options, "api_key")
    assert "secret" not in invocation.output


def test_api_key_on_target_origin_merges_without_losing_other_headers():
    invocation, calls = invoke_with_engine([
        "hijack", "-t", TARGET, "--api-key", "secret-key",
        "-H", "Authorization: Bearer target-secret",
        "--cleanup-url", CLEANUP, *CONTROLS,
    ])
    assert invocation.exit_code == 0
    assert calls[0][0].headers_for(TARGET) == {
        "Authorization": "Bearer target-secret", "X-Api-Key": "secret-key",
    }


def test_ingest_origin_api_key_does_not_follow_redirect_to_chat_origin():
    _, calls = invoke_with_engine([
        "poison", "-t", TARGET, "--ingest-url", "https://ingest.test/documents",
        "--api-key", "secret-key", "--cleanup-url",
        "https://ingest.test/documents/{id}", *CONTROLS,
    ])
    profile = calls[0][0]
    observed = []

    def respond(request):
        observed.append((str(request.url), request.headers.get("x-api-key")))
        if request.url.host == "ingest.test":
            return httpx.Response(302, headers={"Location": TARGET})
        return httpx.Response(200)

    with OriginBoundClient(profile, transport=httpx.MockTransport(respond)) as client:
        response = client.get("https://ingest.test/documents", follow_redirects=True)
    assert response.status_code == 200
    assert observed == [
        ("https://ingest.test/documents", "secret-key"),
        (TARGET, None),
    ]


def test_ambiguous_api_key_scope_fails_before_engine():
    invocation, calls = invoke_with_engine([
        "poison", "-t", TARGET, "--api-key", "secret-key",
        "-H", "X-Api-Key: other", "--cleanup-url", CLEANUP, *CONTROLS,
    ])
    assert invocation.exit_code == 3
    assert calls == []
    assert "secret" not in invocation.output


@pytest.mark.parametrize("ingest_url", [
    "https://name:secret@ingest.test/documents", "https://ingest.test:bogus/documents",
    "https://ingest.test/documents\x0d\x0a", "file:///tmp/documents",
])
def test_invalid_ingest_scope_fails_before_engine_and_hides_secret(ingest_url):
    invocation, calls = invoke_with_engine([
        "poison", "-t", TARGET, "--ingest-url", ingest_url,
        "--api-key", "secret-key", "--cleanup-url", CLEANUP, *CONTROLS,
    ])
    assert invocation.exit_code == 3
    assert calls == []
    assert "secret" not in invocation.output
    assert "\x0d" not in invocation.output


def test_engine_exception_is_fixed_and_secret_free():
    with patch("ragdrag.cli.run_engagement", side_effect=ValueError("secret-key")):
        invocation = CliRunner().invoke(cli, ["--quiet", "scan", "-t", TARGET,
                                               "-H", "Authorization: secret-key"])
    assert invocation.exit_code == 4
    assert "secret-key" not in invocation.output


def test_engine_preflight_configuration_error_preserves_output_and_exits_three(tmp_path):
    output = tmp_path / "run.json"
    output.write_text("original")
    with patch("ragdrag.cli.run_engagement",
               side_effect=InvalidConfiguration("Authorization: secret-key")):
        invocation = CliRunner().invoke(cli, ["--quiet", "scan", "-t", TARGET,
                                               "-o", str(output)])
    assert invocation.exit_code == 3
    assert output.read_text() == "original"
    assert "secret-key" not in invocation.output


def test_engine_warning_does_not_echo_secret_even_in_verbose_mode():
    def fake_run(profile, phases, options):
        logger = logging.getLogger("ragdrag.core.poison")
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
        try:
            logger.warning("Authorization: secret-key")
        finally:
            logger.removeHandler(handler)
        return outcome()

    with patch("ragdrag.cli.run_engagement", side_effect=fake_run):
        invocation = CliRunner().invoke(cli, ["--verbose", "--quiet", "scan", "-t", TARGET,
                                               "-H", "Authorization: secret-key"])
    assert invocation.exit_code == 0
    assert "secret-key" not in invocation.output


@pytest.mark.parametrize("code", [ExitCode.CLEAN, ExitCode.PARTIAL,
                                  ExitCode.INVALID_CONFIGURATION, ExitCode.EXECUTION_FAILURE,
                                  ExitCode.UNRESOLVED_CLEANUP])
def test_engine_exit_and_report_are_preserved(tmp_path, code):
    output = tmp_path / "run.json"
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "-o", str(output),
    ], result=outcome(code, "failed" if code == ExitCode.EXECUTION_FAILURE else "partial"))
    assert invocation.exit_code == int(code)
    assert len(calls) == 1
    assert json.loads(output.read_text())["run"]["exit_code"] == int(code)


def test_interruption_still_generates_report(tmp_path):
    output = tmp_path / "interrupted.json"
    interrupted = outcome(ExitCode.PARTIAL, "interrupted")
    with patch("ragdrag.cli.run_engagement", side_effect=EngagementInterrupted(
        interrupted.run, interrupted.evidence, interrupted.mutations,
    ), create=True):
        invocation = CliRunner().invoke(cli, ["--quiet", "scan", "-t", TARGET,
                                              "-o", str(output)])
    assert invocation.exit_code == 2
    assert json.loads(output.read_text())["run"]["status"] == "interrupted"


def test_preflight_failure_preserves_existing_report(tmp_path):
    output = tmp_path / "run.json"
    output.write_text("original")
    invocation, calls = invoke_with_engine([
        "scan", "-t", TARGET, "-p", "R4", "-o", str(output),
    ])
    assert invocation.exit_code == 3
    assert calls == []
    assert output.read_text() == "original"


@pytest.mark.parametrize("destination", ["empty", "directory", "file-parent"])
def test_invalid_output_destination_is_rejected_before_engagement(tmp_path, destination):
    original = tmp_path / "existing.json"
    original.write_text("original")
    if destination == "empty":
        output = ""
    elif destination == "directory":
        output = str(tmp_path)
    else:
        parent = tmp_path / "parent.txt"
        parent.write_text("parent")
        output = str(parent / "report.json")
    invocation, calls = invoke_with_engine(["scan", "-t", TARGET, "-o", output])
    assert invocation.exit_code == 3
    assert calls == []
    assert original.read_text() == "original"


@pytest.mark.parametrize("case", ["readonly-file", "readonly-parent", "readonly-ancestor"])
def test_deterministically_unwritable_output_rejects_before_engagement(tmp_path, case):
    original = tmp_path / "existing.json"
    original.write_text("original")
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    if case == "readonly-file":
        output = original
        output.chmod(0o444)
    elif case == "readonly-parent":
        output = readonly / "new.json"
        readonly.chmod(0o555)
    else:
        output = readonly / "missing" / "nested" / "new.json"
        readonly.chmod(0o555)
    try:
        invocation, calls = invoke_with_engine(["scan", "-t", TARGET, "-o", str(output)])
        assert invocation.exit_code == 3
        assert calls == []
        assert original.read_text() == "original"
        assert not output.exists() if output != original else output.read_text() == "original"
    finally:
        original.chmod(0o644)
        readonly.chmod(0o755)


def test_writable_existing_output_inside_nonwritable_parent_is_usable(tmp_path):
    parent = tmp_path / "readonly"
    parent.mkdir()
    output = parent / "existing.json"
    output.write_text("original")
    parent.chmod(0o555)
    try:
        invocation, calls = invoke_with_engine(["scan", "-t", TARGET, "-o", str(output)])
        assert invocation.exit_code == 0
        assert len(calls) == 1
        assert json.loads(output.read_text())["schema_version"] == "1.0"
    finally:
        parent.chmod(0o755)


def test_effective_access_check_rejects_acl_like_denial(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("original")
    actual_access = os.access
    observed = []

    def access(path, mode, *, effective_ids=False):
        observed.append((Path(path), mode, effective_ids))
        if Path(path) == output:
            return False
        return actual_access(path, mode, effective_ids=effective_ids)

    monkeypatch.setattr(os, "access", access)
    invocation, calls = invoke_with_engine(["scan", "-t", TARGET, "-o", str(output)])
    assert invocation.exit_code == 3
    assert calls == []
    assert any(path == output and effective for path, _, effective in observed)
    assert output.read_text() == "original"


def test_hostile_access_error_is_fixed_and_rejects_before_engagement(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("original")

    def access(path, mode, *, effective_ids=False):
        raise OSError("secret path from access")

    monkeypatch.setattr(os, "access", access)
    invocation, calls = invoke_with_engine(["scan", "-t", TARGET, "-o", str(output)])
    assert invocation.exit_code == 3
    assert calls == []
    assert "secret" not in invocation.output
    assert output.read_text() == "original"


def test_output_symlink_rejects_before_engagement(tmp_path):
    original = tmp_path / "original.json"
    original.write_text("original")
    alias = tmp_path / "alias.json"
    alias.symlink_to(original)
    invocation, calls = invoke_with_engine(["scan", "-t", TARGET, "-o", str(alias)])
    assert invocation.exit_code == 3
    assert calls == []
    assert original.read_text() == "original"


@pytest.mark.parametrize("code,expected", [
    (ExitCode.CLEAN, 4), (ExitCode.FINDINGS, 4), (ExitCode.PARTIAL, 4),
    (ExitCode.EXECUTION_FAILURE, 4), (ExitCode.UNRESOLVED_CLEANUP, 5),
])
def test_late_report_write_failure_retains_summary_and_exit_precedence(tmp_path, code, expected):
    output = tmp_path / "existing.json"
    output.write_text("original")
    run_outcome = outcome(code, "partial" if code in (ExitCode.PARTIAL,
                                                        ExitCode.UNRESOLVED_CLEANUP) else "completed")
    if code == ExitCode.UNRESOLVED_CLEANUP:
        run_outcome.mutations.append(MutationRecord(
            "mutation-1", "phase.r4", "target", "create", "doc-1", "DELETE",
            CleanupState.UNRESOLVED, 1,
        ))
    calls = []

    def fake_run(profile, phases, options):
        calls.append(phases)
        return run_outcome

    with patch("ragdrag.cli.run_engagement", side_effect=fake_run), patch.object(
        Path, "write_text", side_effect=OSError("disk full: secret-path"),
    ):
        invocation = CliRunner().invoke(cli, ["--quiet", "scan", "-t", TARGET,
                                               "-o", str(output)])
    assert invocation.exit_code == expected
    assert calls == [["R1", "R2", "R3"]]
    assert "Capabilities: 0" in invocation.output
    unresolved = 1 if code == ExitCode.UNRESOLVED_CLEANUP else 0
    assert f"Unresolved cleanup: {unresolved}" in invocation.output
    assert "Report output failed." in invocation.output
    assert "secret-path" not in invocation.output
    assert output.read_text() == "original"


def test_interrupted_outcome_keeps_summary_when_report_write_fails(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("original")
    interrupted = outcome(ExitCode.PARTIAL, "interrupted")
    with patch("ragdrag.cli.run_engagement", side_effect=EngagementInterrupted(
        interrupted.run, interrupted.evidence, interrupted.mutations,
    )), patch.object(Path, "write_text", side_effect=OSError("secret disk error")):
        invocation = CliRunner().invoke(cli, ["--quiet", "scan", "-t", TARGET,
                                               "-o", str(output)])
    assert invocation.exit_code == 4
    assert "Capabilities: 0" in invocation.output
    assert "secret" not in invocation.output
    assert output.read_text() == "original"


def test_report_rejects_malformed_and_control_bearing_input(tmp_path):
    source = tmp_path / "report.json"
    destination = tmp_path / "out.json"
    destination.write_text("original")
    for contents in ('{"schema_version":"1.0"}', '{"bad":"\\u001b[31m"}'):
        source.write_text(contents)
        invocation = CliRunner().invoke(cli, ["--quiet", "report", "-i", str(source),
                                              "-o", str(destination)])
        assert invocation.exit_code == 3
        assert "\x1b" not in invocation.output
        assert destination.read_text() == "original"


def test_report_validates_schema_before_formatting(tmp_path):
    report_path = tmp_path / "run.json"
    scanned, _ = invoke_with_engine(["scan", "-t", TARGET, "-o", str(report_path)])
    assert scanned.exit_code == 0
    shown = CliRunner().invoke(cli, ["--quiet", "report", "-i", str(report_path)])
    assert shown.exit_code == 0
    assert json.loads(shown.output)["schema_version"] == "1.0"


def test_report_rejects_control_characters_in_otherwise_valid_schema(tmp_path):
    source = tmp_path / "run.json"
    scanned, _ = invoke_with_engine(["scan", "-t", TARGET, "-o", str(source)])
    assert scanned.exit_code == 0
    data = json.loads(source.read_text())
    data["tool"]["version"] = "\x1b[31m"
    source.write_text(json.dumps(data))
    shown = CliRunner().invoke(cli, ["--quiet", "report", "-i", str(source)])
    assert shown.exit_code == 3
    assert "\x1b" not in shown.output


def test_report_rejects_duplicate_keys_in_otherwise_valid_schema(tmp_path):
    source = tmp_path / "run.json"
    scanned, _ = invoke_with_engine(["scan", "-t", TARGET, "-o", str(source)])
    assert scanned.exit_code == 0
    contents = source.read_text().replace(
        '"schema_version": "1.0"',
        '"schema_version": "0.1", "schema_version": "1.0"', 1,
    )
    source.write_text(contents)
    shown = CliRunner().invoke(cli, ["--quiet", "report", "-i", str(source)])
    assert shown.exit_code == 3


def test_listen_help_keeps_established_options():
    invocation = CliRunner().invoke(cli, ["--quiet", "listen", "--help"])
    assert invocation.exit_code == 0
    for option in ("--port", "--host", "--output", "--tls"):
        assert option in invocation.output


@pytest.mark.parametrize("host", [
    "0.0.0.0", "192.0.2.1", "example.test", "localhost.example.test",
    "::", "::ffff:127.0.0.1", "[::1]", "127.0.0.1.evil.test",
])
def test_listen_nonloopback_or_ambiguous_bind_requires_ack(host):
    with patch("ragdrag.core.listener.start_listener") as start:
        invocation = CliRunner().invoke(cli, ["--quiet", "listen", "--host", host])
    assert invocation.exit_code == 3
    assert "--allow-public" in invocation.output
    start.assert_not_called()


def test_listen_defaults_to_loopback_and_forwards_limits_raw_and_tls():
    with patch("ragdrag.core.listener.start_listener") as start:
        invocation = CliRunner().invoke(cli, [
            "--quiet", "listen", "--port", "8444", "--output", "capture.jsonl",
            "--tls", "--max-body-bytes", "42", "--max-concurrency", "3", "--store-raw",
        ])
    assert invocation.exit_code == 0
    start.assert_called_once_with(
        host="127.0.0.1", port=8444, output="capture.jsonl", tls=True,
        max_body_bytes=42, max_concurrency=3, store_raw=True, allow_public=False,
    )


@pytest.mark.parametrize("option,value", [
    ("--max-body-bytes", "0"), ("--max-body-bytes", "1048577"),
    ("--max-body-bytes", "1.5"), ("--max-concurrency", "0"),
    ("--max-concurrency", "65"), ("--max-concurrency", "1.5"),
])
def test_listen_limits_reject_invalid_and_out_of_range_values(option, value):
    with patch("ragdrag.core.listener.start_listener") as start:
        invocation = CliRunner().invoke(cli, ["--quiet", "listen", option, value])
    assert invocation.exit_code == 3
    start.assert_not_called()


def test_listen_public_ack_is_forwarded():
    with patch("ragdrag.core.listener.start_listener") as start:
        invocation = CliRunner().invoke(cli, [
            "--quiet", "listen", "--host", "0.0.0.0", "--allow-public",
        ])
    assert invocation.exit_code == 0
    assert start.call_args.kwargs["allow_public"] is True
    assert start.call_args.kwargs["host"] == "0.0.0.0"


class TestValidateUrl:
    def test_accepts_http(self):
        assert _validate_url("http://example.com") == "http://example.com"

    def test_accepts_https(self):
        assert _validate_url("https://example.com") == "https://example.com"

    def test_rejects_missing_scheme(self):
        with pytest.raises(BadParameter, match="must start with"):
            _validate_url("example.com")

    def test_rejects_ftp(self):
        with pytest.raises(BadParameter, match="must start with"):
            _validate_url("ftp://example.com")

    def test_rejects_empty_host(self):
        """'http://' alone used to slip through prefix-only validation."""
        with pytest.raises(BadParameter, match="missing host"):
            _validate_url("http://")

    def test_strips_trailing_slash(self):
        """Canonicalization: trailing slash on non-root paths is stripped."""
        assert _validate_url("https://example.com/api/") == "https://example.com/api"

    def test_preserves_root_slash(self):
        """Root path '/' should not be stripped to empty."""
        assert _validate_url("https://example.com/") == "https://example.com/"

    def test_drops_fragment(self):
        """Fragments don't travel over the wire; drop them during canonicalization."""
        assert _validate_url("https://example.com/api#anchor") == "https://example.com/api"

    def test_preserves_query_string(self):
        assert _validate_url("https://example.com/api?key=value") == "https://example.com/api?key=value"

    def test_preserves_port(self):
        assert _validate_url("http://example.com:8080/api") == "http://example.com:8080/api"

    def test_preserves_path(self):
        assert _validate_url("https://example.com/v1/query") == "https://example.com/v1/query"
