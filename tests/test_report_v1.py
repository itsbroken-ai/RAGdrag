"""Report v1 contracts at the untrusted serialization boundary."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from ragdrag.engine.models import (
    CapabilityResult, CapabilityStatus, CleanupState, EvidenceItem, EvidenceState,
    ExitCode, Finding, ImpactLevel, MutationRecord, OutcomeCode, RunResult,
)
from ragdrag.engine.phases import EngagementOutcome, PHASE_METADATA
from ragdrag.engine.profile import TargetProfile
from ragdrag.reporters import json_report
from ragdrag import __version__


SCHEMA = Path(__file__).parents[1] / "ragdrag/reporters/schemas/report-v1.schema.json"


def profile():
    return TargetProfile.from_cli(
        "https://target.test/chat", headers={"Authorization": "Bearer profile-secret"},
        cookie="sid=cookie-secret",
    )


def outcome(*, status="completed", exit_code=ExitCode.CLEAN, capabilities=None):
    return EngagementOutcome(
        run=RunResult(
            run_id="run-1", target="https://target.test/chat",
            started_at="2026-09-29T12:00:00+00:00",
            ended_at="2026-09-29T12:00:01+00:00",
            capabilities=[] if capabilities is None else capabilities,
            exit_code=exit_code, status=status,
        ), evidence=[], mutations=[],
    )


def capability(*, status=CapabilityStatus.PARTIAL, findings=None):
    return CapabilityResult(
        capability_id="phase.r3", technique_ids=("RD-0301",), status=status,
        impact=ImpactLevel.ACTIVE, started_at="start", ended_at="end", trials=2,
        controls=["negative-control"], findings=[] if findings is None else findings,
        evidence_ids=["ev-1"], outcomes=[OutcomeCode.AUTHENTICATION_REQUIRED],
        errors=["fixed failure"], mutation_ids=["mu-1"],
        cleanup_state=CleanupState.UNRESOLVED, summary="phase partial",
        payload={"tested": 2},
    )


def validate_report(report):
    schema = json.loads(SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)
    return schema


@pytest.mark.parametrize("status,exit_code", [
    ("completed", ExitCode.CLEAN), ("partial", ExitCode.PARTIAL),
    ("interrupted", ExitCode.PARTIAL), ("failed", ExitCode.EXECUTION_FAILURE),
])
def test_empty_and_interrupted_reports_have_exact_schema(status, exit_code):
    report = json_report.generate_run_report(outcome(status=status, exit_code=exit_code), profile())
    schema = validate_report(report)
    assert set(report) == set(schema["required"])
    assert report["run"]["status"] == status
    assert report["run"]["exit_code"] == int(exit_code)
    assert report["summary"]["total_findings"] == 0
    assert json.loads(json.dumps(report)) == report


def test_partial_report_preserves_typed_results_and_flattened_findings():
    finding = Finding("RD-0301", "Direct Knowledge Extraction", "high", "safe detail",
                      evidence_state=EvidenceState.VALIDATED, finding_id="fi-1")
    item = capability(findings=[finding])
    run = outcome(status="partial", exit_code=ExitCode.UNRESOLVED_CLEANUP, capabilities=[item])
    run.evidence.append(EvidenceItem("ev-1", "phase.r3", "trial-1", "time",
                                     EvidenceState.OBSERVED, "response", {"method": "POST"},
                                     {"status_code": 403}))
    run.mutations.append(MutationRecord("mu-1", "phase.r3", "target", "create", "obj",
                                        "delete", CleanupState.UNRESOLVED, 1, ["ev-1"]))
    report = json_report.generate_run_report(run, profile())
    validate_report(report)
    assert report["capabilities"][0]["finding_ids"] == ["fi-1"]
    assert report["capabilities"][0]["controls"] == ["negative-control"]
    assert report["capabilities"][0]["outcomes"] == ["authentication-required"]
    assert report["capabilities"][0]["errors"] == ["<redacted>"]
    assert report["findings"][0]["finding_id"] == "fi-1"
    assert report["summary"]["validated"] == 1
    assert report["summary"]["total_findings"] == 1
    assert report["summary"]["unresolved_mutations"] == 1
    assert report["mutations"][0]["state"] == "unresolved"
    assert report["evidence"][0]["state"] == "observed"


def test_implementation_rows_follow_live_phase_metadata(monkeypatch):
    report = json_report.generate_run_report(outcome(), profile())
    rows = {row["phase"]: row for row in report["implementation_status"]}
    assert set(rows) == set(PHASE_METADATA)
    assert rows["R3"]["technique_ids"] == list(PHASE_METADATA["R3"].technique_ids)
    assert rows["R4"]["impact"] == "mutating"


def test_schema_rejects_unknown_nested_fields():
    report = json_report.generate_run_report(outcome(capabilities=[capability()]), profile())
    schema = validate_report(report)
    report["capabilities"][0]["surprise"] = True
    assert list(Draft202012Validator(schema).iter_errors(report))


def test_exception_message_and_unknown_free_text_are_withheld():
    run = outcome(capabilities=[capability()])
    item = run.run.capabilities[0]
    item.errors = ["RuntimeError: violetmoon", "violetmoon"]
    item.summary = "violetmoon"
    item.payload = {"note": "violetmoon"}
    run.evidence.append(EvidenceItem("ev-1", "phase.r3", "trial-1", "time",
                                     EvidenceState.OBSERVED, "violetmoon",
                                     {"note": "violetmoon"}, {}))
    report = json_report.generate_run_report(run, profile())
    validate_report(report)
    assert "violetmoon" not in json.dumps(report)
    assert report["capabilities"][0]["errors"]


def test_target_userinfo_query_and_opaque_path_segments_are_withheld():
    target = TargetProfile.from_cli(
        "https://alice:violetmoon@target.test/violetmoon/chat?api_key=violetmoon"
    )
    report = json_report.generate_run_report(outcome(), target)
    validate_report(report)
    assert "violetmoon" not in json.dumps(report)
    assert "alice" not in json.dumps(report)
    assert report["target"]["url"].endswith("/redacted/chat")


def test_opaque_secret_identifiers_and_config_names_keep_references_without_leaking():
    finding = Finding("RD-0301", "Direct Knowledge Extraction", "high", "safe",
                      finding_id="violetmoon")
    item = capability(findings=[finding])
    item.controls = ["violetmoon"]
    item.evidence_ids = ["violetmoon"]
    item.mutation_ids = ["violetmoon"]
    run = outcome(capabilities=[item])
    run.run.run_id = "violetmoon"
    run.evidence.append(EvidenceItem("violetmoon", "phase.r3", "violetmoon", "violetmoon",
                                     EvidenceState.OBSERVED, "response", {}, {}))
    run.mutations.append(MutationRecord("violetmoon", "phase.r3", "scope", "create",
                                        "object", "delete", CleanupState.UNRESOLVED,
                                        evidence_ids=["violetmoon"]))
    custom_profile = TargetProfile.from_cli("https://target.test/chat", query_field="violetmoon",
                                             response_field="violetmoon")
    report = json_report.generate_run_report(run, custom_profile)
    validate_report(report)
    assert "violetmoon" not in json.dumps(report)
    assert report["capabilities"][0]["finding_ids"] == [report["findings"][0]["finding_id"]]
    assert report["capabilities"][0]["evidence_ids"] == [report["evidence"][0]["evidence_id"]]
    assert report["capabilities"][0]["mutation_ids"] == [report["mutations"][0]["mutation_id"]]


def test_untrusted_error_class_and_artifact_digest_cannot_escape():
    run = outcome(capabilities=[capability()])
    run.run.capabilities[0].errors = ["VioletmoonError: private message"]
    run.evidence.append(EvidenceItem("ev-1", "phase.r3", "trial-1", "time",
                                     EvidenceState.OBSERVED, "response", {}, {},
                                     artifact_digest="violetmoon"))
    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_run_report(run, profile())
    run.evidence[0].artifact_digest = None
    report = json_report.generate_run_report(run, profile())
    assert "Violetmoon" not in json.dumps(report)


def test_entire_report_redacts_keys_values_free_text_and_profile_secrets(tmp_path):
    run = outcome(capabilities=[capability()])
    run.run.capabilities[0].summary = "Bearer profile-secret"
    run.run.capabilities[0].errors = ["Cookie: sid=cookie-secret"]
    run.run.capabilities[0].payload = {"api_key=payload-secret": "Bearer payload-secret"}
    run.evidence.append(EvidenceItem("ev-1", "phase.r3", "trial-1", "time",
                                     EvidenceState.OBSERVED, "response", {
                                         "Authorization": "Bearer header-secret",
                                         "Cookie": "sid=cookie-secret",
                                     }, {"raw_response": "raw-secret-document"}))
    path = tmp_path / "report.json"
    report = json_report.generate_run_report(run, profile(), path)
    validate_report(report)
    dumped = json.dumps(report)
    for secret in ("profile-secret", "cookie-secret", "payload-secret", "header-secret", "raw-secret-document"):
        assert secret not in dumped
        assert secret not in path.read_text()
    assert json.loads(path.read_text()) == report


class Hostile:
    def __str__(self):
        raise AssertionError("str hook called")

    def __repr__(self):
        raise AssertionError("repr hook called")


@pytest.mark.parametrize("location", ["payload_value", "payload_key", "summary", "outcome", "evidence"])
def test_malformed_objects_fail_with_fixed_non_reflective_error(location):
    run = outcome(capabilities=[capability()])
    item = run.run.capabilities[0]
    if location == "payload_value":
        item.payload = {"bad": Hostile()}
    elif location == "payload_key":
        item.payload = {Hostile(): "bad"}
    elif location == "summary":
        item.summary = Hostile()
    elif location == "outcome":
        item.outcomes = [Hostile()]
    else:
        run.evidence = [Hostile()]
    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_run_report(run, profile())


def test_legacy_adapter_emits_schema_without_invoking_phase_to_dict():
    class Legacy:
        target = "https://target.test/chat"
        findings = [Finding("RD-0101", "RAG Presence Detection", "high", "safe")]

        def to_dict(self):
            raise AssertionError("to_dict must not run")

    report = json_report.generate_report(Legacy())
    validate_report(report)
    assert len(report["capabilities"]) == 1
    assert report["summary"]["total_findings"] == 1


def test_empty_legacy_phase_keeps_its_capability_identity(tmp_path):
    from ragdrag.core.probe import ProbeResult

    path = tmp_path / "legacy.json"
    report = json_report.generate_report(ProbeResult("https://target.test/chat"), path)
    validate_report(report)
    assert report["capabilities"][0]["capability_id"] == "phase.r2"
    assert json.loads(path.read_text()) == report


def test_summary_uses_canonical_counts_and_hides_redacted_fields():
    report = json_report.generate_run_report(outcome(capabilities=[capability()]), profile())
    rendered = json_report.format_summary(report, color=False)
    assert "partial" in rendered.lower()
    assert "authentication-required" in rendered
    assert "unresolved" in rendered.lower()
    assert "<redacted>" not in rendered


def test_sensitive_artifact_is_complete_owner_only_and_digest_only(tmp_path):
    path = tmp_path / "raw.bin"
    metadata = json_report.write_sensitive_artifact(path, b"secret")
    assert path.read_bytes() == b"secret"
    assert path.stat().st_mode & 0o777 == 0o600
    assert metadata == {"path": str(path), "size_bytes": 6,
                        "sha256": hashlib.sha256(b"secret").hexdigest()}
    assert b"secret" not in json.dumps(metadata).encode()


def test_sensitive_artifact_overwrites_regular_file_and_restricts_mode(tmp_path):
    path = tmp_path / "raw.bin"
    path.write_bytes(b"longer previous material")
    path.chmod(0o644)
    json_report.write_sensitive_artifact(path, b"new")
    assert path.read_bytes() == b"new"
    assert path.stat().st_mode & 0o777 == 0o600


def test_sensitive_artifact_rejects_symlink_and_nonregular_target(tmp_path):
    real = tmp_path / "real"
    real.write_bytes(b"unchanged")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(OSError):
        json_report.write_sensitive_artifact(link, b"secret")
    assert real.read_bytes() == b"unchanged"
    with pytest.raises(OSError):
        json_report.write_sensitive_artifact(tmp_path, b"secret")


def test_package_data_declares_report_schema():
    config = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert '"reporters/schemas/*.json"' in config
    assert '"jsonschema>=4.23"' in config


@pytest.mark.parametrize("key,value", [
    ("vector", [0.125, 0.25, 0.5]),
    ("token", 123456789),
    ("raw_response", {"status_code": 987654321}),
    ("Authorization", [123456789]),
    ("Cookie", {"matched": 123456789}),
    ("body", [{"tested": 123456789}]),
])
@pytest.mark.parametrize("sink", ["payload", "finding", "request", "response"])
def test_sensitive_dynamic_keys_withhold_whole_subtrees(sink, key, value, tmp_path):
    finding = Finding("RD-0301", "Direct Knowledge Extraction", "high", "safe",
                      evidence={"outer": {key: value}}, finding_id="fi-1")
    item = capability(findings=[finding])
    run = outcome(capabilities=[item])
    evidence = EvidenceItem("ev-1", "phase.r3", "trial-1", "2026-09-29T12:00:00+00:00",
                            EvidenceState.OBSERVED, "response", {}, {})
    if sink == "payload":
        item.payload = {"outer": {key: value}}
    elif sink == "request":
        evidence.request_summary = {"outer": {key: value}}
        run.evidence.append(evidence)
    elif sink == "response":
        evidence.response_summary = {"outer": {key: value}}
        run.evidence.append(evidence)
    path = tmp_path / "report.json"
    report = json_report.generate_run_report(run, profile(), path)
    validate_report(report)
    nested = {
        "payload": report["capabilities"][0]["payload"],
        "finding": report["findings"][0]["evidence"],
        "request": report["evidence"][0]["request_summary"] if run.evidence else {},
        "response": report["evidence"][0]["response_summary"] if run.evidence else {},
    }[sink]
    assert nested["<redacted-key>"]["<redacted-key>"] == "<redacted>"
    assert json.loads(path.read_text()) == report


@pytest.mark.parametrize("token", [
    "completed", "capabilities", "schema_version", "summary", "tool", "ragdrag",
    "validated", "active-non-mutating", "R1", "phase.r1", "RD-0101", __version__,
])
def test_credentials_colliding_with_schema_or_public_metadata_leave_canonical_report(token, tmp_path):
    collision = TargetProfile.from_cli("https://target.test/chat",
                                       headers={"Authorization": f"Bearer {token}"})
    run = outcome(capabilities=[capability(status=CapabilityStatus.COMPLETED)])
    path = tmp_path / "report.json"
    report = json_report.generate_run_report(run, collision, path)
    validate_report(report)
    assert report["schema_version"] == "1.0"
    assert report["tool"] == {"name": "ragdrag", "version": __version__}
    assert report["run"]["status"] == "completed"
    assert report["capabilities"][0]["status"] == "completed"
    assert report["summary"]["completed"] == 1
    assert report["implementation_status"][0]["phase"] == "R1"
    assert json.loads(path.read_text()) == report


def test_hostile_legacy_metaclass_dispatch_has_fixed_error_without_hooks():
    called = []

    class HostileMeta(type):
        def __hash__(cls):
            called.append("hash")
            raise RuntimeError("synthetic-private-marker")

        def __eq__(cls, other):
            called.append("eq")
            raise RuntimeError("synthetic-private-marker")

        def __str__(cls):
            called.append("str")
            raise RuntimeError("synthetic-private-marker")

        def __repr__(cls):
            called.append("repr")
            raise RuntimeError("synthetic-private-marker")

    class Legacy(metaclass=HostileMeta):
        target = "https://target.test/chat"
        findings = []

    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_report(Legacy())
    assert called == []


def test_exfiltrate_adapter_collects_all_documented_findings_once():
    from ragdrag.core.exfiltrate import ExfilFinding, ExfiltrateResult

    ordinary = ExfilFinding("RD-0301", "Direct", "high", "general", "safe", "safe")
    bypass = ExfilFinding("RD-0302", "Guardrail", "high", "general", "safe", "safe")
    result = ExfiltrateResult("https://target.test/chat", total_queries=3,
                             findings=[ordinary], guardrail_detected=True,
                             guardrail_bypass_findings=[bypass, ordinary])
    report = json_report.generate_report(result)
    validate_report(report)
    assert report["capabilities"][0]["capability_id"] == "phase.r3"
    assert report["summary"]["total_findings"] == 2
    assert report["capabilities"][0]["finding_ids"] == [f["finding_id"] for f in report["findings"]]
    assert {f["technique_id"] for f in report["findings"]} == {"RD-0301", "RD-0302"}
    assert report["capabilities"][0]["payload"]["total_queries"] == 3
    assert report["capabilities"][0]["payload"]["guardrail_detected"] is True
    assert report["run"]["exit_code"] == 1


def test_sensitive_artifact_rejects_symlinked_ancestor_without_touching_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    victim = real / "existing"
    victim.write_bytes(b"original")
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError, match="^sensitive artifact write failed$"):
        json_report.write_sensitive_artifact(linked / "existing", b"synthetic-sensitive")
    assert victim.read_bytes() == b"original"
    assert sorted(p.name for p in real.iterdir()) == ["existing"]


def test_sensitive_artifact_replaces_hardlink_without_mutating_sibling(tmp_path):
    sibling = tmp_path / "sibling"
    sibling.write_bytes(b"original")
    alias = tmp_path / "alias"
    os.link(sibling, alias)
    metadata = json_report.write_sensitive_artifact(alias, b"new material")
    assert sibling.read_bytes() == b"original"
    assert alias.read_bytes() == b"new material"
    assert alias.stat().st_mode & 0o777 == 0o600
    assert metadata["size_bytes"] == len(b"new material")


def test_sensitive_artifact_failed_write_preserves_existing_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "artifact"
    path.write_bytes(b"original")
    real_write = os.write
    calls = 0

    def partial_then_fail(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, data[:2])
        raise OSError("private storage failure")

    monkeypatch.setattr(os, "write", partial_then_fail)
    with pytest.raises(OSError, match="^sensitive artifact write failed$"):
        json_report.write_sensitive_artifact(path, b"synthetic-sensitive")
    assert path.read_bytes() == b"original"
    assert [p.name for p in tmp_path.iterdir()] == ["artifact"]


@pytest.mark.parametrize("url,expected", [
    ("https://target.test", "https://target.test"),
    ("https://target.test/", "https://target.test/"),
    ("https://target.test/chat", "https://target.test/chat"),
    ("https://target.test/api/v1/chat", "https://target.test/api/v1/chat"),
    ("https://target.test/api//chat", "https://target.test/api//chat"),
    ("https://target.test/private/chat", "https://target.test/redacted/chat"),
])
def test_target_sanitization_preserves_real_delimiters(url, expected):
    report = json_report.generate_run_report(outcome(), TargetProfile.from_cli(url))
    assert report["target"]["url"] == expected


def test_target_sanitization_preserves_existing_trailing_delimiter():
    target = TargetProfile.from_cli("https://target.test/chat")
    object.__setattr__(target, "target_url", "https://target.test/chat/")
    report = json_report.generate_run_report(outcome(), target)
    assert report["target"]["url"] == "https://target.test/chat/"


@pytest.mark.parametrize("place", ["run_id", "payload_key", "payload_value", "huge_int", "huge_text"])
def test_malformed_utf8_or_unbounded_json_fails_before_output(place, tmp_path):
    run = outcome(capabilities=[capability()])
    path = tmp_path / "report.json"
    if place == "run_id":
        run.run.run_id = "private\ud800"
    elif place == "payload_key":
        run.run.capabilities[0].payload = {"private\ud800": 1}
    elif place == "payload_value":
        run.run.capabilities[0].payload = {"tested": "private\ud800"}
    elif place == "huge_int":
        run.run.capabilities[0].payload = {"tested": 10**5000}
    else:
        run.run.capabilities[0].payload = {"tested": "x" * 1_000_001}
    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_run_report(run, profile(), path)
    assert not path.exists()


def test_final_schema_gate_rejects_corrupted_redaction_result_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(json_report, "redact", lambda report: {"schema_version": "1.0"})
    path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_run_report(outcome(), profile(), path)
    assert not path.exists()


def test_report_output_path_rejects_custom_hooks_and_surrogates_before_write(tmp_path):
    called = []

    class HostilePath:
        def __fspath__(self):
            called.append("fspath")
            raise RuntimeError("private path marker")

        def __str__(self):
            called.append("str")
            raise RuntimeError("private path marker")

    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_run_report(outcome(), profile(), HostilePath())
    assert called == []
    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_run_report(outcome(), profile(), str(tmp_path / "private\ud800.json"))


def test_configured_uuid_secret_is_removed_from_all_joined_references():
    token = "7caf785b-987d-44f0-b1b5-36bc1753ec22"
    secret_profile = TargetProfile.from_cli("https://target.test/chat",
                                            headers={"Authorization": "Bearer " + token})
    finding = Finding("RD-0301", "Direct", "high", "safe", finding_id=token)
    cap = capability(findings=[finding])
    cap.capability_id = token
    cap.evidence_ids = [token]
    cap.mutation_ids = [token]
    run = outcome(capabilities=[cap])
    run.run.run_id = token
    run.evidence.append(EvidenceItem(token, token, token, "2026-09-29T12:00:00+00:00",
                                     EvidenceState.OBSERVED, "response", {}, {}))
    run.mutations.append(MutationRecord(token, token, "scope", "create", "obj", "delete",
                                        CleanupState.UNRESOLVED, evidence_ids=[token]))
    report = json_report.generate_run_report(run, secret_profile)
    validate_report(report)
    serialized = json.dumps(report, allow_nan=False, ensure_ascii=False)
    serialized.encode("utf-8")
    assert token not in serialized
    assert token not in json_report.format_summary(report, color=False)
    assert report["capabilities"][0]["capability_id"] == report["evidence"][0]["capability_id"]
    assert report["capabilities"][0]["capability_id"] == report["mutations"][0]["capability_id"]
    assert report["capabilities"][0]["finding_ids"] == [report["findings"][0]["finding_id"]]
    assert report["capabilities"][0]["evidence_ids"] == [report["evidence"][0]["evidence_id"]]
    assert report["mutations"][0]["evidence_ids"] == [report["evidence"][0]["evidence_id"]]
    assert report["capabilities"][0]["mutation_ids"] == [report["mutations"][0]["mutation_id"]]


def test_configured_hex_digest_secret_becomes_valid_safe_digest():
    token = "73d14cbb36d073e19f3a13e61973361ecf91a2108b7a31fd7ef40b9d391fc401"
    secret_profile = TargetProfile.from_cli("https://target.test/chat",
                                            headers={"X-Api-Key": token})
    run = outcome()
    run.evidence.append(EvidenceItem("ev-1", "phase.r3", "trial-1",
                                     "2026-09-29T12:00:00+00:00", EvidenceState.OBSERVED,
                                     "response", {}, {}, artifact_digest=token))
    report = json_report.generate_run_report(run, secret_profile)
    validate_report(report)
    digest = report["evidence"][0]["artifact_digest"]
    assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
    assert token not in json.dumps(report, allow_nan=False)


def test_configured_host_secret_maps_target_and_origins_consistently():
    from ragdrag.engine.profile import canonical_origin

    token = "violetmoon"
    secret_profile = TargetProfile.from_cli(
        "https://violetmoon.target.test:8443/chat",
        headers={"Authorization": "Bearer " + token},
        additional_origin_headers={"https://violetmoon.other.test:9443": {}},
    )
    report = json_report.generate_run_report(outcome(), secret_profile)
    validate_report(report)
    serialized = json.dumps(report, allow_nan=False)
    assert token not in serialized
    assert token not in json_report.format_summary(report, color=False)
    assert canonical_origin(report["target"]["url"]) in report["target"]["approved_origins"]
    assert len(report["target"]["approved_origins"]) == 2


@pytest.mark.parametrize("token,location", [
    ("POST", "method"),
    ("negative-control", "control"),
    ("123456789", "number"),
    ("legacy phase request failed", "error"),
    ("chat", "path"),
])
def test_configured_secret_is_removed_from_other_untrusted_value_sinks(token, location):
    secret_profile = TargetProfile.from_cli("https://target.test/chat",
                                            headers={"Authorization": "Bearer " + token})
    cap = capability()
    run = outcome(capabilities=[cap])
    if location == "method":
        cap.payload = {"method": token}
    elif location == "control":
        cap.controls = [token]
    elif location == "number":
        cap.payload = {"tested": int(token)}
    elif location == "error":
        cap.errors = [token]
    report = json_report.generate_run_report(run, secret_profile)
    validate_report(report)
    assert token not in json.dumps(report, allow_nan=False)
    assert token not in json_report.format_summary(report, color=False)


def test_temp_name_collision_does_not_delete_preexisting_file(tmp_path, monkeypatch):
    target = tmp_path / "artifact"
    target.write_bytes(b"original")
    collision = tmp_path / ".ragdrag-fixed.tmp"
    collision.write_bytes(b"unrelated-existing")
    monkeypatch.setattr(json_report, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    with pytest.raises(OSError, match="^sensitive artifact write failed$"):
        json_report.write_sensitive_artifact(target, b"new bytes")
    assert target.read_bytes() == b"original"
    assert collision.read_bytes() == b"unrelated-existing"


def test_temp_cleanup_does_not_delete_swapped_unrelated_entry(tmp_path, monkeypatch):
    target = tmp_path / "artifact"
    target.write_bytes(b"original")
    temp = tmp_path / ".ragdrag-fixed.tmp"
    moved = tmp_path / "created-moved"
    monkeypatch.setattr(json_report, "uuid4", lambda: SimpleNamespace(hex="fixed"))

    def swap_then_fail(fd, mode):
        temp.rename(moved)
        temp.write_bytes(b"unrelated-existing")
        raise OSError("private storage failure")

    monkeypatch.setattr(os, "fchmod", swap_then_fail)
    with pytest.raises(OSError, match="^sensitive artifact write failed$"):
        json_report.write_sensitive_artifact(target, b"new bytes")
    assert target.read_bytes() == b"original"
    assert temp.read_bytes() == b"unrelated-existing"
    assert moved.exists()


def test_hostile_metaclass_scalar_cannot_escape_construction_final_gate_or_paths(monkeypatch):
    called = []

    class Meta(type):
        def __eq__(cls, other):
            called.append("eq")
            raise RuntimeError("synthetic-private-marker")

        def __hash__(cls):
            called.append("hash")
            raise RuntimeError("synthetic-private-marker")

        def __str__(cls):
            called.append("str")
            raise RuntimeError("synthetic-private-marker")

        def __repr__(cls):
            called.append("repr")
            raise RuntimeError("synthetic-private-marker")

    class Scalar(metaclass=Meta):
        pass

    cases = []
    run = outcome(capabilities=[capability()])
    run.run.capabilities[0].payload = {"tested": Scalar()}
    cases.append(lambda: json_report.generate_run_report(run, profile()))
    run2 = outcome(capabilities=[capability()])
    run2.run.capabilities[0].controls = Scalar()
    cases.append(lambda: json_report.generate_run_report(run2, profile()))
    run3 = outcome(capabilities=[capability()])
    run3.run.capabilities[0].technique_ids = Scalar()
    cases.append(lambda: json_report.generate_run_report(run3, profile()))
    cases.append(lambda: json_report.write_sensitive_artifact(Scalar(), b"fixture"))
    original_redact = json_report.redact

    def inject_after_redaction(report):
        safe = original_redact(report)
        safe["capabilities"][0]["payload"]["tested"] = Scalar()
        return safe

    for call in cases:
        with pytest.raises(ValueError, match="^invalid report data$"):
            call()
    monkeypatch.setattr(json_report, "redact", inject_after_redaction)
    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_run_report(outcome(capabilities=[capability()]), profile())
    assert called == []


@pytest.mark.parametrize("source,domain", [
    ("run-123", "run_id"), ("fi-123", "finding_id"),
    ("ev-123", "evidence_id"), ("mu-123", "mutation_id"),
    ("trial-123", "trial_id"), ("phase.r3", "capability_id"),
    ("RD-0301", "technique_id"),
])
def test_configured_reference_secret_keeps_domain_joins_and_public_metadata(source, domain):
    secret_profile = TargetProfile.from_cli("https://target.test/chat",
                                            headers={"X-Api-Key": source})
    finding = Finding("RD-0301", "Direct", "high", "safe", finding_id="fi-123")
    cap = capability(findings=[finding])
    cap.capability_id = "phase.r3"
    cap.evidence_ids = ["ev-123"]
    cap.mutation_ids = ["mu-123"]
    run = outcome(capabilities=[cap])
    run.run.run_id = "run-123"
    run.evidence.append(EvidenceItem("ev-123", "phase.r3", "trial-123", "2026-09-29T12:00:00+00:00",
                                     EvidenceState.OBSERVED, "response", {}, {}))
    run.mutations.append(MutationRecord("mu-123", "phase.r3", "scope", "create", "obj", "delete",
                                        CleanupState.UNRESOLVED, evidence_ids=["ev-123"]))
    report = json_report.generate_run_report(run, secret_profile)
    validate_report(report)
    assert report["capabilities"][0]["finding_ids"] == [report["findings"][0]["finding_id"]]
    assert report["capabilities"][0]["evidence_ids"] == [report["evidence"][0]["evidence_id"]]
    assert report["capabilities"][0]["mutation_ids"] == [report["mutations"][0]["mutation_id"]]
    assert report["capabilities"][0]["capability_id"] == report["evidence"][0]["capability_id"]
    assert report["capabilities"][0]["capability_id"] == report["mutations"][0]["capability_id"]
    assert report["capabilities"][0]["technique_ids"] == [report["findings"][0]["technique_id"]]
    assert report["implementation_status"][2]["technique_ids"] == list(PHASE_METADATA["R3"].technique_ids)
    assert report["implementation_status"][2]["capability"] == PHASE_METADATA["R3"].capability_id
    assert source not in json.dumps({"run": report["run"], "capabilities": report["capabilities"],
                                    "findings": report["findings"], "evidence": report["evidence"],
                                    "mutations": report["mutations"]}, allow_nan=False)


def test_configured_secret_in_cookie_and_nested_allowed_strings_is_removed():
    token = "session-cookie-123"
    secret_profile = TargetProfile.from_cli("https://target.test/chat", cookie="sid=" + token)
    cap = capability()
    cap.payload = {"method": "POST", "nested": {"method": token, token: [token]}}
    run = outcome(capabilities=[cap])
    run.evidence.append(EvidenceItem("ev-1", "phase.r3", "trial-1", "2026-09-29T12:00:00+00:00",
                                     EvidenceState.OBSERVED, "response", {"method": token},
                                     {"sensitivity": token}))
    report = json_report.generate_run_report(run, secret_profile)
    validate_report(report)
    assert token not in json.dumps(report, allow_nan=False)


def test_hostile_metaclass_rejected_across_sequence_and_legacy_allowlists():
    from ragdrag.core.probe import ProbeResult

    called = []

    class Meta(type):
        def __eq__(cls, other):
            called.append("eq")
            raise RuntimeError("synthetic-private-marker")

        def __hash__(cls):
            called.append("hash")
            raise RuntimeError("synthetic-private-marker")

    class Scalar(metaclass=Meta):
        pass

    for field in ("technique_ids", "controls", "evidence_ids", "errors", "mutation_ids"):
        run = outcome(capabilities=[capability()])
        setattr(run.run.capabilities[0], field, Scalar())
        with pytest.raises(ValueError, match="^invalid report data$"):
            json_report.generate_run_report(run, profile())
    legacy = ProbeResult("https://target.test/chat")
    legacy.retrieval_count = Scalar()
    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_report(legacy)
    assert called == []


@pytest.mark.parametrize("token,value", [
    ("0", 0), ("0.0", 0.0), ("true", True), ("false", False),
    ("null", None), ("<redacted>", 123), ("redacted", 123),
])
@pytest.mark.parametrize("sink", ["payload", "finding", "request", "response"])
def test_final_dynamic_sinks_withhold_scalar_spelling_and_redaction_marker(
    token, value, sink, tmp_path,
):
    secret_profile = TargetProfile.from_cli("https://target.test/chat", headers={"X-Api-Key": token})
    dynamic = {"documents_injected" if token in {"<redacted>", "redacted"} else "tested": value}
    finding = Finding("RD-0301", "Direct", "high", "safe", evidence=dynamic if sink == "finding" else {})
    cap = capability(findings=[finding])
    if sink == "payload":
        cap.payload = dynamic
    run = outcome(capabilities=[cap])
    if sink in {"request", "response"}:
        run.evidence.append(EvidenceItem("ev-1", "phase.r3", "trial-1", "2026-09-29T12:00:00+00:00",
                                         EvidenceState.OBSERVED, "response",
                                         dynamic if sink == "request" else {},
                                         dynamic if sink == "response" else {}))
    output = tmp_path / "report.json"
    report = json_report.generate_run_report(run, secret_profile, output)
    validate_report(report)
    assert json.loads(output.read_text()) == report
    exposed = {
        "payload": report["capabilities"][0]["payload"],
        "finding": report["findings"][0]["evidence"],
        "request": report["evidence"][0]["request_summary"] if run.evidence else {},
        "response": report["evidence"][0]["response_summary"] if run.evidence else {},
    }[sink]
    assert token not in json.dumps(exposed, allow_nan=False, ensure_ascii=False)


def test_overlapping_credentials_cannot_reappear_in_replacement_values():
    secret_profile = TargetProfile.from_cli(
        "https://target.test/chat",
        headers={"X-Api-Key": "0", "Authorization": "Bearer <redacted>",
                 "X-Other": "redacted", "X-Marker": "safe-"},
    )
    cap = capability()
    cap.payload = {"tested": 0, "documents_injected": 123}
    report = json_report.generate_run_report(outcome(capabilities=[cap]), secret_profile)
    validate_report(report)
    dynamic = json.dumps(report["capabilities"][0]["payload"], allow_nan=False)
    for token in ("0", "<redacted>", "redacted", "safe-"):
        assert token not in dynamic


def test_unrepresentable_credential_collision_fails_before_output(tmp_path):
    secret_profile = TargetProfile.from_cli("https://g.xyz:80/", headers={"X-Api-Key": "0"})
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="^invalid report data$"):
        json_report.generate_run_report(outcome(), secret_profile, output)
    assert not output.exists()


def unknown_technique_outcome(technique_id="RD-9999"):
    finding = Finding(technique_id, "untrusted-source-name", "high", "", finding_id="fi-1")
    cap = capability(findings=[finding])
    cap.technique_ids = (technique_id,)
    cap.started_at = cap.ended_at = "2026-09-29T12:00:00+00:00"
    cap.controls, cap.errors, cap.summary = [], [], ""
    return outcome(capabilities=[cap])


@pytest.mark.parametrize("technique_id", ["RD-9999", "unregistered-technique"])
@pytest.mark.parametrize("tokens", [
    ("<redacted>",), ("redacted",), ("<red",), ("acted>",),
    ("<redacted>", "redacted", "acted>", "safe-"),
])
def test_unknown_technique_name_withholds_marker_credentials_in_all_outputs(
    technique_id, tokens, tmp_path,
):
    secret_profile = TargetProfile.from_cli(
        "https://target.test/chat",
        headers={f"X-Fixture-{index}": token for index, token in enumerate(tokens)},
    )
    run = unknown_technique_outcome(technique_id)
    before = asdict(run)
    output = tmp_path / "report.json"
    report = json_report.generate_run_report(run, secret_profile, output)
    validate_report(report)
    serialized = json.dumps(report, allow_nan=False, ensure_ascii=False)
    assert json.loads(serialized.encode("utf-8").decode("utf-8")) == report
    written = output.read_text(encoding="utf-8")
    assert json.loads(written) == report
    for rendered in (serialized, written, json_report.format_summary(report, color=False),
                     json_report.format_summary(report, color=True)):
        for token in (*tokens, "untrusted-source-name"):
            assert token not in rendered
    assert report["summary"]["total_findings"] == report["summary"]["inferred"] == 1
    assert report["capabilities"][0]["finding_ids"] == [report["findings"][0]["finding_id"]]
    assert report["capabilities"][0]["technique_ids"] == [report["findings"][0]["technique_id"]]
    assert json_report.generate_run_report(run, secret_profile, output) == report
    assert output.read_text(encoding="utf-8") == written
    assert asdict(run) == before


@pytest.mark.parametrize("substring", [False, True])
def test_unknown_technique_name_rechecks_candidate_against_all_credentials(substring, tmp_path):
    run = unknown_technique_outcome()
    before = asdict(run)
    first_profile = TargetProfile.from_cli("https://target.test/chat", headers={"X-Key": "<redacted>"})
    first = json_report.generate_run_report(run, first_profile)["findings"][0]["technique_name"]
    assert "<redacted>" not in first
    collision = first[-16:] if substring else first
    secret_profile = TargetProfile.from_cli(
        "https://target.test/chat", headers={"X-Key": "<redacted>", "X-Other": collision},
    )
    output = tmp_path / "report.json"
    report = json_report.generate_run_report(run, secret_profile, output)
    validate_report(report)
    for rendered in (json.dumps(report, allow_nan=False, ensure_ascii=False),
                     output.read_text(encoding="utf-8"), json_report.format_summary(report, color=False)):
        assert "<redacted>" not in rendered
        assert collision not in rendered
    assert json_report.generate_run_report(run, secret_profile) == report
    assert asdict(run) == before


@pytest.mark.parametrize("existing_output", [False, True])
def test_unknown_technique_name_rejects_exhausted_candidates_before_output(
    existing_output, tmp_path, monkeypatch,
):
    collision = "b" * 64
    secret_profile = TargetProfile.from_cli(
        "https://target.test/chat", headers={"X-Key": "<redacted>", "X-Other": collision},
    )
    run = unknown_technique_outcome()
    before = asdict(run)
    output = tmp_path / "report.json" if existing_output else tmp_path / "new" / "report.json"
    if existing_output:
        output.write_bytes(b"original report")
    # Exhaust the candidate source deterministically; only the unknown-name marker
    # collides in this fixture, so rejection must come from that field's final pass.
    monkeypatch.setattr(json_report.hashlib, "sha256", lambda value: SimpleNamespace(hexdigest=lambda: collision))
    for _ in range(2):
        with pytest.raises(ValueError, match="^invalid report data$"):
            json_report.generate_run_report(run, secret_profile, output)
        if existing_output:
            assert output.read_bytes() == b"original report"
        else:
            assert not output.parent.exists()
    assert asdict(run) == before


@pytest.mark.parametrize("phase", list(PHASE_METADATA))
@pytest.mark.parametrize("id_collision", [False, True])
def test_registered_technique_name_remains_exact_with_unknown_fallback(phase, id_collision, tmp_path):
    metadata = PHASE_METADATA[phase]
    technique_id = metadata.technique_ids[0]
    headers = {"X-Key": "<redacted>", "X-Name": metadata.title}
    if id_collision:
        headers["X-Id"] = technique_id
    secret_profile = TargetProfile.from_cli("https://target.test/chat", headers=headers)
    run = unknown_technique_outcome()
    cap = run.run.capabilities[0]
    cap.findings.insert(0, Finding(technique_id, "untrusted-source-name", "high", "", finding_id="fi-2"))
    cap.technique_ids = (technique_id, "RD-9999")
    before = asdict(run)
    output = tmp_path / "report.json"
    report = json_report.generate_run_report(run, secret_profile, output)
    validate_report(report)
    assert report["findings"][0]["technique_name"] == metadata.title
    assert metadata.title in json_report.format_summary(report, color=False)
    for token in headers.values():
        assert token not in report["findings"][1]["technique_name"]
    assert report["capabilities"][0]["technique_ids"] == [item["technique_id"] for item in report["findings"]]
    assert report["capabilities"][0]["finding_ids"] == [item["finding_id"] for item in report["findings"]]
    assert report["summary"]["total_findings"] == report["summary"]["inferred"] == 2
    public_row = next(row for row in report["implementation_status"] if row["phase"] == phase)
    assert public_row["technique_ids"] == list(metadata.technique_ids)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert json_report.generate_run_report(run, secret_profile) == report
    assert asdict(run) == before


@pytest.mark.parametrize("nested", [False, True])
def test_legacy_adapter_rejects_hostile_finding_without_metaclass_hooks(nested):
    from ragdrag.core.probe import ProbeResult

    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append("hash")
            raise RuntimeError("synthetic-private-marker")

        def __eq__(cls, other):
            calls.append("eq")
            raise RuntimeError("synthetic-private-marker")

    class Scalar(metaclass=Meta):
        pass

    legacy = ProbeResult("https://target.test/chat")
    legacy.findings = ([Finding("RD-0201", "Probe", "high", "safe", evidence={"matched": Scalar()})]
                       if nested else [Scalar()])
    if nested:
        report = json_report.generate_report(legacy)
        validate_report(report)
        assert report["findings"][0]["evidence"] == {}
    else:
        with pytest.raises(ValueError, match="^invalid report data$"):
            json_report.generate_report(legacy)
    assert calls == []
