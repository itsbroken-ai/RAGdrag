"""Behavioral contracts for the legacy phase compatibility layer."""

import json

import httpx
import pytest

from ragdrag.adapters.chat import ChatResponseError
from ragdrag.core.exfiltrate import ExfilFinding, ExfiltrateResult
from ragdrag.core.fingerprint import FingerprintResult
from ragdrag.core.poison import PoisonResult
from ragdrag.engine.capability import CapabilityContext
from ragdrag.engine.evidence import EvidenceStore
from ragdrag.engine.models import (
    CapabilityStatus, CleanupState, EvidenceState, ExitCode, Finding, ImpactLevel, OutcomeCode,
)
from ragdrag.engine.mutations import MutationLedger
from ragdrag.engine.phases import (
    PHASE_METADATA, PhaseOptions, build_phase_capabilities, normalize_finding,
    run_engagement,
)
from ragdrag.engine.profile import TargetProfile
from ragdrag.engine.runner import EngagementRunner
from ragdrag.engine.transport import OriginBoundClient


def test_phase_metadata_and_deduplicated_order():
    assert [PHASE_METADATA[key].impact for key in ("R1", "R2", "R3", "R4", "R5", "R6")] == [
        ImpactLevel.ACTIVE, ImpactLevel.ACTIVE, ImpactLevel.ACTIVE,
        ImpactLevel.MUTATING, ImpactLevel.MUTATING, ImpactLevel.ACTIVE,
    ]
    selected = build_phase_capabilities(["R3", "R1", "R3"], PhaseOptions())
    assert [phase.metadata.capability_id for phase in selected] == ["phase.r3", "phase.r1"]
    with pytest.raises(ValueError, match="Unknown phase"):
        build_phase_capabilities(["R7"], PhaseOptions())


def test_exfiltration_normalization_preserves_safe_fields_and_classifies_sensitivity():
    source = ExfilFinding(
        "RD-0301", "Direct extraction", "high", "internal_doc",
        "Internal marker observed", "query", evidence={"matched": True},
    )
    normalized = normalize_finding(source)
    assert isinstance(normalized, Finding)
    assert normalized.technique_id == "RD-0301"
    assert normalized.detail == "<redacted>"
    assert normalized.confidence == "high"
    assert normalized.severity == "medium"
    assert normalized.evidence == {"matched": True, "sensitivity": "internal_doc"}
    assert normalized.confidence_basis == "legacy phase heuristic; inspect referenced evidence"


def test_existing_evidence_state_and_basis_survive_normalization():
    source = Finding(
        "RD-0101", "Presence", "low", "Latency observed",
        evidence={"delta_ms": 210}, evidence_state=EvidenceState.OBSERVED,
        confidence_basis="Measured response latency",
    )
    normalized = normalize_finding(source)
    assert normalized.evidence_state is EvidenceState.OBSERVED
    assert normalized.confidence_basis == "legacy phase heuristic; inspect referenced evidence"
    assert normalized.evidence == {"delta_ms": 210}


def test_unrecognized_legacy_text_is_not_retained_in_evidence():
    source = Finding(
        "RD-0101", "Presence", "low", "Latency observed",
        evidence={"debug_note": "test-only-secret", "delta_ms": 210},
    )
    normalized = normalize_finding(source)
    assert normalized.evidence == {"delta_ms": 210}


def test_credential_bearing_legacy_detail_is_withheld():
    source = Finding(
        "RD-0101", "Presence", "low", "API key: test-only-secret",
    )
    normalized = normalize_finding(source)
    assert normalized.detail == "<redacted>"


def test_unknown_sensitivity_text_is_not_retained():
    source = ExfilFinding(
        "RD-0301", "Direct extraction", "medium", "test-only-secret",
        "Marker observed", "query",
    )
    normalized = normalize_finding(source)
    assert normalized.evidence["sensitivity"] == "<redacted>"


def test_non_string_sensitivity_is_withheld_without_reflecting_it():
    source = ExfilFinding("RD-0301", "Direct extraction", "medium", ["SYNTHETIC_PRIVATE_SAMPLE"], "private", "query")
    normalized = normalize_finding(source)
    assert normalized.evidence["sensitivity"] == "<redacted>"
    assert normalized.detail == "<redacted>"


@pytest.mark.parametrize("sensitivity", ["credential", "internal_doc", "system-prompt", "secret", "evasion", "unknown"])
def test_sensitive_legacy_classification_never_retains_raw_detail(sensitivity):
    source = ExfilFinding(
        "RD-0301", "Direct extraction", "medium", sensitivity,
        "SYNTHETIC_PRIVATE_SAMPLE", "query",
    )
    normalized = normalize_finding(source)
    assert "SYNTHETIC_PRIVATE_SAMPLE" not in repr(normalized)
    assert normalized.detail == "<redacted>"


@pytest.mark.parametrize("field,value", [
    ("technique_id", "SYNTHETIC_PRIVATE_SAMPLE"),
    ("confidence", "SYNTHETIC_PRIVATE_SAMPLE"),
    ("severity", "bogus"),
    ("evidence_state", "bogus"),
])
def test_invalid_finding_fields_fail_with_fixed_diagnostic(field, value):
    source = Finding("RD-0101", "Presence", "low", "safe")
    setattr(source, field, value)
    with pytest.raises(ValueError) as caught:
        normalize_finding(source)
    assert str(caught.value) == "legacy finding has invalid canonical fields"


def test_normalization_detaches_custom_strings_and_untrusted_metadata():
    class HostileString(str):
        def __repr__(self):
            raise AssertionError("custom representation called")

        def __str__(self):
            raise AssertionError("custom string called")

    source = Finding("RD-0101", "Presence", "low", "safe", evidence={"delta_ms": 210})
    source.technique_name = HostileString("SYNTHETIC_PRIVATE_SAMPLE")
    source.confidence_basis = HostileString("SYNTHETIC_PRIVATE_SAMPLE")
    normalized = normalize_finding(source)
    assert type(normalized.technique_name) is str
    assert type(normalized.confidence_basis) is str
    assert "SYNTHETIC_PRIVATE_SAMPLE" not in repr(normalized)


def test_evidence_keys_must_be_exact_trusted_strings():
    class HostileKey(str):
        def __repr__(self):
            raise AssertionError("custom representation called")

    source = Finding("RD-0101", "Presence", "low", "safe", evidence={
        HostileKey("matched"): True,
        "delta_ms": 210,
        "SYNTHETIC_PRIVATE_SAMPLE": 1,
    })
    normalized = normalize_finding(source)
    assert normalized.evidence == {"delta_ms": 210}


def test_normalize_finding_rejects_hostile_metaclass_without_hooks():
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

    with pytest.raises(ValueError, match="^legacy finding has invalid canonical fields$"):
        normalize_finding(Scalar())
    assert calls == []


def test_normalize_finding_drops_hostile_retained_evidence_scalar_without_hooks():
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

    source = Finding("RD-0201", "Probe", "high", "safe",
                     evidence={"matched": Scalar(), "match_count": 2})
    normalized = normalize_finding(source)
    assert normalized.evidence == {"match_count": 2}
    assert calls == []


def test_untrusted_payload_keys_and_nested_maps_are_withheld(monkeypatch):
    source = FingerprintResult("https://target.test/chat")
    source.to_dict = lambda: {
        "SYNTHETIC_PRIVATE_SAMPLE": True,
        "timing_stats": {"SYNTHETIC_PRIVATE_SAMPLE": 1},
        "findings": [{"detail": "SYNTHETIC_PRIVATE_SAMPLE"}],
        "rag_detected": False,
    }
    monkeypatch.setattr("ragdrag.engine.phases.run_full_fingerprint", lambda *args, **kwargs: source)
    profile = TargetProfile.from_cli("https://target.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R1"], PhaseOptions())[0].execute(context)
    assert "SYNTHETIC_PRIVATE_SAMPLE" not in repr(result)
    assert result.payload == {"rag_detected": False, "findings": "<redacted>"}


def test_one_legacy_call_records_sanitized_evidence_and_payload(monkeypatch):
    calls = []
    source = ExfiltrateResult("https://target.test/chat", total_queries=1)
    source.findings.append(ExfilFinding(
        "RD-0301", "Direct extraction", "high", "credential",
        "Found credential: test-only-secret", "query", raw_response="test-only-secret",
        evidence={"matches": ["test-only-secret"], "match_count": 1},
    ))

    def legacy(target, client, **kwargs):
        calls.append((target, client, kwargs))
        return source

    monkeypatch.setattr("ragdrag.engine.phases.run_exfiltrate", legacy)
    profile = TargetProfile.from_cli("https://target.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R3"], PhaseOptions(deep=False))[0].execute(context)
    assert len(calls) == 1
    assert calls[0][1] is client
    assert result.status is CapabilityStatus.COMPLETED
    assert len(result.findings) == 1
    assert result.findings[0].severity == "high"
    assert result.evidence_ids == [context.evidence.items[0].evidence_id]
    assert "test-only-secret" not in repr(result)


@pytest.mark.parametrize("phase", ["R4", "R5"])
def test_missing_cleanup_blocks_mutating_phase_before_legacy_execution(monkeypatch, phase):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy phase executed without cleanup")

    monkeypatch.setattr(
        "ragdrag.engine.phases.run_poison" if phase == "R4" else "ragdrag.engine.phases.run_hijack",
        forbidden,
    )
    profile = TargetProfile.from_cli("https://target.test/chat", impact_ceiling=ImpactLevel.MUTATING)
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities([phase], PhaseOptions())[0].execute(context)
    assert result.status is CapabilityStatus.BLOCKED
    assert result.outcomes == [OutcomeCode.BLOCKED_BY_CONTROL]


def test_fingerprint_wrapper_invokes_legacy_once(monkeypatch):
    calls = []

    def legacy(target, client, **kwargs):
        calls.append((target, client, kwargs))
        return FingerprintResult(target)

    monkeypatch.setattr("ragdrag.engine.phases.run_full_fingerprint", legacy)
    profile = TargetProfile.from_cli("https://target.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R1"], PhaseOptions(scan_ports=False))[0].execute(context)
    assert len(calls) == 1
    assert calls[0][1] is client
    assert calls[0][2]["scan_ports"] is False
    assert result.status is CapabilityStatus.COMPLETED


def test_unrecognized_legacy_payload_text_is_not_retained(monkeypatch):
    source = FingerprintResult("https://target.test/chat")
    source.to_dict = lambda: {"debug_note": "test-only-secret", "rag_detected": False}
    monkeypatch.setattr("ragdrag.engine.phases.run_full_fingerprint", lambda *args, **kwargs: source)
    profile = TargetProfile.from_cli("https://target.test/chat")
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R1"], PhaseOptions())[0].execute(context)
    assert result.payload == {"rag_detected": False}


def test_r6_configured_history_and_session_reach_real_chat_requests(monkeypatch):
    monkeypatch.setattr("ragdrag.core.evade.assess_substitution_bypass", lambda *args, **kwargs: [])
    monkeypatch.setattr("ragdrag.core.evade.assess_obfuscation_effectiveness", lambda *args, **kwargs: [])
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "Safe response"})

    profile = TargetProfile.from_cli(
        "https://target.test/chat", query_field="prompt", response_field="answer",
        history_field="history", session_field="session_id", session_id="session-test",
    )
    options = PhaseOptions(
        query_field="prompt", response_field="answer", history_field="history",
        session_field="session_id", session_id="session-test",
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(respond)) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R6"], options)[0].execute(context)
    assert result.status is CapabilityStatus.COMPLETED
    assert len(sent) == 10
    assert all(item["session_id"] == "session-test" for item in sent)
    assert sent[0]["history"] == [{"role": "user", "content": sent[0]["prompt"]}]
    assert sent[1]["history"][:2] == [
        {"role": "user", "content": sent[0]["prompt"]},
        {"role": "assistant", "content": "Safe response"},
    ]


def test_r6_uses_profile_conversation_fields_when_options_are_default(monkeypatch):
    monkeypatch.setattr("ragdrag.core.evade.assess_substitution_bypass", lambda *args, **kwargs: [])
    monkeypatch.setattr("ragdrag.core.evade.assess_obfuscation_effectiveness", lambda *args, **kwargs: [])
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "Safe response"})

    profile = TargetProfile.from_cli(
        "https://target.test/chat", query_field="prompt", response_field="answer",
        history_field="history", session_field="session_id", session_id="session-test",
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(respond)) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R6"], PhaseOptions())[0].execute(context)
    assert result.status is CapabilityStatus.COMPLETED
    assert sent[0]["prompt"]
    assert sent[0]["session_id"] == "session-test"
    assert sent[1]["history"][1] == {"role": "assistant", "content": "Safe response"}


def test_r6_without_real_conversation_state_reports_not_applicable(monkeypatch):
    monkeypatch.setattr("ragdrag.core.evade.assess_substitution_bypass", lambda *args, **kwargs: [])
    monkeypatch.setattr("ragdrag.core.evade.assess_obfuscation_effectiveness", lambda *args, **kwargs: [])
    profile = TargetProfile.from_cli("https://target.test/chat", response_field="answer")
    with OriginBoundClient(
        profile, transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"answer": "Safe response"})),
    ) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R6"], PhaseOptions(response_field="answer"))[0].execute(context)
    assert result.status is CapabilityStatus.PARTIAL
    assert result.outcomes == [OutcomeCode.CAPABILITY_NOT_APPLICABLE]


def test_engagement_uses_one_mutating_phase_and_ledger_cleanup(monkeypatch):
    calls = []
    profile = TargetProfile.from_cli(
        "https://target.test/chat", impact_ceiling=ImpactLevel.MUTATING,
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=transport),
    )

    def legacy(target, client, **kwargs):
        calls.append((target, client))
        record = kwargs["mutations"].record_attempt("phase.r4", target, "POST", "doc-1", "DELETE")
        kwargs["mutations"].mark_created(record.mutation_id)
        kwargs["mutations"].register_cleanup(record.mutation_id, lambda: CleanupState.REMOVED)
        return PoisonResult(target)

    monkeypatch.setattr("ragdrag.engine.phases.run_poison", legacy)
    outcome = run_engagement(
        profile, ["R4"], PhaseOptions(
            cleanup_url="https://target.test/documents/{id}",
            established_controls=frozenset({"baseline", "negative-control", "cleanup-verification"}),
        ),
    )
    assert len(calls) == 1
    assert outcome.run.capabilities[0].status is CapabilityStatus.COMPLETED
    assert outcome.run.capabilities[0].controls == ["baseline", "cleanup-verification", "negative-control"]
    assert outcome.run.capabilities[0].cleanup_state is CleanupState.REMOVED
    assert len(outcome.mutations) == 1
    assert outcome.mutations[0].state is CleanupState.REMOVED


def test_cleanup_route_does_not_authorize_unestablished_mutation_controls(monkeypatch):
    profile = TargetProfile.from_cli("https://target.test/chat", impact_ceiling=ImpactLevel.MUTATING)
    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    monkeypatch.setattr("ragdrag.engine.phases.run_poison", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unsafe call")))
    outcome = run_engagement(profile, ["R4"], PhaseOptions(cleanup_url="https://target.test/documents/{id}"))
    assert outcome.run.capabilities[0].status is CapabilityStatus.BLOCKED
    assert outcome.mutations == []


def test_established_controls_reject_unknown_and_custom_names():
    class CustomName(str):
        pass

    with pytest.raises(ValueError, match="established controls must be known"):
        PhaseOptions(established_controls=frozenset({"not-a-control"}))
    with pytest.raises(ValueError, match="established controls must be known"):
        PhaseOptions(established_controls=frozenset({CustomName("baseline")}))


@pytest.mark.parametrize("missing", ["baseline", "negative-control", "cleanup-verification"])
def test_each_mutation_control_is_required(monkeypatch, missing):
    profile = TargetProfile.from_cli("https://target.test/chat", impact_ceiling=ImpactLevel.MUTATING)
    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    monkeypatch.setattr("ragdrag.engine.phases.run_poison", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unsafe call")))
    controls = frozenset({"baseline", "negative-control", "cleanup-verification"} - {missing})
    outcome = run_engagement(
        profile, ["R4"], PhaseOptions(cleanup_url="https://target.test/documents/{id}", established_controls=controls),
    )
    assert outcome.run.capabilities[0].status is CapabilityStatus.BLOCKED


def test_invalid_cleanup_route_blocks_even_with_established_controls(monkeypatch):
    profile = TargetProfile.from_cli("https://target.test/chat", impact_ceiling=ImpactLevel.MUTATING)
    monkeypatch.setattr("ragdrag.engine.phases.run_poison", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unsafe call")))
    options = PhaseOptions(
        cleanup_url="https://other.test/docs/{id}",
        established_controls=frozenset({"baseline", "negative-control", "cleanup-verification"}),
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R4"], options)[0].execute(context)
    assert result.status is CapabilityStatus.BLOCKED
    assert result.outcomes == [OutcomeCode.BLOCKED_BY_CONTROL]


def test_non_string_cleanup_route_blocks_before_mutation(monkeypatch):
    profile = TargetProfile.from_cli("https://target.test/chat", impact_ceiling=ImpactLevel.MUTATING)
    monkeypatch.setattr("ragdrag.engine.phases.run_poison", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unsafe call")))
    options = PhaseOptions(
        cleanup_url=42,
        established_controls=frozenset({"baseline", "negative-control", "cleanup-verification"}),
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R4"], options)[0].execute(context)
    assert result.status is CapabilityStatus.BLOCKED
    assert result.outcomes == [OutcomeCode.BLOCKED_BY_CONTROL]


def test_engagement_policy_blocks_mutating_phase_below_ceiling(monkeypatch):
    profile = TargetProfile.from_cli("https://target.test/chat", impact_ceiling=ImpactLevel.ACTIVE)
    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    monkeypatch.setattr(
        "ragdrag.engine.phases.run_poison",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mutating phase executed")),
    )
    outcome = run_engagement(
        profile, ["R4"], PhaseOptions(cleanup_url="https://target.test/documents/{id}"),
    )
    assert outcome.run.capabilities[0].status is CapabilityStatus.BLOCKED
    assert outcome.mutations == []


@pytest.mark.parametrize(("phase_status", "expected"), [
    (401, OutcomeCode.AUTHENTICATION_REQUIRED),
    (403, OutcomeCode.AUTHENTICATION_REQUIRED),
    (404, OutcomeCode.INVALID_TARGET),
    (429, OutcomeCode.RATE_LIMITED),
    (500, OutcomeCode.INDETERMINATE),
])
def test_swallowed_phase_http_failure_is_typed_and_not_clean(monkeypatch, phase_status, expected):
    monkeypatch.setattr("ragdrag.core.exfiltrate.EXTRACTION_QUERIES", ["benign local probe"])
    seen = []

    def respond(request):
        seen.append(request.method)
        return httpx.Response(200) if request.method == "HEAD" else httpx.Response(phase_status, text="failure")

    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(respond)),
    )
    profile = TargetProfile.from_cli("https://target.test/chat")
    outcome = run_engagement(profile, ["R3"], PhaseOptions(deep=False))
    assert seen == ["HEAD", "POST"]
    assert outcome.run.capabilities[0].status is CapabilityStatus.PARTIAL
    assert outcome.run.capabilities[0].outcomes == [expected]
    assert outcome.run.exit_code is not ExitCode.CLEAN


def test_swallowed_phase_connection_failure_is_unreachable(monkeypatch):
    monkeypatch.setattr("ragdrag.core.exfiltrate.EXTRACTION_QUERIES", ["benign local probe"])

    def respond(request):
        if request.method == "HEAD":
            return httpx.Response(200)
        raise httpx.ConnectError("synthetic connection failure", request=request)

    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(respond)),
    )
    outcome = run_engagement(TargetProfile.from_cli("https://target.test/chat"), ["R3"], PhaseOptions(deep=False))
    assert outcome.run.capabilities[0].status is CapabilityStatus.PARTIAL
    assert outcome.run.capabilities[0].outcomes == [OutcomeCode.UNREACHABLE]


@pytest.mark.parametrize("swallow", [True, False])
def test_hostile_transport_exception_cannot_make_phase_clean(monkeypatch, swallow):
    hooks = []
    phase_calls = []

    class HostileHTTPError(httpx.HTTPError):
        @property
        def __class__(self):
            hooks.append("class")
            raise httpx.HTTPError("synthetic nested failure")

        def __str__(self):
            hooks.append("str")
            raise AssertionError("hostile exception rendering")

        def __repr__(self):
            hooks.append("repr")
            raise AssertionError("hostile exception representation")

    def respond(request):
        if request.method == "HEAD":
            return httpx.Response(200)
        raise HostileHTTPError("synthetic request failure")

    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(respond)),
    )

    def legacy(target, client, **kwargs):
        phase_calls.append(1)
        if swallow:
            try:
                client.post(target, json={"query": "inert"})
            except httpx.HTTPError:
                pass
        else:
            client.post(target, json={"query": "inert"})
        return FingerprintResult(target)

    monkeypatch.setattr("ragdrag.engine.phases.run_full_fingerprint", legacy)
    outcome = run_engagement(TargetProfile.from_cli("https://target.test/chat"), ["R1"], PhaseOptions())
    assert phase_calls == [1]
    assert outcome.run.requests_used == 2
    assert outcome.run.capabilities[0].status is CapabilityStatus.PARTIAL
    assert outcome.run.capabilities[0].outcomes == [OutcomeCode.INDETERMINATE]
    assert outcome.run.exit_code is not ExitCode.CLEAN
    assert "str" not in hooks and "repr" not in hooks
    assert hooks == []


def test_malformed_configured_phase_response_is_unsupported(monkeypatch):
    monkeypatch.setattr("ragdrag.core.exfiltrate.EXTRACTION_QUERIES", ["benign local probe"])

    def respond(request):
        return httpx.Response(200) if request.method == "HEAD" else httpx.Response(200, text="not-json")

    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(respond)),
    )
    profile = TargetProfile.from_cli("https://target.test/chat", response_field="answer")
    outcome = run_engagement(profile, ["R3"], PhaseOptions(deep=False))
    assert outcome.run.capabilities[0].outcomes == [OutcomeCode.UNSUPPORTED_RESPONSE]
    assert outcome.run.exit_code is not ExitCode.CLEAN


def test_successful_negative_phase_assessment_can_complete_clean(monkeypatch):
    monkeypatch.setattr("ragdrag.core.exfiltrate.EXTRACTION_QUERIES", ["benign local probe"])

    def respond(request):
        return httpx.Response(200) if request.method == "HEAD" else httpx.Response(200, json={"answer": "ordinary reply"})

    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(respond)),
    )
    profile = TargetProfile.from_cli("https://target.test/chat", response_field="answer")
    outcome = run_engagement(profile, ["R3"], PhaseOptions(deep=False))
    assert outcome.run.capabilities[0].status is CapabilityStatus.COMPLETED
    assert outcome.run.exit_code is ExitCode.CLEAN


def test_target_get_failure_inside_phase_is_not_clean(monkeypatch):
    profile = TargetProfile.from_cli("https://target.test/chat")

    def respond(request):
        return httpx.Response(200) if request.method == "HEAD" else httpx.Response(404)

    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(respond)),
    )

    def legacy(target, client, **kwargs):
        client.get(target)
        return FingerprintResult(target)

    monkeypatch.setattr("ragdrag.engine.phases.run_full_fingerprint", legacy)
    outcome = run_engagement(profile, ["R1"], PhaseOptions())
    assert outcome.run.capabilities[0].status is CapabilityStatus.PARTIAL
    assert outcome.run.capabilities[0].outcomes == [OutcomeCode.INVALID_TARGET]


def test_propagated_http_status_keeps_observed_authentication_outcome(monkeypatch):
    profile = TargetProfile.from_cli("https://target.test/chat")

    def respond(request):
        return httpx.Response(200) if request.method == "HEAD" else httpx.Response(401)

    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(respond)),
    )

    def legacy(target, client, **kwargs):
        client.get(target).raise_for_status()
        return FingerprintResult(target)

    monkeypatch.setattr("ragdrag.engine.phases.run_full_fingerprint", legacy)
    outcome = run_engagement(profile, ["R1"], PhaseOptions())
    assert outcome.run.capabilities[0].status is CapabilityStatus.PARTIAL
    assert outcome.run.capabilities[0].outcomes == [OutcomeCode.AUTHENTICATION_REQUIRED]


@pytest.mark.parametrize("statuses,expected", [
    ([401, 429], [OutcomeCode.AUTHENTICATION_REQUIRED, OutcomeCode.RATE_LIMITED]),
    ([429, 401], [OutcomeCode.RATE_LIMITED, OutcomeCode.AUTHENTICATION_REQUIRED]),
    ([401, 401], [OutcomeCode.AUTHENTICATION_REQUIRED]),
])
def test_propagated_later_http_failure_preserves_all_distinct_observations(monkeypatch, statuses, expected):
    replies = iter(statuses)
    profile = TargetProfile.from_cli("https://target.test/chat")

    def respond(request):
        return httpx.Response(200) if request.method == "HEAD" else httpx.Response(next(replies))

    monkeypatch.setattr(
        "ragdrag.engine.phases.OriginBoundClient",
        lambda selected: OriginBoundClient(selected, transport=httpx.MockTransport(respond)),
    )

    def legacy(target, client, **kwargs):
        client.get(target)
        client.get(target).raise_for_status()
        return FingerprintResult(target)

    monkeypatch.setattr("ragdrag.engine.phases.run_full_fingerprint", legacy)
    outcome = run_engagement(profile, ["R1"], PhaseOptions())
    assert outcome.run.capabilities[0].status is CapabilityStatus.PARTIAL
    assert outcome.run.capabilities[0].outcomes == expected


@pytest.mark.parametrize("last_failure,expected", [
    ("transport", [OutcomeCode.AUTHENTICATION_REQUIRED, OutcomeCode.UNREACHABLE]),
    ("chat", [OutcomeCode.AUTHENTICATION_REQUIRED, OutcomeCode.UNSUPPORTED_RESPONSE]),
])
def test_propagated_typed_failure_appends_to_prior_observations(monkeypatch, last_failure, expected):
    profile = TargetProfile.from_cli("https://target.test/chat")
    calls = []

    def respond(request):
        calls.append(request)
        if last_failure == "transport":
            if len(calls) == 1:
                return httpx.Response(401)
            raise httpx.ConnectError("synthetic", request=request)
        return httpx.Response(401)

    with OriginBoundClient(profile, transport=httpx.MockTransport(respond)) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")

        def legacy(target, phase_client, **kwargs):
            phase_client.get(target)
            if last_failure == "transport":
                phase_client.get(target)
            raise ChatResponseError(OutcomeCode.UNSUPPORTED_RESPONSE, "synthetic")

        monkeypatch.setattr("ragdrag.engine.phases.run_full_fingerprint", legacy)
        result = build_phase_capabilities(["R1"], PhaseOptions())[0].execute(context)
    assert result.status is CapabilityStatus.PARTIAL
    assert result.outcomes == expected


def test_hostile_typed_exception_metadata_still_reports_and_cleans_up(monkeypatch):
    class HostileNestedError(ValueError):
        def __str__(self):
            raise RuntimeError("hostile nested rendering")

    class HostileChatError(ChatResponseError):
        @property
        def outcome(self):
            raise HostileNestedError()

        @outcome.setter
        def outcome(self, value):
            pass

    profile = TargetProfile.from_cli("https://target.test/chat", impact_ceiling=ImpactLevel.MUTATING)
    source = PoisonResult(profile.target_url)
    source.to_dict = lambda: (_ for _ in ()).throw(HostileChatError(OutcomeCode.UNSUPPORTED_RESPONSE, "synthetic"))
    cleanup_calls = []

    def legacy(target, client, **kwargs):
        record = kwargs["mutations"].record_attempt("phase.r4", target, "POST", "doc-1", "DELETE")
        kwargs["mutations"].mark_created(record.mutation_id)
        kwargs["mutations"].register_cleanup(
            record.mutation_id, lambda: (cleanup_calls.append(1), CleanupState.REMOVED)[1],
        )
        return source

    monkeypatch.setattr("ragdrag.engine.phases.run_poison", legacy)
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        runner = EngagementRunner(
            profile, client,
            authorized_controls={"tests/test_poison.py", "baseline", "negative-control", "cleanup-verification"},
        )
        run = runner.run(build_phase_capabilities(["R4"], PhaseOptions(
            cleanup_url="https://target.test/docs/{id}",
            established_controls=frozenset({"baseline", "negative-control", "cleanup-verification"}),
        )))
    assert cleanup_calls == [1]
    assert run.capabilities[0].status in {CapabilityStatus.FAILED, CapabilityStatus.PARTIAL}
    assert run.capabilities[0].outcomes == [OutcomeCode.INDETERMINATE]
    assert run.capabilities[0].cleanup_state is CleanupState.REMOVED
    assert "hostile nested rendering" not in repr(run)


def test_untrusted_typed_exception_outcome_is_never_stringified(monkeypatch):
    class HostileValue:
        def __str__(self):
            raise AssertionError("hostile outcome rendering")

        def __repr__(self):
            raise AssertionError("hostile outcome representation")

    class HostileChatError(ChatResponseError):
        @property
        def outcome(self):
            return HostileValue()

        @outcome.setter
        def outcome(self, value):
            pass

    profile = TargetProfile.from_cli("https://target.test/chat")
    monkeypatch.setattr(
        "ragdrag.engine.phases.run_full_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(HostileChatError(OutcomeCode.UNSUPPORTED_RESPONSE, "synthetic")),
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        result = build_phase_capabilities(["R1"], PhaseOptions())[0].execute(context)
    assert result.status is CapabilityStatus.PARTIAL
    assert result.outcomes == [OutcomeCode.INDETERMINATE]
    assert "hostile outcome" not in repr(result)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_typed_exception_metadata_interruption_is_not_swallowed(monkeypatch, interrupt):
    class InterruptingChatError(ChatResponseError):
        @property
        def outcome(self):
            raise interrupt()

        @outcome.setter
        def outcome(self, value):
            pass

    profile = TargetProfile.from_cli("https://target.test/chat")
    monkeypatch.setattr(
        "ragdrag.engine.phases.run_full_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(InterruptingChatError(OutcomeCode.UNSUPPORTED_RESPONSE, "synthetic")),
    )
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        context = CapabilityContext(profile, client, EvidenceStore(), MutationLedger(), "started")
        with pytest.raises(interrupt):
            build_phase_capabilities(["R1"], PhaseOptions())[0].execute(context)


def test_hostile_legacy_serialization_error_keeps_report_and_cleanup(monkeypatch):
    class HostileError(ValueError):
        def __str__(self):
            raise RuntimeError("hostile exception rendering")

    profile = TargetProfile.from_cli("https://target.test/chat", impact_ceiling=ImpactLevel.MUTATING)
    source = PoisonResult(profile.target_url)

    def bad_to_dict():
        raise HostileError()

    source.to_dict = bad_to_dict
    cleanup_calls = []

    def legacy(target, client, **kwargs):
        record = kwargs["mutations"].record_attempt("phase.r4", target, "POST", "doc-1", "DELETE")
        kwargs["mutations"].mark_created(record.mutation_id)
        kwargs["mutations"].register_cleanup(
            record.mutation_id, lambda: (cleanup_calls.append(1), CleanupState.REMOVED)[1],
        )
        return source

    monkeypatch.setattr("ragdrag.engine.phases.run_poison", legacy)
    with OriginBoundClient(profile, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        runner = EngagementRunner(
            profile, client,
            authorized_controls={"tests/test_poison.py", "baseline", "negative-control", "cleanup-verification"},
        )
        run = runner.run(build_phase_capabilities(["R4"], PhaseOptions(
            cleanup_url="https://target.test/docs/{id}",
            established_controls=frozenset({"baseline", "negative-control", "cleanup-verification"}),
        )))
    assert cleanup_calls == [1]
    assert run.capabilities[0].status is CapabilityStatus.FAILED
    assert run.capabilities[0].cleanup_state is CleanupState.REMOVED
    assert "hostile exception rendering" not in repr(run)
