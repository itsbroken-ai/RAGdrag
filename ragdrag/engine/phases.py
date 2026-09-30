"""Compatibility capabilities for the existing R1–R6 phase entry points."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any

import httpx

from ragdrag.adapters.chat import ChatResponseError
from ragdrag.core.evade import run_evade
from ragdrag.core.exfiltrate import ExfilFinding, run_exfiltrate
from ragdrag.core.fingerprint import run_full_fingerprint
from ragdrag.core.hijack import run_hijack
from ragdrag.core.poison import CleanupStrategy, run_poison
from ragdrag.core.probe import run_probe
from ragdrag.engine.capability import CapabilityContext
from ragdrag.engine.models import (
    CapabilityMetadata, CapabilityResult, CapabilityStatus, EvidenceItem,
    EvidenceState, Finding, ImpactLevel, MutationRecord, OutcomeCode, RunResult,
    VALID_CONFIDENCE,
)
from ragdrag.engine.profile import TargetProfile, canonical_origin
from ragdrag.engine.redaction import REDACTED
from ragdrag.engine.runner import EngagementRunner
from ragdrag.engine.transport import OriginBoundClient, PhaseObservation, RequestBudgetExceeded, ResponseTooLarge

MAX_CLEANUP_URL_LENGTH = 2048


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PhaseOptions:
    query_field: str = "query"
    response_field: str | None = None
    history_field: str | None = None
    session_field: str | None = None
    session_id: str | None = None
    scan_ports: bool = True
    deep: bool = True
    ingest_url: str | None = None
    cleanup_url: str | None = None
    listener_host: str | None = None
    callback_url: str | None = None
    camouflage: bool = False
    established_controls: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        allowed = {"baseline", "negative-control", "cleanup-verification"}
        if type(self.established_controls) is not frozenset or any(
            type(item) is not str or item not in allowed for item in self.established_controls
        ):
            raise ValueError("established controls must be known control names")


PHASE_METADATA: dict[str, CapabilityMetadata] = {
    "R1": CapabilityMetadata("phase.r1", ("RD-0101", "RD-0102"), "Fingerprint", "validated", ImpactLevel.ACTIVE, ("chat",), 3, False, validation_test="tests/test_fingerprint.py"),
    "R2": CapabilityMetadata("phase.r2", ("RD-0201", "RD-0202", "RD-0203", "RD-0204", "RD-0205"), "Probe", "validated", ImpactLevel.ACTIVE, ("chat",), 3, False, validation_test="tests/test_probe.py"),
    "R3": CapabilityMetadata("phase.r3", ("RD-0301", "RD-0302"), "Exfiltrate", "validated", ImpactLevel.ACTIVE, ("chat",), 3, False, validation_test="tests/test_exfiltrate.py"),
    "R4": CapabilityMetadata("phase.r4", ("RD-0401", "RD-0402", "RD-0403", "RD-0404"), "Poison", "validated", ImpactLevel.MUTATING, ("chat",), 3, True, ("baseline", "negative-control", "cleanup-verification"), "tests/test_poison.py"),
    "R5": CapabilityMetadata("phase.r5", ("RD-0501", "RD-0502", "RD-0503", "RD-0504"), "Hijack", "validated", ImpactLevel.MUTATING, ("chat",), 3, True, ("baseline", "negative-control", "cleanup-verification"), "tests/test_hijack.py"),
    "R6": CapabilityMetadata("phase.r6", ("RD-0601", "RD-0603", "RD-0604"), "Evade", "validated", ImpactLevel.ACTIVE, ("chat",), 3, False, validation_test="tests/test_evade.py"),
}


_KNOWN_TECHNIQUES = {
    technique_id: metadata.title
    for metadata in PHASE_METADATA.values()
    for technique_id in metadata.technique_ids
}
_SEVERITIES = {"critical", "high", "medium", "low", "info"}
_SENSITIVITIES = {"credential", "internal_doc", "system-prompt", "secret", "evasion", "general"}
_SAFE_EVIDENCE_FIELDS = {
    "matched", "delta_ms", "knowledge_mean_ms", "general_mean_ms", "match_count",
    "verified", "match_ratio", "dominance_ratio", "total_queries", "redirected_queries",
    "redirect_ratio", "documents_injected", "documents_attempted", "saturation_pct",
    "marker_appearances", "steps_completed", "sensitive_info_found", "tested",
    "blocked", "status_code", "camouflaged", "estimated_top_k", "best_score",
    "worst_score", "auth_required", "found", "guardrail_bypass", "direct_blocked",
    "bypass_blocked", "state_mechanism",
}
_SAFE_PAYLOAD_FIELDS = {
    "rag_detected", "total_queries", "guardrail_detected", "trap_active",
    "instruction_injected", "substitutions_tested", "substitutions_bypassed",
    "camouflage_effective", "obfuscation_effective", "context_saturation_pct",
    "redirected_queries", "tool_calls_triggered", "persistence_verified",
    "chunk_size_estimate", "similarity_threshold", "retrieval_count", "dominance_score",
}
_REDACTED_PAYLOAD_FIELDS = {"findings", "guardrail_bypass_findings", "injected_documents"}


def _safe_scalar(value: object) -> bool:
    return (value is None or type(value) is bool or type(value) is int
            or (type(value) is float and isfinite(value)))


def _safe_evidence(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("legacy finding has invalid canonical fields")
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str or key not in _SAFE_EVIDENCE_FIELDS:
            continue
        if key == "state_mechanism":
            if type(item) is str and item in {"history", "session", "cookie"}:
                safe[key] = item
        elif _safe_scalar(item):
            safe[key] = item
    return safe


def normalize_finding(value: Finding | ExfilFinding) -> Finding:
    """Detach one legacy finding, excluding raw extraction samples."""
    if type(value) is not Finding and type(value) is not ExfilFinding:
        raise ValueError("legacy finding has invalid canonical fields")
    if (
        type(value.technique_id) is not str
        or value.technique_id not in _KNOWN_TECHNIQUES
        or type(value.confidence) is not str
        or value.confidence not in VALID_CONFIDENCE
    ):
        raise ValueError("legacy finding has invalid canonical fields")
    if type(value) is Finding and (
        type(value.severity) is not str
        or value.severity not in _SEVERITIES
        or type(value.evidence_state) is not EvidenceState
    ):
        raise ValueError("legacy finding has invalid canonical fields")
    evidence = _safe_evidence(value.evidence)
    sensitivity = value.sensitivity if type(value) is ExfilFinding else None
    safe_sensitivity = sensitivity if type(sensitivity) is str and sensitivity in _SENSITIVITIES else None
    if sensitivity is not None:
        evidence["sensitivity"] = safe_sensitivity or REDACTED
    severity = (
        "high" if safe_sensitivity in {"credential", "system-prompt", "secret"}
        else "medium" if safe_sensitivity == "internal_doc"
        else value.severity if type(value) is Finding else "medium"
    )
    return Finding(
        technique_id=value.technique_id,
        technique_name=_KNOWN_TECHNIQUES[value.technique_id],
        confidence=value.confidence,
        detail=REDACTED,
        evidence=evidence,
        severity=severity,
        evidence_state=value.evidence_state if type(value) is Finding else EvidenceState.INFERRED,
        confidence_basis="legacy phase heuristic; inspect referenced evidence",
    )


def _safe_payload(value: object) -> dict[str, Any]:
    """Preserve aggregate shape while withholding untrusted legacy details."""
    if type(value) is not dict:
        raise ValueError("legacy phase returned an invalid payload")
    safe: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str:
            continue
        if key in _REDACTED_PAYLOAD_FIELDS:
            safe[key] = REDACTED
        elif key in _SAFE_PAYLOAD_FIELDS and _safe_scalar(item):
            safe[key] = item
    return safe


def _status_outcome(status: int) -> OutcomeCode:
    if status in (401, 403):
        return OutcomeCode.AUTHENTICATION_REQUIRED
    if status == 404:
        return OutcomeCode.INVALID_TARGET
    if status == 429:
        return OutcomeCode.RATE_LIMITED
    return OutcomeCode.INDETERMINATE


def _classify_phase_exception(exc: Exception) -> tuple[OutcomeCode | None, bool]:
    """Read exception metadata only behind a fixed, non-reflective boundary."""
    try:
        bases = type.mro(type(exc))
        if any(base is ChatResponseError for base in bases):
            candidate = exc.outcome
            return (
                candidate if type(candidate) is OutcomeCode else OutcomeCode.INDETERMINATE,
                True,
            )
        if any(base is httpx.HTTPStatusError for base in bases):
            status = exc.response.status_code
            return (_status_outcome(status) if type(status) is int else OutcomeCode.INDETERMINATE, True)
        if any(base is httpx.TransportError for base in bases):
            return OutcomeCode.UNREACHABLE, True
        if any(base is ResponseTooLarge or base is httpx.DecodingError for base in bases):
            return OutcomeCode.UNSUPPORTED_RESPONSE, True
        if any(base is RequestBudgetExceeded for base in bases):
            return OutcomeCode.INDETERMINATE, True
    except Exception:
        return OutcomeCode.INDETERMINATE, True
    return None, False


class PhaseCapability:
    def __init__(self, phase: str, options: PhaseOptions) -> None:
        self.phase = phase
        self.options = options
        self.metadata = PHASE_METADATA[phase]

    def _blocked(self, started: str, reason: str, outcome: OutcomeCode) -> CapabilityResult:
        return CapabilityResult(
            self.metadata.capability_id, self.metadata.technique_ids,
            CapabilityStatus.BLOCKED, self.metadata.impact,
            started, _now(), outcomes=[outcome], errors=[reason],
        )

    def _failed(self, started: str, outcomes: list[OutcomeCode], *, request_failure: bool) -> CapabilityResult:
        return CapabilityResult(
            self.metadata.capability_id, self.metadata.technique_ids,
            CapabilityStatus.PARTIAL if request_failure else CapabilityStatus.FAILED,
            self.metadata.impact, started, _now(),
            outcomes=outcomes,
            errors=["legacy phase request failed" if request_failure else "legacy phase conversion failed"],
        )

    def _run_legacy(
        self, context: CapabilityContext, common: dict[str, Any], cleanup: CleanupStrategy | None,
    ) -> Any:
        options = self.options
        target = context.profile.target_url
        client = context.client
        if self.phase == "R1":
            return run_full_fingerprint(target, client, scan_ports=options.scan_ports, **common)
        if self.phase == "R2":
            return run_probe(target, client, depth="full" if options.deep else "quick", **common)
        if self.phase == "R3":
            return run_exfiltrate(target, client, deep=options.deep, **common)
        if self.phase == "R4":
            return run_poison(
                target, client, listener_host=options.listener_host, ingest_url=options.ingest_url,
                mutations=context.mutations, cleanup_strategy=cleanup, **common,
            )
        if self.phase == "R5":
            return run_hijack(
                target, client, callback_url=options.callback_url, ingest_url=options.ingest_url,
                use_camouflage=options.camouflage, mutations=context.mutations,
                cleanup_strategy=cleanup, **common,
            )
        return run_evade(
            target, client, history_field=options.history_field or context.profile.history_field,
            session_field=options.session_field or context.profile.session_field,
            session_id=options.session_id or context.profile.session_id,
            **common,
        )

    def execute(self, context: CapabilityContext) -> CapabilityResult:
        started = _now()
        metadata = self.metadata
        options = self.options
        profile = context.profile
        client = context.client
        if metadata.impact is ImpactLevel.MUTATING:
            if profile.impact_ceiling is not ImpactLevel.MUTATING:
                return self._blocked(started, "mutating impact is not authorized", OutcomeCode.BLOCKED_BY_CONTROL)
            if not set(metadata.required_controls).issubset(options.established_controls):
                return self._blocked(started, "required mutation controls are not established", OutcomeCode.BLOCKED_BY_CONTROL)
            if (type(options.cleanup_url) is not str or not options.cleanup_url
                    or len(options.cleanup_url) > MAX_CLEANUP_URL_LENGTH):
                return self._blocked(started, "cleanup URL is required", OutcomeCode.BLOCKED_BY_CONTROL)
            try:
                cleanup = CleanupStrategy(options.cleanup_url)
                cleanup_url = cleanup.url_for("preflight-id")
                if canonical_origin(cleanup_url) not in profile.approved_origins:
                    raise ValueError("cleanup origin is not approved")
            except ValueError:
                return self._blocked(started, "cleanup URL is invalid or outside approved origins", OutcomeCode.BLOCKED_BY_CONTROL)
        else:
            cleanup = None
        common = {
            "query_field": options.query_field if options.query_field != "query" else profile.query_field,
            "response_field": options.response_field or profile.response_field,
        }
        observations: PhaseObservation | None = None
        try:
            with client.observe_phase(common["response_field"]) as observations:
                legacy = self._run_legacy(context, common, cleanup)

            findings = [normalize_finding(item) for item in legacy.findings]
            if self.phase == "R3":
                findings.extend(normalize_finding(item) for item in legacy.guardrail_bypass_findings)
            payload = _safe_payload(legacy.to_dict())
            no_stateful_r6 = self.phase == "R6" and not any(
                finding.technique_id == "RD-0604" for finding in findings
            )
            outcomes = list(observations.snapshot()) if observations is not None else []
            if no_stateful_r6:
                outcomes.append(OutcomeCode.CAPABILITY_NOT_APPLICABLE)
            evidence_ids = []
            for index, finding in enumerate(findings, start=1):
                evidence = context.evidence.record(
                    capability_id=metadata.capability_id,
                    trial_id=f"{metadata.capability_id}.{index}",
                    state=finding.evidence_state,
                    confidence_basis=finding.confidence_basis,
                    request_summary={"phase": self.phase, "technique_id": finding.technique_id},
                    response_summary={"finding": True, "severity": finding.severity, "confidence": finding.confidence},
                )
                evidence_ids.append(evidence.evidence_id)
            return CapabilityResult(
                metadata.capability_id, metadata.technique_ids,
                CapabilityStatus.PARTIAL if outcomes else CapabilityStatus.COMPLETED,
                metadata.impact, started, _now(),
                trials=1, controls=sorted(options.established_controls) if metadata.creates_mutations else [],
                findings=findings, evidence_ids=evidence_ids,
                outcomes=outcomes,
                summary=f"{self.phase} completed with {len(findings)} finding(s)",
                payload=payload,
            )
        except Exception as exc:
            observed = list(observations.snapshot()) if observations is not None else []
            outcomes = observed.copy()
            exception_outcome, request_failure = _classify_phase_exception(exc)
            if exception_outcome is not None and exception_outcome not in outcomes:
                outcomes.append(exception_outcome)
            if not outcomes:
                outcomes = [OutcomeCode.INDETERMINATE]
            return self._failed(started, outcomes, request_failure=request_failure or bool(observed))


def build_phase_capabilities(phases: list[str], options: PhaseOptions) -> list[PhaseCapability]:
    selected: list[PhaseCapability] = []
    seen: set[str] = set()
    for phase in phases:
        if phase not in PHASE_METADATA:
            raise ValueError(f"Unknown phase: {phase}")
        if phase not in seen:
            selected.append(PhaseCapability(phase, options))
            seen.add(phase)
    return selected


@dataclass
class EngagementOutcome:
    run: RunResult
    evidence: list[EvidenceItem]
    mutations: list[MutationRecord]


def run_engagement(profile: TargetProfile, phases: list[str], options: PhaseOptions) -> EngagementOutcome:
    capabilities = build_phase_capabilities(phases, options)
    controls = {item.metadata.validation_test for item in capabilities if item.metadata.validation_test}
    controls.update(options.established_controls)
    with OriginBoundClient(profile) as client:
        runner = EngagementRunner(profile, client, authorized_controls=controls)
        run = runner.run(capabilities)
        return EngagementOutcome(run, list(runner.evidence.items), list(runner.mutations.records))
