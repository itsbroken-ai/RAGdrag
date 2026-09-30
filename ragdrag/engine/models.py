"""Stable data contracts shared by engine capabilities and reporters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import isfinite
from typing import Any, Literal, get_args
from uuid import uuid4

Confidence = Literal["high", "medium", "low"]
Severity = Literal["critical", "high", "medium", "low", "info"]
VALID_CONFIDENCE: tuple[str, ...] = get_args(Confidence)


class ImpactLevel(str, Enum):
    PASSIVE = "passive"
    ACTIVE = "active-non-mutating"
    MUTATING = "mutating"


class EvidenceState(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    VALIDATED = "validated"


class CapabilityStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


class OutcomeCode(str, Enum):
    UNREACHABLE = "unreachable"
    AUTHENTICATION_REQUIRED = "authentication-required"
    UNSUPPORTED_RESPONSE = "unsupported-response"
    INVALID_TARGET = "invalid-target"
    RATE_LIMITED = "rate-limited"
    BLOCKED_BY_CONTROL = "blocked-by-control"
    CAPABILITY_NOT_APPLICABLE = "capability-not-applicable"
    INDETERMINATE = "indeterminate"


class CleanupState(str, Enum):
    NOT_CREATED = "not-created"
    ACTIVE = "active"
    REMOVED = "removed"
    RESTORED = "restored"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


class ExitCode(IntEnum):
    CLEAN = 0
    FINDINGS = 1
    PARTIAL = 2
    INVALID_CONFIGURATION = 3
    EXECUTION_FAILURE = 4
    UNRESOLVED_CLEANUP = 5


@dataclass(frozen=True)
class RequestBudget:
    max_requests: int = 100
    timeout_seconds: float = 30.0
    max_response_bytes: int = 2_097_152
    max_redirects: int = 5
    max_concurrency: int = 4

    def __post_init__(self) -> None:
        for name, minimum in (
            ("max_requests", 1),
            ("max_response_bytes", 1),
            ("max_redirects", 0),
            ("max_concurrency", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

        timeout = self.timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")


@dataclass
class Finding:
    technique_id: str
    technique_name: str
    confidence: Confidence
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    severity: Severity = "medium"
    evidence_state: EvidenceState = EvidenceState.INFERRED
    confidence_basis: str = ""
    affected_component: str = ""
    remediation: str = ""
    finding_id: str = field(default_factory=lambda: str(uuid4()), compare=False)

    def __post_init__(self) -> None:
        if self.confidence not in VALID_CONFIDENCE:
            raise ValueError(f"Invalid confidence {self.confidence!r}; must be one of {VALID_CONFIDENCE}")


@dataclass
class EvidenceItem:
    evidence_id: str
    capability_id: str
    trial_id: str
    timestamp: str
    state: EvidenceState
    confidence_basis: str
    request_summary: dict[str, Any]
    response_summary: dict[str, Any]
    artifact_digest: str | None = None


@dataclass
class MutationRecord:
    mutation_id: str
    capability_id: str
    target_scope: str
    operation: str
    object_id: str
    cleanup_method: str
    state: CleanupState = CleanupState.UNKNOWN
    cleanup_attempts: int = 0
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CapabilityMetadata:
    capability_id: str
    technique_ids: tuple[str, ...]
    title: str
    maturity: str
    impact: ImpactLevel
    adapter_types: tuple[str, ...]
    max_trials: int
    creates_mutations: bool
    required_controls: tuple[str, ...] = ()
    validation_test: str | None = None


@dataclass
class CapabilityResult:
    capability_id: str
    technique_ids: tuple[str, ...]
    status: CapabilityStatus
    impact: ImpactLevel
    started_at: str
    ended_at: str
    trials: int = 0
    controls: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    outcomes: list[OutcomeCode] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    mutation_ids: list[str] = field(default_factory=list)
    cleanup_state: CleanupState = CleanupState.NOT_CREATED
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: str
    target: str
    started_at: str
    ended_at: str
    capabilities: list[CapabilityResult]
    exit_code: ExitCode
    status: Literal["completed", "partial", "failed", "interrupted"] = "completed"
    requests_used: int = 0
    evidence: list[EvidenceItem] = field(default_factory=list)
    mutations: list[MutationRecord] = field(default_factory=list)
