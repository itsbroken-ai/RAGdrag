"""Generic, origin-bound capability execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

import httpx

from ragdrag.engine.capability import (
    Capability,
    CapabilityContext,
    CapabilityRegistry,
    InvalidConfiguration,
    SafetyPolicy,
)
from ragdrag.engine.evidence import EvidenceStore
from ragdrag.engine.models import (
    CapabilityMetadata,
    CapabilityResult,
    CapabilityStatus,
    CleanupState,
    EvidenceItem,
    ExitCode,
    Finding,
    ImpactLevel,
    MutationRecord,
    OutcomeCode,
    RunResult,
)
from ragdrag.engine.mutations import MutationLedger
from ragdrag.engine.profile import TargetProfile
from ragdrag.engine.transport import HTTPClient, RequestBudgetExceeded, ResponseTooLarge


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EngagementInterrupted(KeyboardInterrupt):
    def __init__(
        self,
        run: RunResult,
        evidence: list[EvidenceItem],
        mutations: list[MutationRecord],
    ) -> None:
        super().__init__("engagement interrupted")
        self.run = run
        self.evidence = evidence
        self.mutations = mutations


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    outcome: OutcomeCode | None = None
    detail: str = ""


def classify_preflight(response: httpx.Response) -> PreflightResult:
    status = response.status_code
    if 200 <= status < 300 or status == 405:
        return PreflightResult(True)
    if status in {401, 403}:
        return PreflightResult(False, OutcomeCode.AUTHENTICATION_REQUIRED, "target requires authentication")
    if status == 404:
        return PreflightResult(False, OutcomeCode.INVALID_TARGET, "target path returned 404")
    if status == 429:
        return PreflightResult(False, OutcomeCode.RATE_LIMITED, "target rate limited preflight")
    return PreflightResult(False, OutcomeCode.INDETERMINATE, f"preflight returned HTTP {status}")


def choose_exit_code(results: Iterable[CapabilityResult], mutations: MutationLedger) -> ExitCode:
    results = list(results)
    if mutations.has_unresolved:
        return ExitCode.UNRESOLVED_CLEANUP
    if any(result.status is CapabilityStatus.FAILED for result in results):
        return ExitCode.EXECUTION_FAILURE
    if any(result.status in {CapabilityStatus.PARTIAL, CapabilityStatus.BLOCKED} for result in results):
        return ExitCode.PARTIAL
    if any(OutcomeCode.INDETERMINATE in result.outcomes for result in results):
        return ExitCode.PARTIAL
    if any(result.findings for result in results):
        return ExitCode.FINDINGS
    return ExitCode.CLEAN


def choose_run_status(exit_code: ExitCode) -> str:
    if exit_code in {ExitCode.CLEAN, ExitCode.FINDINGS}:
        return "completed"
    if exit_code is ExitCode.PARTIAL:
        return "partial"
    return "failed"


def _exception_outcome(exc: Exception) -> OutcomeCode:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return OutcomeCode.UNREACHABLE
    if isinstance(exc, ResponseTooLarge):
        return OutcomeCode.UNSUPPORTED_RESPONSE
    return OutcomeCode.INDETERMINATE


def _cleanup_state(records: list[MutationRecord]) -> CleanupState:
    if not records:
        return CleanupState.NOT_CREATED
    states = {record.state for record in records}
    if states & {CleanupState.ACTIVE, CleanupState.UNKNOWN, CleanupState.UNRESOLVED}:
        return CleanupState.UNRESOLVED
    if CleanupState.RESTORED in states:
        return CleanupState.RESTORED
    if CleanupState.REMOVED in states:
        return CleanupState.REMOVED
    return CleanupState.NOT_CREATED


def _validate_string_list(value: object, field: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"capability result has invalid {field}")
    return value


def _validate_result(
    result: object,
    metadata: CapabilityMetadata,
    evidence: EvidenceStore,
    mutations: MutationLedger,
) -> CapabilityResult:
    if (
        type(result) is not CapabilityResult
        or type(result.capability_id) is not str
        or result.capability_id != metadata.capability_id
        or type(result.status) is not CapabilityStatus
        or result.impact is not metadata.impact
        or type(result.technique_ids) is not tuple
        or any(type(item) is not str for item in result.technique_ids)
        or result.technique_ids != metadata.technique_ids
    ):
        raise ValueError("capability returned an invalid result")
    if type(result.started_at) is not str or type(result.ended_at) is not str:
        raise ValueError("capability result has invalid timestamps")
    if type(result.trials) is not int:
        raise ValueError("capability result has invalid trial count")
    if type(result.summary) is not str:
        raise ValueError("capability result has invalid summary")
    if type(result.payload) is not dict or any(type(key) is not str for key in result.payload):
        raise ValueError("capability result has invalid payload")
    evidence_ids = _validate_string_list(result.evidence_ids, "evidence IDs")
    mutation_ids = _validate_string_list(result.mutation_ids, "mutation IDs")
    _validate_string_list(result.controls, "controls")
    _validate_string_list(result.errors, "errors")
    if type(result.outcomes) is not list or any(type(item) is not OutcomeCode for item in result.outcomes):
        raise ValueError("capability result has invalid outcomes")
    if type(result.findings) is not list or any(type(item) is not Finding for item in result.findings):
        raise ValueError("capability result has invalid findings")
    known_evidence = {item.evidence_id for item in evidence.items}
    known_mutations = {item.mutation_id for item in mutations.records}
    if any(item not in known_evidence for item in evidence_ids):
        raise ValueError("capability result references unknown evidence")
    if any(item not in known_mutations for item in mutation_ids):
        raise ValueError("capability result references unknown mutation")
    return result


class EngagementRunner:
    def __init__(
        self,
        profile: TargetProfile,
        client: HTTPClient,
        *,
        authorized_controls: Iterable[str] = (),
    ) -> None:
        self.profile = profile
        self.client = client
        self.policy = SafetyPolicy(profile.impact_ceiling, authorized_controls=authorized_controls)
        self._evidence = EvidenceStore()
        self._mutations = MutationLedger()

    @property
    def evidence(self) -> EvidenceStore:
        """The stable evidence store for this runner's current run."""
        return self._evidence

    @property
    def mutations(self) -> MutationLedger:
        """The stable mutation ledger for this runner's current run."""
        return self._mutations

    def preflight(self) -> PreflightResult:
        client_profile = getattr(self.client, "profile", None)
        if client_profile != self.profile:
            raise InvalidConfiguration("client profile does not match engagement profile")
        try:
            response = self.client.head(self.profile.target_url, follow_redirects=False)
        except httpx.TransportError as exc:
            return PreflightResult(False, OutcomeCode.UNREACHABLE, str(exc))
        except RequestBudgetExceeded as exc:
            return PreflightResult(False, OutcomeCode.INDETERMINATE, str(exc))
        except (ResponseTooLarge, httpx.DecodingError) as exc:
            return PreflightResult(False, OutcomeCode.UNSUPPORTED_RESPONSE, str(exc))
        except httpx.HTTPError as exc:
            return PreflightResult(False, OutcomeCode.INDETERMINATE, str(exc))
        return classify_preflight(response)

    def run(self, capabilities: Iterable[Capability]) -> RunResult:
        selected = CapabilityRegistry(capabilities).select()
        started = _now()
        run_id = str(uuid4())
        requests_start = getattr(self.client, "requests_used", 0)
        results: list[CapabilityResult] = []
        self.evidence.items.clear()
        self.mutations.clear()
        context = CapabilityContext(self.profile, self.client, self.evidence, self.mutations, started)
        interrupted = False
        interruption_cause: BaseException | None = None
        preflight_failed = False
        preflight_exit: ExitCode | None = None

        try:
            try:
                check = self.preflight()
            except InvalidConfiguration:
                raise
            except Exception as exc:
                preflight_failed = True
                check = PreflightResult(False, OutcomeCode.INDETERMINATE, f"{type(exc).__name__}: {exc}")
            except BaseException as exc:
                interrupted = True
                interruption_cause = exc
                check = PreflightResult(False, OutcomeCode.INDETERMINATE, type(exc).__name__)
            if not check.ok:
                preflight_exit = (
                    ExitCode.INVALID_CONFIGURATION
                    if check.outcome is OutcomeCode.INVALID_TARGET
                    else ExitCode.EXECUTION_FAILURE if preflight_failed else ExitCode.PARTIAL
                )
                results.append(
                    CapabilityResult(
                        "preflight", (),
                        CapabilityStatus.PARTIAL if interrupted else CapabilityStatus.FAILED if preflight_failed else CapabilityStatus.BLOCKED,
                        ImpactLevel.PASSIVE,
                        started, _now(), outcomes=[check.outcome or OutcomeCode.INDETERMINATE],
                        errors=[check.detail] if check.detail else [],
                    )
                )
            else:
                for capability in selected:
                    metadata = capability.metadata
                    capability_started = _now()
                    allowed, reason = self.policy.authorize(metadata)
                    if not allowed:
                        results.append(
                            CapabilityResult(
                                metadata.capability_id, metadata.technique_ids,
                                CapabilityStatus.BLOCKED, metadata.impact if type(metadata.impact) is ImpactLevel else ImpactLevel.PASSIVE,
                                capability_started, _now(), outcomes=[OutcomeCode.BLOCKED_BY_CONTROL],
                                errors=[reason],
                            )
                        )
                        continue
                    evidence_start = len(self.evidence.items)
                    mutation_start = len(self.mutations.records)
                    recovered_from_exception = False
                    try:
                        result = _validate_result(
                            capability.execute(context), metadata, self.evidence, self.mutations
                        )
                        result.evidence_ids = list(dict.fromkeys(
                            [*result.evidence_ids, *(item.evidence_id for item in self.evidence.items[evidence_start:])]
                        ))
                        result.mutation_ids = list(dict.fromkeys(
                            [*result.mutation_ids, *(item.mutation_id for item in self.mutations.records[mutation_start:])]
                        ))
                    except Exception as exc:
                        recovered_from_exception = True
                        result = CapabilityResult(
                            metadata.capability_id, metadata.technique_ids,
                            CapabilityStatus.FAILED, metadata.impact,
                            capability_started, _now(),
                            outcomes=[_exception_outcome(exc)],
                            errors=[f"{type(exc).__name__}: {exc}"],
                        )
                    except BaseException as exc:
                        recovered_from_exception = True
                        interrupted = True
                        interruption_cause = exc
                        result = CapabilityResult(
                            metadata.capability_id, metadata.technique_ids,
                            CapabilityStatus.PARTIAL, metadata.impact,
                            capability_started, _now(), outcomes=[OutcomeCode.INDETERMINATE],
                            errors=[type(exc).__name__],
                        )
                    if recovered_from_exception:
                        result.evidence_ids = [item.evidence_id for item in self.evidence.items[evidence_start:]]
                        result.mutation_ids = [item.mutation_id for item in self.mutations.records[mutation_start:]]
                    results.append(result)
                    if interrupted:
                        break
        finally:
            self.mutations.cleanup_all()
        if self.mutations.cleanup_interruption is not None:
            interrupted = True
            if interruption_cause is None:
                interruption_cause = self.mutations.cleanup_interruption

        records_by_id = {record.mutation_id: record for record in self.mutations.records}
        for result in results:
            result.cleanup_state = _cleanup_state(
                [records_by_id[item] for item in result.mutation_ids if item in records_by_id]
            )

        exit_code = (
            ExitCode.UNRESOLVED_CLEANUP if self.mutations.has_unresolved else ExitCode.PARTIAL
        ) if interrupted else (preflight_exit or choose_exit_code(results, self.mutations))
        run = RunResult(
            run_id, self.profile.target_url, started, _now(), results, exit_code,
            status="interrupted" if interrupted else choose_run_status(exit_code),
            requests_used=getattr(self.client, "requests_used", 0) - requests_start,
            evidence=list(self.evidence.items), mutations=list(self.mutations.records),
        )
        if interrupted:
            raise EngagementInterrupted(run, list(self.evidence.items), list(self.mutations.records)) from interruption_cause
        return run
