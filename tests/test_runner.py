"""Inert contract tests for the engagement control plane."""

import asyncio
from dataclasses import asdict, replace

import httpx
import pytest

from ragdrag.engine.capability import CapabilityContext, CapabilityRegistry, SafetyPolicy
from ragdrag.engine.evidence import EvidenceStore
from ragdrag.engine.models import (
    CapabilityMetadata,
    CapabilityResult,
    CapabilityStatus,
    CleanupState,
    EvidenceState,
    ExitCode,
    Finding,
    ImpactLevel,
    OutcomeCode,
    RequestBudget,
)
from ragdrag.engine.profile import TargetProfile
from ragdrag.engine.mutations import MutationLedger
from ragdrag.engine.runner import EngagementInterrupted, EngagementRunner, InvalidConfiguration
from ragdrag.engine.transport import OriginBoundClient
from ragdrag.core.poison import CleanupStrategy, inject_document


def metadata(capability_id="synthetic.one", **changes):
    value = CapabilityMetadata(
        capability_id, ("RD-0001",), "Synthetic", "validated",
        ImpactLevel.ACTIVE, ("synthetic",), 1, False,
    )
    return replace(value, **changes)


def profile(**kwargs):
    return TargetProfile.from_cli("https://target.test/path", **kwargs)


def run_with_transport(capabilities=(), *, target=None, handler=None):
    target = target or profile()
    handler = handler or (lambda request: httpx.Response(200))
    with OriginBoundClient(target, transport=httpx.MockTransport(handler)) as client:
        return EngagementRunner(target, client).run(capabilities)


def test_runner_stores_are_read_only_stable_and_reset_between_runs():
    target = profile()
    with OriginBoundClient(target, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        runner = EngagementRunner(target, client)
        evidence = runner.evidence
        mutations = runner.mutations
        with pytest.raises(AttributeError):
            runner.evidence = EvidenceStore()
        with pytest.raises(AttributeError):
            runner.mutations = MutationLedger()
        runner.run([])
        assert runner.evidence is evidence
        assert runner.mutations is mutations
        evidence.record(
            capability_id="test", trial_id="one", state=EvidenceState.OBSERVED,
            confidence_basis="test", request_summary={}, response_summary={},
        )
        mutations.record_attempt("test", target.target_url, "POST", "doc-1", "DELETE")
        runner.run([])
        assert runner.evidence is evidence and evidence.items == []
        assert runner.mutations is mutations and mutations.records == []


def result_for(context, capability_id="synthetic.one", **changes):
    value = CapabilityResult(
        capability_id, ("RD-0001",), CapabilityStatus.COMPLETED,
        ImpactLevel.ACTIVE, context.started_at, context.started_at,
    )
    return replace(value, **changes)


def test_policy_blocks_write_risk_beyond_ceiling():
    policy = SafetyPolicy(ImpactLevel.ACTIVE)
    for risky in (
        metadata(impact=ImpactLevel.MUTATING),
        metadata(creates_mutations=True),
    ):
        allowed, _ = policy.authorize(risky)
        assert allowed is False


def test_policy_requires_explicit_opt_in_and_fails_closed_for_unknown_metadata():
    restricted = metadata(required_controls=("approved-control",), validation_test="approved-test")
    assert SafetyPolicy(ImpactLevel.ACTIVE).authorize(restricted)[0] is False
    assert SafetyPolicy(
        ImpactLevel.ACTIVE, authorized_controls={"approved-control", "approved-test"}
    ).authorize(restricted)[0] is True
    assert SafetyPolicy(ImpactLevel.ACTIVE).authorize(metadata(impact="unknown"))[0] is False
    assert SafetyPolicy(ImpactLevel.ACTIVE).authorize(None)[0] is False


def test_registry_selection_is_ordered_and_rejects_unknown_or_duplicate_ids():
    class Synthetic:
        def __init__(self, name):
            self.metadata = metadata(name)

        def execute(self, context):
            return result_for(context, self.metadata.capability_id)

    registry = CapabilityRegistry([Synthetic("second"), Synthetic("first")])
    assert [item.metadata.capability_id for item in registry.select()] == ["second", "first"]
    assert [item.metadata.capability_id for item in registry.select(["first", "second"])] == ["first", "second"]
    with pytest.raises(InvalidConfiguration):
        registry.select(["missing"])
    with pytest.raises(InvalidConfiguration):
        registry.select(["first", "first"])
    with pytest.raises(InvalidConfiguration):
        registry.register(Synthetic("first"))


def test_denied_capability_is_blocked_without_execution():
    class Synthetic:
        metadata = metadata(impact=ImpactLevel.MUTATING, creates_mutations=True)

        def execute(self, context):
            raise AssertionError("denied capability executed")

    run = run_with_transport([Synthetic()])
    assert run.capabilities[0].status is CapabilityStatus.BLOCKED
    assert run.exit_code is ExitCode.PARTIAL


@pytest.mark.parametrize(
    ("status", "exit_code", "outcome"),
    [
        (401, ExitCode.PARTIAL, OutcomeCode.AUTHENTICATION_REQUIRED),
        (403, ExitCode.PARTIAL, OutcomeCode.AUTHENTICATION_REQUIRED),
        (404, ExitCode.INVALID_CONFIGURATION, OutcomeCode.INVALID_TARGET),
        (429, ExitCode.PARTIAL, OutcomeCode.RATE_LIMITED),
        (500, ExitCode.PARTIAL, OutcomeCode.INDETERMINATE),
    ],
)
def test_preflight_failure_is_typed_without_findings(status, exit_code, outcome):
    run = run_with_transport(handler=lambda request: httpx.Response(status))
    assert run.exit_code is exit_code
    assert len(run.capabilities) == 1
    assert run.capabilities[0].capability_id == "preflight"
    assert run.capabilities[0].outcomes == [outcome]
    assert run.capabilities[0].findings == []


@pytest.mark.parametrize("status", [200, 204, 405])
def test_reachable_preflight_uses_one_head_to_exact_target(status):
    requests = []

    def respond(request):
        requests.append((request.method, str(request.url)))
        return httpx.Response(status)

    run = run_with_transport(handler=respond)
    assert run.exit_code is ExitCode.CLEAN
    assert requests == [("HEAD", "https://target.test/path")]
    assert run.requests_used == 1


def test_preflight_redirect_does_not_follow_or_execute_capability():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://elsewhere.test"})

    run = run_with_transport(handler=respond)
    assert run.exit_code is ExitCode.PARTIAL
    assert run.capabilities[0].outcomes == [OutcomeCode.INDETERMINATE]
    assert len(requests) == 1


def test_preflight_consumes_shared_budget_and_capability_failure_is_preserved():
    class Synthetic:
        metadata = metadata()

        def execute(self, context):
            context.client.head(context.profile.target_url)
            return result_for(context)

    run = run_with_transport([Synthetic()], target=profile(budget=RequestBudget(max_requests=1)))
    assert run.exit_code is ExitCode.EXECUTION_FAILURE
    assert run.capabilities[0].status is CapabilityStatus.FAILED
    assert run.capabilities[0].errors


def test_shared_context_and_ordered_results_preserve_evidence_and_mutations():
    contexts = []

    class Synthetic:
        def __init__(self, name):
            self.metadata = metadata(name)

        def execute(self, context):
            contexts.append(context)
            evidence = context.evidence.record(
                capability_id=self.metadata.capability_id, trial_id="one",
                state=EvidenceState.OBSERVED, confidence_basis="synthetic",
                request_summary={}, response_summary={},
            )
            return result_for(context, self.metadata.capability_id, evidence_ids=[evidence.evidence_id])

    target = profile()
    with OriginBoundClient(target, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        runner = EngagementRunner(target, client)
        run = runner.run([Synthetic("first"), Synthetic("second")])
    assert [entry.capability_id for entry in run.capabilities] == ["first", "second"]
    assert contexts[0] is contexts[1]
    assert contexts[0].profile is target
    assert contexts[0].client is client
    assert len(runner.evidence.items) == 2
    assert [entry.evidence_ids[0] for entry in run.capabilities] == [item.evidence_id for item in runner.evidence.items]


def test_failure_after_mutation_runs_cleanup_and_keeps_partial_evidence():
    cleaned = []

    class Synthetic:
        metadata = metadata(impact=ImpactLevel.MUTATING, creates_mutations=True)

        def execute(self, context):
            context.evidence.record(
                capability_id="synthetic.one", trial_id="one", state=EvidenceState.OBSERVED,
                confidence_basis="synthetic", request_summary={}, response_summary={},
            )
            record = context.mutations.record_attempt("synthetic.one", context.profile.target_url, "write", "item", "undo")
            context.mutations.mark_created(record.mutation_id)
            context.mutations.register_cleanup(record.mutation_id, lambda: cleaned.append(True) or CleanupState.REMOVED)
            raise RuntimeError("synthetic failure")

    target = profile(impact_ceiling=ImpactLevel.MUTATING)
    with OriginBoundClient(target, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        runner = EngagementRunner(target, client)
        run = runner.run([Synthetic()])
    assert cleaned == [True]
    assert run.exit_code is ExitCode.EXECUTION_FAILURE
    assert run.capabilities[0].errors
    assert run.capabilities[0].evidence_ids == [runner.evidence.items[0].evidence_id]
    assert run.capabilities[0].mutation_ids == [runner.mutations.records[0].mutation_id]
    assert runner.mutations.records[0].state is CleanupState.REMOVED
    assert run.evidence == runner.evidence.items
    assert run.mutations == runner.mutations.records
    assert run.capabilities[0].cleanup_state is CleanupState.REMOVED


def test_runner_cleans_document_after_post_write_exception():
    target_url = "http://testrag.local/chat"
    ingest_url = "http://testrag.local/ingest"
    methods = []

    def handler(request):
        methods.append(request.method)
        if request.method == "HEAD":
            return httpx.Response(200)
        if request.method == "POST":
            return httpx.Response(201, json={"id": "doc-1"})
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    class InjectionThenCrash:
        metadata = CapabilityMetadata(
            "test.inject", ("RD-0401",), "Injection test", "validated",
            ImpactLevel.MUTATING, ("chat",), 1, True,
        )

        def execute(self, context):
            inject_document(
                target_url, context.client, "canary", ingest_url=ingest_url,
                mutations=context.mutations,
                cleanup_strategy=CleanupStrategy("http://testrag.local/documents/{id}"),
            )
            raise RuntimeError("stop after write")

    target = TargetProfile.from_cli(target_url, impact_ceiling=ImpactLevel.MUTATING)
    with OriginBoundClient(target, transport=httpx.MockTransport(handler)) as client:
        run = EngagementRunner(target, client).run([InjectionThenCrash()])
    assert methods == ["HEAD", "POST", "DELETE"]
    assert run.exit_code is ExitCode.EXECUTION_FAILURE
    assert run.mutations[0].state is CleanupState.REMOVED


@pytest.mark.parametrize("response,expected_methods", [
    (httpx.Response(302, headers={"Location": "/receipt"}), ["HEAD", "POST"]),
    (httpx.Response(201, json={"id": {"private": "nested-canary"}}), ["HEAD", "POST"]),
])
def test_runner_keeps_unidentified_or_redirected_write_unresolved(response, expected_methods):
    target_url = "http://testrag.local/chat"
    methods = []

    def handler(request):
        methods.append(request.method)
        if request.method == "HEAD":
            return httpx.Response(200)
        if request.url.path == "/ingest":
            return response
        if request.url.path == "/receipt":
            return httpx.Response(201, json={"id": "redirected"})
        return httpx.Response(404)

    class Injection:
        metadata = CapabilityMetadata(
            "test.inject", ("RD-0401",), "Injection test", "validated",
            ImpactLevel.MUTATING, ("chat",), 1, True,
        )

        def execute(self, context):
            inject_document(
                target_url, context.client, "canary", ingest_url="http://testrag.local/ingest",
                mutations=context.mutations,
                cleanup_strategy=CleanupStrategy("http://testrag.local/documents/{id}"),
            )
            raise RuntimeError("stop after ambiguous write")

    profile = TargetProfile.from_cli(target_url, impact_ceiling=ImpactLevel.MUTATING)
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        run = EngagementRunner(profile, client).run([Injection()])
    assert methods == expected_methods
    assert run.exit_code is ExitCode.UNRESOLVED_CLEANUP
    assert run.mutations[0].state is CleanupState.UNRESOLVED
    assert "nested-canary" not in repr(run.mutations)


@pytest.mark.parametrize("error_type,expected_exit,expected_state", [
    (httpx.ConnectTimeout, ExitCode.EXECUTION_FAILURE, CleanupState.NOT_CREATED),
    (httpx.PoolTimeout, ExitCode.EXECUTION_FAILURE, CleanupState.NOT_CREATED),
    (httpx.ReadTimeout, ExitCode.UNRESOLVED_CLEANUP, CleanupState.UNRESOLVED),
])
def test_runner_distinguishes_presend_and_ambiguous_timeout(error_type, expected_exit, expected_state):
    target_url = "http://testrag.local/chat"

    def handler(request):
        if request.method == "HEAD":
            return httpx.Response(200)
        raise error_type("synthetic")

    class Injection:
        metadata = CapabilityMetadata(
            "test.inject", ("RD-0401",), "Injection test", "validated",
            ImpactLevel.MUTATING, ("chat",), 1, True,
        )

        def execute(self, context):
            inject_document(
                target_url, context.client, "canary", ingest_url="http://testrag.local/ingest",
                mutations=context.mutations,
                cleanup_strategy=CleanupStrategy("http://testrag.local/documents/{id}"),
            )
            raise RuntimeError("stop after transport error")

    profile = TargetProfile.from_cli(target_url, impact_ceiling=ImpactLevel.MUTATING)
    with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
        run = EngagementRunner(profile, client).run([Injection()])
    assert run.exit_code is expected_exit
    assert run.mutations[0].state is expected_state


def test_unresolved_cleanup_wins_over_completed_finding():
    class Synthetic:
        metadata = metadata(impact=ImpactLevel.MUTATING, creates_mutations=True)

        def execute(self, context):
            record = context.mutations.record_attempt("synthetic.one", context.profile.target_url, "write", "item", "undo")
            context.mutations.mark_created(record.mutation_id)
            context.mutations.register_cleanup(record.mutation_id, lambda: (_ for _ in ()).throw(RuntimeError("no")))
            finding = Finding("RD-0001", "Synthetic", "high", "synthetic finding")
            return result_for(context, impact=ImpactLevel.MUTATING, findings=[finding])

    run = run_with_transport([Synthetic()], target=profile(impact_ceiling=ImpactLevel.MUTATING))
    assert run.exit_code is ExitCode.UNRESOLVED_CLEANUP
    assert run.capabilities[0].cleanup_state is CleanupState.UNRESOLVED


@pytest.mark.parametrize("cleanup_ok,expected", [(True, ExitCode.PARTIAL), (False, ExitCode.UNRESOLVED_CLEANUP)])
def test_interrupt_carries_cleaned_partial_run_and_state(cleanup_ok, expected):
    class Synthetic:
        metadata = metadata(impact=ImpactLevel.MUTATING, creates_mutations=True)

        def execute(self, context):
            context.evidence.record(
                capability_id="synthetic.one", trial_id="one", state=EvidenceState.OBSERVED,
                confidence_basis="synthetic", request_summary={}, response_summary={},
            )
            record = context.mutations.record_attempt("synthetic.one", context.profile.target_url, "write", "item", "undo")
            context.mutations.mark_created(record.mutation_id)
            context.mutations.register_cleanup(record.mutation_id, lambda: CleanupState.REMOVED if cleanup_ok else CleanupState.UNRESOLVED)
            raise KeyboardInterrupt

    target = profile(impact_ceiling=ImpactLevel.MUTATING)
    with OriginBoundClient(target, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        with pytest.raises(EngagementInterrupted) as raised:
            EngagementRunner(target, client).run([Synthetic()])
    interrupted = raised.value
    assert interrupted.run.status == "interrupted"
    assert interrupted.run.exit_code is expected
    assert interrupted.run.capabilities[0].status is CapabilityStatus.PARTIAL
    assert interrupted.run.capabilities[0].evidence_ids == [interrupted.evidence[0].evidence_id]
    assert interrupted.run.capabilities[0].mutation_ids == [interrupted.mutations[0].mutation_id]
    assert interrupted.mutations[0].state is (CleanupState.REMOVED if cleanup_ok else CleanupState.UNRESOLVED)
    assert interrupted.run.requests_used == 1


def test_invalid_profile_client_pair_raises_configuration_error():
    first = profile()
    second = TargetProfile.from_cli("https://other.test")
    with OriginBoundClient(first, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        with pytest.raises(InvalidConfiguration):
            EngagementRunner(second, client).run([])


def test_preflight_interrupt_carries_partial_run():
    target = profile()

    def interrupt(request):
        raise KeyboardInterrupt

    with OriginBoundClient(target, transport=httpx.MockTransport(interrupt)) as client:
        with pytest.raises(EngagementInterrupted) as raised:
            EngagementRunner(target, client).run([])
    assert raised.value.run.status == "interrupted"
    assert raised.value.run.exit_code is ExitCode.PARTIAL
    assert raised.value.run.capabilities[0].capability_id == "preflight"
    assert raised.value.run.requests_used == 1


@pytest.mark.parametrize("signal", [KeyboardInterrupt, asyncio.CancelledError])
def test_cleanup_interrupt_attempts_later_callbacks_and_carries_snapshot(signal):
    attempted = []

    class Synthetic:
        metadata = metadata(impact=ImpactLevel.MUTATING, creates_mutations=True)

        def execute(self, context):
            context.evidence.record(
                capability_id="synthetic.one", trial_id="one", state=EvidenceState.OBSERVED,
                confidence_basis="synthetic", request_summary={}, response_summary={},
            )
            first = context.mutations.record_attempt("synthetic.one", context.profile.target_url, "write", "one", "undo")
            second = context.mutations.record_attempt("synthetic.one", context.profile.target_url, "write", "two", "undo")
            context.mutations.mark_created(first.mutation_id)
            context.mutations.mark_created(second.mutation_id)

            def interrupted_cleanup():
                attempted.append("one")
                raise signal

            def later_cleanup():
                attempted.append("two")
                return CleanupState.REMOVED

            context.mutations.register_cleanup(first.mutation_id, interrupted_cleanup)
            context.mutations.register_cleanup(second.mutation_id, later_cleanup)
            return result_for(context, impact=ImpactLevel.MUTATING)

    target = profile(impact_ceiling=ImpactLevel.MUTATING)
    with OriginBoundClient(target, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        with pytest.raises(EngagementInterrupted) as raised:
            EngagementRunner(target, client).run([Synthetic()])
    run = raised.value.run
    assert attempted == ["one", "two"]
    assert run.exit_code is ExitCode.UNRESOLVED_CLEANUP
    assert run.status == "interrupted"
    assert [result.status for result in run.capabilities] == [CapabilityStatus.COMPLETED]
    assert run.capabilities[0].evidence_ids == [run.evidence[0].evidence_id]
    assert [record.state for record in run.mutations] == [CleanupState.UNRESOLVED, CleanupState.REMOVED]
    assert raised.value.evidence == run.evidence
    assert raised.value.mutations == run.mutations


def test_invalid_capability_result_fails_closed():
    class Synthetic:
        metadata = metadata()

        def execute(self, context):
            return result_for(context, status="unrecognized")

    run = run_with_transport([Synthetic()])
    assert run.exit_code is ExitCode.EXECUTION_FAILURE
    assert run.capabilities[0].status is CapabilityStatus.FAILED


def test_unexpected_preflight_error_is_typed_failure():
    def broken(request):
        raise RuntimeError("synthetic transport error")

    run = run_with_transport(handler=broken)
    assert run.exit_code is ExitCode.EXECUTION_FAILURE
    assert run.capabilities[0].status is CapabilityStatus.FAILED
    assert run.capabilities[0].outcomes == [OutcomeCode.INDETERMINATE]


def test_request_usage_is_for_this_run_when_client_is_reused():
    target = profile()
    with OriginBoundClient(target, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        first = EngagementRunner(target, client).run([])
        second = EngagementRunner(target, client).run([])
    assert first.requests_used == 1
    assert second.requests_used == 1


def test_completed_finding_uses_findings_exit_code():
    class Synthetic:
        metadata = metadata()

        def execute(self, context):
            return result_for(context, findings=[Finding("RD-0001", "Synthetic", "high", "synthetic")])

    run = run_with_transport([Synthetic()])
    assert run.exit_code is ExitCode.FINDINGS
    assert run.status == "completed"


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("evidence_ids", None),
        ("evidence_ids", [[]]),
        ("evidence_ids", ["unknown-evidence"]),
        ("mutation_ids", None),
        ("mutation_ids", [{}]),
        ("mutation_ids", ["unknown-mutation"]),
        ("outcomes", "indeterminate"),
        ("outcomes", ["indeterminate"]),
        ("findings", None),
        ("findings", ["invented-finding"]),
        ("errors", [1]),
        ("controls", ()),
    ],
)
@pytest.mark.parametrize("cleanup_state", [CleanupState.REMOVED, CleanupState.UNRESOLVED])
def test_malformed_result_becomes_typed_failure_after_cleanup(bad_field, bad_value, cleanup_state):
    class Synthetic:
        metadata = metadata(impact=ImpactLevel.MUTATING, creates_mutations=True)

        def execute(self, context):
            evidence = context.evidence.record(
                capability_id="synthetic.one", trial_id="one", state=EvidenceState.OBSERVED,
                confidence_basis="synthetic", request_summary={}, response_summary={},
            )
            record = context.mutations.record_attempt(
                "synthetic.one", context.profile.target_url, "write", "item", "undo"
            )
            context.mutations.mark_created(record.mutation_id)
            context.mutations.register_cleanup(record.mutation_id, lambda: cleanup_state)
            value = result_for(
                context, impact=ImpactLevel.MUTATING,
                evidence_ids=[evidence.evidence_id], mutation_ids=[record.mutation_id],
            )
            return replace(value, **{bad_field: bad_value})

    run = run_with_transport([Synthetic()], target=profile(impact_ceiling=ImpactLevel.MUTATING))
    failure = run.capabilities[0]
    assert failure.status is CapabilityStatus.FAILED
    assert failure.errors
    assert failure.evidence_ids == [run.evidence[0].evidence_id]
    assert failure.mutation_ids == [run.mutations[0].mutation_id]
    assert run.mutations[0].cleanup_attempts == 1
    assert run.mutations[0].state is cleanup_state
    assert run.exit_code is (
        ExitCode.EXECUTION_FAILURE if cleanup_state is CleanupState.REMOVED
        else ExitCode.UNRESOLVED_CLEANUP
    )


@pytest.mark.parametrize(
    "failed,has_finding,unresolved,expected",
    [
        (False, False, False, ExitCode.PARTIAL),
        (False, True, False, ExitCode.PARTIAL),
        (True, False, False, ExitCode.EXECUTION_FAILURE),
        (False, False, True, ExitCode.UNRESOLVED_CLEANUP),
    ],
)
def test_indeterminate_outcome_exit_precedence(failed, has_finding, unresolved, expected):
    class Synthetic:
        metadata = metadata(impact=ImpactLevel.MUTATING, creates_mutations=unresolved)

        def execute(self, context):
            if unresolved:
                record = context.mutations.record_attempt(
                    "synthetic.one", context.profile.target_url, "write", "item", "undo"
                )
                context.mutations.mark_created(record.mutation_id)
            findings = [Finding("RD-0001", "Synthetic", "high", "synthetic")] if has_finding else []
            return result_for(
                context, impact=ImpactLevel.MUTATING,
                status=CapabilityStatus.FAILED if failed else CapabilityStatus.COMPLETED,
                outcomes=[OutcomeCode.INDETERMINATE], findings=findings,
            )

    run = run_with_transport([Synthetic()], target=profile(impact_ceiling=ImpactLevel.MUTATING))
    assert run.exit_code is expected


@pytest.mark.parametrize("unresolved,expected", [(False, ExitCode.PARTIAL), (True, ExitCode.UNRESOLVED_CLEANUP)])
def test_prior_failure_then_interrupt_keeps_results_and_interruption_exit(unresolved, expected):
    class Fails:
        metadata = metadata("synthetic.fail")

        def execute(self, context):
            raise RuntimeError("first failed")

    class Interrupts:
        metadata = metadata("synthetic.interrupt", impact=ImpactLevel.MUTATING, creates_mutations=unresolved)

        def execute(self, context):
            if unresolved:
                record = context.mutations.record_attempt(
                    "synthetic.interrupt", context.profile.target_url, "write", "item", "undo"
                )
                context.mutations.mark_created(record.mutation_id)
            raise KeyboardInterrupt

    target = profile(impact_ceiling=ImpactLevel.MUTATING)
    with OriginBoundClient(target, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        with pytest.raises(EngagementInterrupted) as raised:
            EngagementRunner(target, client).run([Fails(), Interrupts()])
    run = raised.value.run
    assert [item.status for item in run.capabilities] == [CapabilityStatus.FAILED, CapabilityStatus.PARTIAL]
    assert run.exit_code is expected
    assert run.status == "interrupted"


@pytest.mark.parametrize("transport_error", [httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError])
def test_preflight_transport_errors_are_unreachable_without_execution(transport_error):
    called = []

    class Synthetic:
        metadata = metadata()

        def execute(self, context):
            called.append(True)
            return result_for(context)

    def broken(request):
        raise transport_error("synthetic transport failure", request=request)

    run = run_with_transport([Synthetic()], handler=broken)
    assert called == []
    assert run.requests_used == 1
    assert run.exit_code is ExitCode.PARTIAL
    assert run.capabilities[0].outcomes == [OutcomeCode.UNREACHABLE]
    assert run.capabilities[0].findings == []


def test_valid_failed_result_keeps_existing_evidence_reference():
    evidence_ids = []

    class Records:
        metadata = metadata("synthetic.record")

        def execute(self, context):
            item = context.evidence.record(
                capability_id="synthetic.record", trial_id="one", state=EvidenceState.OBSERVED,
                confidence_basis="synthetic", request_summary={}, response_summary={},
            )
            evidence_ids.append(item.evidence_id)
            return result_for(context, "synthetic.record")

    class ReportsFailure:
        metadata = metadata("synthetic.failure")

        def execute(self, context):
            return result_for(
                context, "synthetic.failure", status=CapabilityStatus.FAILED,
                evidence_ids=[evidence_ids[0]],
            )

    run = run_with_transport([Records(), ReportsFailure()])
    assert run.exit_code is ExitCode.EXECUTION_FAILURE
    assert run.capabilities[1].evidence_ids == evidence_ids


def test_oversized_preflight_response_is_unsupported():
    run = run_with_transport(
        target=profile(budget=RequestBudget(max_response_bytes=1)),
        handler=lambda request: httpx.Response(200, content=b"xx"),
    )
    assert run.requests_used == 1
    assert run.exit_code is ExitCode.PARTIAL
    assert run.capabilities[0].outcomes == [OutcomeCode.UNSUPPORTED_RESPONSE]


def test_preflight_decoding_failure_is_unsupported():
    def broken(request):
        raise httpx.DecodingError("synthetic decode failure", request=request)

    run = run_with_transport(handler=broken)
    assert run.requests_used == 1
    assert run.exit_code is ExitCode.PARTIAL
    assert run.capabilities[0].outcomes == [OutcomeCode.UNSUPPORTED_RESPONSE]


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("started_at", None),
        ("ended_at", []),
        ("trials", []),
        ("trials", True),
        ("summary", {}),
        ("payload", None),
        ("payload", lambda: (item for item in ())),
        ("payload", {1: "non-string key"}),
    ],
)
@pytest.mark.parametrize("cleanup_state", [CleanupState.REMOVED, CleanupState.UNRESOLVED])
def test_malformed_retained_result_field_recovers_serializable_run(
    bad_field, bad_value, cleanup_state
):
    class Synthetic:
        metadata = metadata(impact=ImpactLevel.MUTATING, creates_mutations=True)

        def execute(self, context):
            evidence = context.evidence.record(
                capability_id="synthetic.one", trial_id="one", state=EvidenceState.OBSERVED,
                confidence_basis="synthetic", request_summary={}, response_summary={},
            )
            record = context.mutations.record_attempt(
                "synthetic.one", context.profile.target_url, "write", "item", "undo"
            )
            context.mutations.mark_created(record.mutation_id)
            context.mutations.register_cleanup(record.mutation_id, lambda: cleanup_state)
            value = result_for(
                context, impact=ImpactLevel.MUTATING,
                evidence_ids=[evidence.evidence_id], mutation_ids=[record.mutation_id],
            )
            return replace(value, **{bad_field: bad_value() if callable(bad_value) else bad_value})

    run = run_with_transport([Synthetic()], target=profile(impact_ceiling=ImpactLevel.MUTATING))
    result = run.capabilities[0]
    assert result.status is CapabilityStatus.FAILED
    assert result.errors
    assert result.evidence_ids == [run.evidence[0].evidence_id]
    assert result.mutation_ids == [run.mutations[0].mutation_id]
    assert run.mutations[0].cleanup_attempts == 1
    assert run.mutations[0].state is cleanup_state
    assert run.exit_code is (
        ExitCode.EXECUTION_FAILURE if cleanup_state is CleanupState.REMOVED
        else ExitCode.UNRESOLVED_CLEANUP
    )
    assert asdict(run)["capabilities"][0]["payload"] == {}


@pytest.mark.parametrize("cleanup_state,expected", [
    (CleanupState.REMOVED, ExitCode.PARTIAL),
    (CleanupState.UNRESOLVED, ExitCode.UNRESOLVED_CLEANUP),
])
def test_invalid_earlier_result_then_interrupt_has_serializable_carrier(cleanup_state, expected):
    class Malformed:
        metadata = metadata("synthetic.malformed", impact=ImpactLevel.MUTATING, creates_mutations=True)

        def execute(self, context):
            context.evidence.record(
                capability_id="synthetic.malformed", trial_id="one", state=EvidenceState.OBSERVED,
                confidence_basis="synthetic", request_summary={}, response_summary={},
            )
            record = context.mutations.record_attempt(
                "synthetic.malformed", context.profile.target_url, "write", "item", "undo"
            )
            context.mutations.mark_created(record.mutation_id)
            context.mutations.register_cleanup(record.mutation_id, lambda: cleanup_state)
            return result_for(
                context, "synthetic.malformed", impact=ImpactLevel.MUTATING,
                payload=(item for item in ()),
            )

    class Interrupts:
        metadata = metadata("synthetic.interrupt")

        def execute(self, context):
            raise KeyboardInterrupt

    target = profile(impact_ceiling=ImpactLevel.MUTATING)
    with OriginBoundClient(target, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        with pytest.raises(EngagementInterrupted) as raised:
            EngagementRunner(target, client).run([Malformed(), Interrupts()])
    run = raised.value.run
    assert [item.status for item in run.capabilities] == [CapabilityStatus.FAILED, CapabilityStatus.PARTIAL]
    assert run.capabilities[0].evidence_ids == [run.evidence[0].evidence_id]
    assert run.capabilities[0].mutation_ids == [run.mutations[0].mutation_id]
    assert run.mutations[0].state is cleanup_state
    assert run.status == "interrupted"
    assert run.exit_code is expected
    assert asdict(run)["capabilities"][0]["payload"] == {}


def test_payload_any_values_remain_unrestricted():
    class Synthetic:
        metadata = metadata()

        def execute(self, context):
            return result_for(context, payload={"nested": {1: object()}})

    run = run_with_transport([Synthetic()])
    assert run.exit_code is ExitCode.CLEAN
    assert run.capabilities[0].payload["nested"][1] is not None
