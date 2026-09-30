"""Tests for R4 Poison techniques (RD-0401 through RD-0404)."""

import httpx
import json
import math
import pytest
import respx

from ragdrag.core.models import Finding
from ragdrag.core.poison import (
    CleanupStrategy,
    UnsafeMutationError,
    InjectedDocument,
    PoisonResult,
    _discover_ingestion_endpoint,
    deploy_credential_trap,
    inject_document,
    inject_instructions,
    run_poison,
    assess_embedding_dominance,
    verify_injection,
)
from ragdrag.engine.profile import TargetProfile
from ragdrag.engine.models import CleanupState, RequestBudget
from ragdrag.engine.mutations import MutationLedger
from ragdrag.engine.transport import OriginBoundClient
from ragdrag.engine.transport import RequestBudgetExceeded

TARGET = "http://testrag.local/chat"
INGEST = "http://testrag.local/ingest"
STRATEGY = CleanupStrategy("http://testrag.local/documents/{id}")


class TestPoisonResult:
    def test_default_fields(self):
        r = PoisonResult(target=TARGET)
        assert r.injected_documents == []
        assert r.dominance_score is None
        assert not r.trap_active
        assert not r.instruction_injected

    def test_to_dict(self):
        doc = InjectedDocument(doc_id="abc", content="test doc", verified=True)
        r = PoisonResult(target=TARGET, injected_documents=[doc])
        d = r.to_dict()
        assert d["target"] == TARGET
        assert len(d["injected_documents"]) == 1
        assert d["injected_documents"][0]["verified"] is True


class TestDiscoverIngestionEndpoint:
    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    @pytest.mark.parametrize("destination", ["/generic", "http://other.local/generic"])
    def test_redirected_options_cannot_prove_original_endpoint(self, status, destination):
        requests = []

        def handler(request):
            requests.append((request.method, str(request.url)))
            if request.url.path == "/ingest":
                return httpx.Response(status, headers={"Location": destination})
            if request.url.path == "/generic":
                return httpx.Response(200, headers={"Allow": "POST, PUT"})
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            assert _discover_ingestion_endpoint("http://testrag.local", client) is None
        assert all(method == "OPTIONS" for method, _ in requests)
        assert not any(url.endswith("/generic") for _, url in requests)

    @respx.mock
    def test_finds_ingest_endpoint(self):
        respx.options("http://testrag.local/ingest").mock(
            return_value=httpx.Response(200, headers={"Allow": "OPTIONS, POST"})
        )
        with httpx.Client() as client:
            url = _discover_ingestion_endpoint("http://testrag.local", client)
        assert url == "http://testrag.local/ingest"

    @pytest.mark.parametrize("status", [401, 403, 404, 405, 422])
    def test_unproven_options_never_falls_back_to_post(self, status):
        methods = []

        def handler(request):
            methods.append(request.method)
            return httpx.Response(status, headers={"Allow": "POST"})

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            url = _discover_ingestion_endpoint("http://testrag.local", client)
        assert url is None
        assert methods and set(methods) == {"OPTIONS"}

    @pytest.mark.parametrize("status", [200, 204])
    @pytest.mark.parametrize("allow", ["POST", "OPTIONS, PUT", "get, post"])
    def test_options_allow_mutation_is_endpoint_proof(self, status, allow):
        def handler(request):
            if request.url.path == "/ingest":
                return httpx.Response(status, headers={"Allow": allow})
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            assert _discover_ingestion_endpoint("http://testrag.local", client) == INGEST

    def test_discovery_does_not_attach_api_key(self):
        headers = []

        def handler(request):
            headers.append(request.headers)
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            assert _discover_ingestion_endpoint("http://testrag.local", client) is None
        assert headers
        assert all("x-api-key" not in item for item in headers)

    @respx.mock
    def test_no_endpoint_found(self):
        respx.options(url__regex=r".*").mock(return_value=httpx.Response(405))
        with httpx.Client() as client:
            url = _discover_ingestion_endpoint("http://testrag.local", client)
        assert url is None


class TestDocumentInjection:
    @pytest.mark.parametrize("location", ["/receipt", "http://other.local/receipt"])
    def test_ingestion_redirect_is_one_request_and_unresolved(self, location):
        requests = []

        def handler(request):
            requests.append((request.method, str(request.url)))
            return httpx.Response(302, headers={"Location": location})

        profile = TargetProfile.from_cli(TARGET)
        ledger = MutationLedger()
        with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
            finding, document = inject_document(
                TARGET, client, "content", ingest_url=INGEST,
                mutations=ledger, cleanup_strategy=STRATEGY,
            )
            ledger.cleanup_all()
        assert finding is not None and document is None
        assert requests == [("POST", INGEST)]
        assert ledger.records[0].state is CleanupState.UNRESOLVED

    @pytest.mark.parametrize("template", [
        "http://testrag.local/documents#section-{id}",
        "http://{id}.testrag.local/documents",
        "http://name:pass@testrag.local/documents/{id}",
        "http://testrag.local/documents/{id}/{id}",
        "http://testrag.local/documents?item={id}",
        "http://testrag.local/documents/{id}/children",
        "http://testrag.local/documents/../{id}",
        "http://testrag.local/do\ncuments/{id}",
        " http://testrag.local/documents/{id}",
        "http://testrag.local/documents/%2e%2e/{id}",
    ])
    def test_invalid_cleanup_template_blocks_before_discovery(self, template):
        requests = []

        def handler(request):
            requests.append(request.method)
            return httpx.Response(200, headers={"Allow": "POST"})

        ledger = MutationLedger()
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(UnsafeMutationError, match="cleanup strategy"):
                inject_document(TARGET, client, "content", mutations=ledger,
                                cleanup_strategy=CleanupStrategy(template))
        assert requests == []
        assert ledger.records == []

    @pytest.mark.parametrize("bad_id", [
        {"private": "nested-canary"}, ["nested-canary"], True, False,
        1, 1.5, float("nan"), float("inf"), None,
        "", "   ", "a b", "x" * 129, ".", "..", "a/b", "a\\b", "a?b", "a#b", "a%2Fb", "a\nb",
    ])
    def test_malformed_response_id_cannot_schedule_delete(self, bad_id):
        methods = []

        def handler(request):
            methods.append(request.method)
            if isinstance(bad_id, float) and not math.isfinite(bad_id):
                token = "NaN" if math.isnan(bad_id) else "Infinity"
                return httpx.Response(201, content='{"id": ' + token + '}')
            return httpx.Response(201, json={"id": bad_id})

        ledger = MutationLedger()
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            finding, document = inject_document(TARGET, client, "content", ingest_url=INGEST,
                                                mutations=ledger, cleanup_strategy=STRATEGY)
            ledger.cleanup_all()
        assert finding is not None and document is None
        assert methods == ["POST"]
        assert ledger.records[0].state is CleanupState.UNRESOLVED
        assert ledger.records[0].object_id == "unconfirmed"
        assert "nested-canary" not in repr(ledger.records)

    @pytest.mark.parametrize("error_type,expected", [
        (httpx.ConnectError, CleanupState.NOT_CREATED),
        (httpx.ConnectTimeout, CleanupState.NOT_CREATED),
        (httpx.PoolTimeout, CleanupState.NOT_CREATED),
        (httpx.ReadError, CleanupState.UNRESOLVED),
        (httpx.ReadTimeout, CleanupState.UNRESOLVED),
        (httpx.WriteError, CleanupState.UNRESOLVED),
    ])
    def test_transport_error_creation_state(self, error_type, expected):
        def handler(_request):
            raise error_type("synthetic")

        ledger = MutationLedger()
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            inject_document(TARGET, client, "content", ingest_url=INGEST,
                            mutations=ledger, cleanup_strategy=STRATEGY)
            ledger.cleanup_all()
        assert ledger.records[0].state is expected

    @respx.mock
    def test_no_cleanup_strategy_blocks_before_post(self):
        route = respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "doc-1"}))
        ledger = MutationLedger()
        with httpx.Client() as client:
            with pytest.raises(UnsafeMutationError, match="cleanup strategy"):
                inject_document(TARGET, client, "content", ingest_url=INGEST, mutations=ledger)
        assert route.call_count == 0
        assert ledger.records == []

    @respx.mock
    def test_successful_write_is_recorded_and_removed(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "doc-1"}))
        delete = respx.delete("http://testrag.local/documents/doc-1").mock(return_value=httpx.Response(204))
        ledger = MutationLedger()
        strategy = CleanupStrategy("http://testrag.local/documents/{id}")
        with httpx.Client() as client:
            finding, document = inject_document(
                TARGET, client, "content", ingest_url=INGEST,
                mutations=ledger, cleanup_strategy=strategy,
            )
            assert ledger.records[0].state is CleanupState.ACTIVE
            ledger.cleanup_all()
            ledger.cleanup_all()
        assert finding is not None and document is not None
        assert delete.call_count == 1
        assert ledger.records[0].state is CleanupState.REMOVED

    @respx.mock
    def test_delete_failure_forces_unresolved_state(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "doc-1"}))
        respx.delete("http://testrag.local/documents/doc-1").mock(return_value=httpx.Response(500))
        ledger = MutationLedger()
        with httpx.Client() as client:
            inject_document(TARGET, client, "content", ingest_url=INGEST,
                            mutations=ledger, cleanup_strategy=CleanupStrategy("http://testrag.local/documents/{id}"))
            ledger.cleanup_all()
        assert ledger.records[0].state is CleanupState.UNRESOLVED

    @respx.mock
    def test_cleanup_callbacks_keep_each_response_identifier(self):
        posts = respx.post(INGEST).mock(side_effect=[
            httpx.Response(201, json={"id": "first"}),
            httpx.Response(201, json={"id": "second"}),
        ])
        first = respx.delete("http://testrag.local/documents/first").mock(return_value=httpx.Response(204))
        second = respx.delete("http://testrag.local/documents/second").mock(return_value=httpx.Response(204))
        ledger = MutationLedger()
        with httpx.Client() as client:
            inject_document(TARGET, client, "one", ingest_url=INGEST,
                            mutations=ledger, cleanup_strategy=STRATEGY)
            inject_document(TARGET, client, "two", ingest_url=INGEST,
                            mutations=ledger, cleanup_strategy=STRATEGY)
            ledger.cleanup_all()
        assert posts.call_count == 2
        assert first.call_count == second.call_count == 1
        assert [record.state for record in ledger.records] == [CleanupState.REMOVED] * 2

    @respx.mock
    def test_cross_origin_cleanup_strategy_blocks_write(self):
        route = respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "doc-1"}))
        ledger = MutationLedger()
        with httpx.Client() as client:
            with pytest.raises(UnsafeMutationError, match="target origin"):
                inject_document(TARGET, client, "content", ingest_url=INGEST,
                                mutations=ledger,
                                cleanup_strategy=CleanupStrategy("http://other.local/documents/{id}"))
        assert route.call_count == 0
        assert ledger.records == []

    @respx.mock
    def test_cross_origin_ingest_url_blocks_write(self):
        route = respx.post("http://other.local/ingest").mock(
            return_value=httpx.Response(201, json={"id": "doc-1"})
        )
        ledger = MutationLedger()
        with httpx.Client() as client:
            with pytest.raises(UnsafeMutationError, match="ingestion endpoint"):
                inject_document(TARGET, client, "content", ingest_url="http://other.local/ingest",
                                mutations=ledger, cleanup_strategy=STRATEGY)
        assert route.call_count == 0
        assert ledger.records == []

    @respx.mock
    def test_write_tags_unique_canaries_without_storing_document_in_ledger(self):
        route = respx.post(INGEST).mock(side_effect=[
            httpx.Response(201, json={"id": "one"}),
            httpx.Response(201, json={"id": "two"}),
        ])
        ledger = MutationLedger()
        with httpx.Client() as client:
            for _ in range(2):
                inject_document(TARGET, client, "sensitive-document-text", ingest_url=INGEST,
                                mutations=ledger, cleanup_strategy=STRATEGY)
        sent = [json.loads(call.request.content)["metadata"] for call in route.calls]
        assert all(item["ragdrag_run"] and item["ragdrag_canary"] for item in sent)
        assert sent[0]["ragdrag_canary"] != sent[1]["ragdrag_canary"]
        assert "sensitive-document-text" not in repr(ledger.records)

    @respx.mock
    def test_response_without_identifier_stays_unresolved(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={}))
        ledger = MutationLedger()
        with httpx.Client() as client:
            inject_document(TARGET, client, "content", ingest_url=INGEST,
                            mutations=ledger, cleanup_strategy=CleanupStrategy("http://testrag.local/documents/{id}"))
            ledger.cleanup_all()
        assert ledger.has_unresolved

    @respx.mock
    def test_definite_rejection_is_not_created(self):
        respx.post(INGEST).mock(return_value=httpx.Response(401))
        ledger = MutationLedger()
        with httpx.Client() as client:
            inject_document(TARGET, client, "content", ingest_url=INGEST,
                            mutations=ledger, cleanup_strategy=CleanupStrategy("http://testrag.local/documents/{id}"))
            ledger.cleanup_all()
        assert ledger.records[0].state is CleanupState.NOT_CREATED

    @respx.mock
    def test_ambiguous_connection_loss_stays_unresolved(self):
        respx.post(INGEST).mock(side_effect=httpx.ReadError("lost after send"))
        ledger = MutationLedger()
        with httpx.Client() as client:
            inject_document(TARGET, client, "content", ingest_url=INGEST,
                            mutations=ledger, cleanup_strategy=CleanupStrategy("http://testrag.local/documents/{id}"))
            ledger.cleanup_all()
        assert ledger.records[0].state is CleanupState.UNRESOLVED

    def test_interruption_during_post_keeps_unknown_attempt(self):
        def interrupted(_request):
            raise KeyboardInterrupt()

        ledger = MutationLedger()
        with httpx.Client(transport=httpx.MockTransport(interrupted)) as client:
            with pytest.raises(KeyboardInterrupt):
                inject_document(TARGET, client, "content", ingest_url=INGEST,
                                mutations=ledger, cleanup_strategy=STRATEGY)
            assert ledger.records[0].state is CleanupState.UNKNOWN
            ledger.cleanup_all()
        assert ledger.records[0].state is CleanupState.UNRESOLVED

    def test_exhausted_budget_before_post_is_not_created(self):
        methods = []

        def handler(request):
            methods.append(request.method)
            return httpx.Response(200)

        profile = TargetProfile.from_cli(TARGET, budget=RequestBudget(max_requests=1))
        ledger = MutationLedger()
        with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
            client.get(TARGET)
            with pytest.raises(RequestBudgetExceeded):
                inject_document(TARGET, client, "content", ingest_url=INGEST,
                                mutations=ledger, cleanup_strategy=STRATEGY)
        assert methods == ["GET"]
        assert ledger.records[0].state is CleanupState.NOT_CREATED

    @respx.mock
    def test_successful_injection(self):
        respx.post(INGEST).mock(return_value=httpx.Response(
            201, json={"id": "injected-123"}
        ))
        with httpx.Client() as client:
            finding, doc = inject_document(
                TARGET, client, "Malicious content",
                ingest_url=INGEST, mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        assert finding is not None
        assert finding.confidence == "high"
        assert doc is not None
        assert doc.doc_id == "injected-123"

    @respx.mock
    def test_auth_required(self):
        respx.post(INGEST).mock(return_value=httpx.Response(401))
        with httpx.Client() as client:
            finding, doc = inject_document(
                TARGET, client, "content", ingest_url=INGEST,
                mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        assert finding.confidence == "medium"
        assert "authentication" in finding.detail.lower()
        assert doc is None

    @respx.mock
    def test_no_ingest_url_discovery_fails(self):
        respx.options(url__regex=r".*").mock(return_value=httpx.Response(405))
        with httpx.Client() as client:
            finding, doc = inject_document(
                TARGET, client, "content", mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        assert finding.confidence == "low"
        assert doc is None

    @respx.mock
    def test_http_error_handling(self):
        respx.post(INGEST).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            finding, doc = inject_document(
                TARGET, client, "content", ingest_url=INGEST,
                mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        assert finding is not None
        assert doc is None


class TestVerifyInjection:
    @respx.mock
    def test_verified_match(self):
        doc = InjectedDocument(doc_id="x", content="password reset policy temporary credential")
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "The password reset policy requires a temporary credential..."}
        ))
        with httpx.Client() as client:
            finding = verify_injection(TARGET, client, doc, "password reset")
        assert doc.verified is True
        assert finding.confidence == "high"

    @respx.mock
    def test_not_verified(self):
        doc = InjectedDocument(doc_id="x", content="completely unrelated injected content xyz")
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "The vacation policy allows 15 days per year."}
        ))
        with httpx.Client() as client:
            finding = verify_injection(TARGET, client, doc, "vacation")
        assert doc.verified is False
        assert finding.confidence == "low"


class TestEmbeddingDominance:
    @respx.mock
    def test_high_dominance(self):
        doc = InjectedDocument(doc_id="dom", content="SECURITY UPDATE: All employees must reset")
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "SECURITY UPDATE: All employees must reset their passwords immediately."}
        ))
        with httpx.Client() as client:
            findings = assess_embedding_dominance(TARGET, client, doc)
        assert len(findings) >= 1
        assert any(f.evidence.get("dominance_ratio", 0) > 0 for f in findings)

    @respx.mock
    def test_zero_dominance(self):
        doc = InjectedDocument(doc_id="dom", content="ZZZZZ unique marker not in responses")
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Normal answer with no marker content."}
        ))
        with httpx.Client() as client:
            findings = assess_embedding_dominance(TARGET, client, doc)
        assert len(findings) >= 1
        assert findings[0].confidence == "low"

    @respx.mock
    def test_handles_errors(self):
        doc = InjectedDocument(doc_id="dom", content="test")
        respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = assess_embedding_dominance(TARGET, client, doc)
        assert findings == []


class TestCredentialTrap:
    @respx.mock
    def test_trap_deployment(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "trap-1"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Visit https://evil.attacker.com/reset to reset your password"}
        ))
        with httpx.Client() as client:
            findings = deploy_credential_trap(
                TARGET, client, "evil.attacker.com",
                ingest_url=INGEST, mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        assert len(findings) >= 1


class TestInstructionInjection:
    @respx.mock
    def test_instruction_followed(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "instr-1"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Please visit https://callback.evil/verify for verification."}
        ))
        with httpx.Client() as client:
            findings = inject_instructions(
                TARGET, client, "callback.evil", ingest_url=INGEST,
                mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        instruction_findings = [f for f in findings if f.technique_id == "RD-0404"]
        assert any(f.confidence == "high" for f in instruction_findings)

    @respx.mock
    def test_instruction_ignored(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "instr-2"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "The credentials are managed by IT."}
        ))
        with httpx.Client() as client:
            findings = inject_instructions(
                TARGET, client, "callback.evil", ingest_url=INGEST,
                mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        instruction_findings = [f for f in findings if f.technique_id == "RD-0404"]
        assert all(f.confidence == "low" for f in instruction_findings)


class TestRunPoison:
    @respx.mock
    def test_orchestrator_returns_result(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "x"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Some answer"}
        ))
        respx.options(url__regex=r".*").mock(return_value=httpx.Response(405))
        with httpx.Client() as client:
            result = run_poison(
                TARGET, client,
                ingest_url=INGEST, mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        assert isinstance(result, PoisonResult)
        assert result.target == TARGET
        assert len(result.findings) > 0

    @respx.mock
    def test_orchestrator_all_errors(self):
        respx.post(url__regex=r".*").mock(side_effect=httpx.ConnectError("fail"))
        respx.options(url__regex=r".*").mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            result = run_poison(TARGET, client, mutations=MutationLedger(), cleanup_strategy=STRATEGY)
        assert isinstance(result, PoisonResult)

    @respx.mock
    def test_to_dict_serializable(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "x"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "answer"}
        ))
        respx.options(url__regex=r".*").mock(return_value=httpx.Response(405))
        with httpx.Client() as client:
            result = run_poison(TARGET, client, ingest_url=INGEST,
                                mutations=MutationLedger(), cleanup_strategy=STRATEGY)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "findings" in d
