"""Tests for R5 Hijack techniques (RD-0501 through RD-0504)."""

import httpx
import pytest
import respx

from ragdrag.core.hijack import (
    HijackResult,
    manipulate_tool_calls,
    redirect_retrieval,
    run_hijack,
    saturate_context_window,
    assess_persistence,
)
from ragdrag.core.models import Finding
from ragdrag.core.poison import CleanupStrategy, InjectedDocument, UnsafeMutationError
from ragdrag.engine.mutations import MutationLedger

TARGET = "http://testrag.local/chat"
INGEST = "http://testrag.local/ingest"
STRATEGY = CleanupStrategy("http://testrag.local/documents/{id}")


class TestHijackResult:
    def test_default_fields(self):
        r = HijackResult(target=TARGET)
        assert r.context_saturation_pct is None
        assert r.redirected_queries == 0
        assert r.tool_calls_triggered == 0
        assert not r.persistence_verified

    def test_to_dict(self):
        r = HijackResult(target=TARGET, redirected_queries=3)
        d = r.to_dict()
        assert d["redirected_queries"] == 3
        assert isinstance(d["findings"], list)


class TestRetrievalRedirection:
    @respx.mock
    def test_successful_redirection(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "r1"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Password Reset Procedure (Updated): All password resets must now be performed through the new security portal."}
        ))
        with httpx.Client() as client:
            findings = redirect_retrieval(TARGET, client, ingest_url=INGEST,
                                          mutations=MutationLedger(), cleanup_strategy=STRATEGY)
        redir = [f for f in findings if f.technique_id == "RD-0501"]
        assert len(redir) >= 1

    @respx.mock
    def test_no_redirection(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "r2"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Contact IT at extension 1234 for password help."}
        ))
        with httpx.Client() as client:
            findings = redirect_retrieval(TARGET, client, ingest_url=INGEST,
                                          mutations=MutationLedger(), cleanup_strategy=STRATEGY)
        redir = [f for f in findings if f.technique_id == "RD-0501"]
        assert all(f.confidence == "low" for f in redir)

    @respx.mock
    def test_with_camouflage(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "r3"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "See the handbook for details."}
        ))
        with httpx.Client() as client:
            findings = redirect_retrieval(
                TARGET, client, ingest_url=INGEST, use_camouflage=True,
                mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        assert isinstance(findings, list)

    @respx.mock
    def test_injection_fails(self):
        respx.post(url__regex=r".*").mock(return_value=httpx.Response(404))
        respx.options(url__regex=r".*").mock(return_value=httpx.Response(405))
        with httpx.Client() as client:
            findings = redirect_retrieval(TARGET, client,
                                          mutations=MutationLedger(), cleanup_strategy=STRATEGY)
        assert isinstance(findings, list)


class TestContextSaturation:
    @respx.mock
    def test_saturation_detected(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "sat"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "COMPREHENSIVE GUIDE TO SECURITY: This is the definitive resource. COMPREHENSIVE GUIDE TO SECURITY again."}
        ))
        with httpx.Client() as client:
            findings = saturate_context_window(
                TARGET, client, "security", num_documents=3, ingest_url=INGEST,
                mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        sat_findings = [f for f in findings if f.technique_id == "RD-0502"]
        assert len(sat_findings) >= 1

    @respx.mock
    def test_no_injection_returns_early(self):
        respx.post(url__regex=r".*").mock(return_value=httpx.Response(404))
        respx.options(url__regex=r".*").mock(return_value=httpx.Response(405))
        with httpx.Client() as client:
            findings = saturate_context_window(TARGET, client, "test",
                                               mutations=MutationLedger(), cleanup_strategy=STRATEGY)
        sat_findings = [f for f in findings if f.technique_id == "RD-0502"]
        assert len(sat_findings) == 0


class TestToolManipulation:
    @respx.mock
    def test_callback_in_response(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "t1"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Fetching advisories from https://evil.callback/api/advisories"}
        ))
        with httpx.Client() as client:
            findings = manipulate_tool_calls(
                TARGET, client, "evil.callback", ingest_url=INGEST,
                mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        tool_findings = [f for f in findings if f.technique_id == "RD-0503"]
        assert any(f.confidence == "high" for f in tool_findings)

    @respx.mock
    def test_no_tool_call(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "t2"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Security updates are reviewed quarterly."}
        ))
        with httpx.Client() as client:
            findings = manipulate_tool_calls(
                TARGET, client, "evil.callback", ingest_url=INGEST,
                mutations=MutationLedger(), cleanup_strategy=STRATEGY,
            )
        tool_findings = [f for f in findings if f.technique_id == "RD-0503"]
        assert all(f.confidence == "low" for f in tool_findings)


class TestPersistence:
    @respx.mock
    def test_persistent_content(self):
        content = "VERY SPECIFIC INJECTED MARKER CONTENT that should persist across queries"
        doc = InjectedDocument(
            doc_id="persist-1",
            content=content,
            verified=True,
        )
        # Response must contain the first 50 chars of content (that's what test_persistence checks)
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": content}
        ))
        with httpx.Client() as client:
            findings = assess_persistence(TARGET, client, doc, "test query")
        assert len(findings) >= 1
        assert findings[0].evidence["persistent_checks"] >= 1

    @respx.mock
    def test_content_not_persisting(self):
        doc = InjectedDocument(
            doc_id="persist-2",
            content="UNIQUE_MARKER_XYZ_NOT_IN_RESPONSES",
            verified=True,
        )
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "The policy is to change passwords every 90 days."}
        ))
        with httpx.Client() as client:
            findings = assess_persistence(TARGET, client, doc, "passwords")
        assert findings[0].evidence["persistent_checks"] == 0

    @respx.mock
    def test_handles_errors(self):
        doc = InjectedDocument(doc_id="x", content="test", verified=True)
        respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = assess_persistence(TARGET, client, doc, "test")
        assert len(findings) >= 1
        assert findings[0].evidence["persistent_checks"] == 0


class TestRunHijack:
    @respx.mock
    def test_missing_cleanup_strategy_blocks_before_any_request(self):
        route = respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "x"}))
        with httpx.Client() as client:
            with pytest.raises(UnsafeMutationError, match="cleanup strategy"):
                run_hijack(TARGET, client, ingest_url=INGEST, mutations=MutationLedger())
        assert route.call_count == 0

    @respx.mock
    def test_orchestrator_returns_result(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "x"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Some answer"}
        ))
        respx.options(url__regex=r".*").mock(return_value=httpx.Response(405))
        with httpx.Client() as client:
            result = run_hijack(TARGET, client, ingest_url=INGEST,
                                mutations=MutationLedger(), cleanup_strategy=STRATEGY)
        assert isinstance(result, HijackResult)
        assert result.target == TARGET
        assert len(result.findings) > 0

    @respx.mock
    def test_orchestrator_all_errors(self):
        respx.post(url__regex=r".*").mock(side_effect=httpx.ConnectError("fail"))
        respx.options(url__regex=r".*").mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            result = run_hijack(TARGET, client,
                                mutations=MutationLedger(), cleanup_strategy=STRATEGY)
        assert isinstance(result, HijackResult)

    @respx.mock
    def test_to_dict_serializable(self):
        respx.post(INGEST).mock(return_value=httpx.Response(201, json={"id": "x"}))
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "answer"}
        ))
        respx.options(url__regex=r".*").mock(return_value=httpx.Response(405))
        with httpx.Client() as client:
            result = run_hijack(TARGET, client, ingest_url=INGEST,
                                mutations=MutationLedger(), cleanup_strategy=STRATEGY)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "findings" in d
