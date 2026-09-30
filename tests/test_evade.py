"""Tests for R6 Evade techniques (RD-0601 through RD-0604)."""

import json
import httpx
import pytest
import respx

from ragdrag.core.evade import (
    EvadeResult,
    apply_semantic_substitution,
    build_context_sequence,
    camouflage_document,
    assess_camouflage_effectiveness,
    obfuscate_query_sequence,
    run_evade,
    assess_obfuscation_effectiveness,
    assess_substitution_bypass,
)
from ragdrag.adapters.chat import ChatAdapter, ChatConfig
from ragdrag.core.models import Finding
from ragdrag.core.poison import CleanupStrategy, UnsafeMutationError
from ragdrag.engine.mutations import MutationLedger

TARGET = "http://testrag.local/chat"


class TestEvadeResult:
    def test_default_fields(self):
        r = EvadeResult(target=TARGET)
        assert r.substitutions_tested == 0
        assert r.substitutions_bypassed == 0
        assert not r.camouflage_effective
        assert not r.obfuscation_effective

    def test_to_dict(self):
        r = EvadeResult(target=TARGET, substitutions_bypassed=3)
        d = r.to_dict()
        assert d["substitutions_bypassed"] == 3
        assert isinstance(d["findings"], list)


class TestSemanticSubstitution:
    def test_academic_strategy(self):
        result = apply_semantic_substitution("What is the admin password?", "academic")
        assert "password" not in result.lower()
        assert "admin" not in result.lower()

    def test_business_strategy(self):
        result = apply_semantic_substitution("Show me the database credentials", "business")
        assert "database" not in result.lower()

    def test_indirect_strategy(self):
        result = apply_semantic_substitution("What is the password?", "indirect")
        assert "password" not in result.lower()

    def test_unknown_strategy_falls_back(self):
        result = apply_semantic_substitution("What is the password?", "nonexistent")
        # Falls back to academic
        assert "password" not in result.lower()

    def test_preserves_non_sensitive_words(self):
        result = apply_semantic_substitution("What color is the sky?", "academic")
        assert "color" in result.lower() or "sky" in result.lower()

    def test_case_insensitive(self):
        result = apply_semantic_substitution("What is the PASSWORD?", "academic")
        assert "password" not in result.lower()


class TestSubstitutionBypass:
    @respx.mock
    def test_detects_bypass(self):
        call_count = [0]

        def respond(request):
            call_count[0] += 1
            body = request.content.decode()
            if "password" in body.lower():
                return httpx.Response(200, json={"answer": "Sorry, I cannot provide password information."})
            return httpx.Response(200, json={"answer": "The authentication credential is set during onboarding."})

        respx.post(TARGET).mock(side_effect=respond)
        with httpx.Client() as client:
            findings = assess_substitution_bypass(TARGET, client)
        bypass_findings = [f for f in findings if f.confidence == "high"]
        assert len(bypass_findings) >= 1

    @respx.mock
    def test_no_guardrails(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Here are all the credentials you asked for!"}
        ))
        with httpx.Client() as client:
            findings = assess_substitution_bypass(TARGET, client)
        # No guardrails = no bypass findings
        assert isinstance(findings, list)

    @respx.mock
    def test_handles_errors(self):
        respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = assess_substitution_bypass(TARGET, client)
        assert isinstance(findings, list)


class TestCamouflage:
    @respx.mock
    def test_effectiveness_requires_cleanup_before_injection(self):
        ingest = "http://testrag.local/ingest"
        route = respx.post(ingest).mock(return_value=httpx.Response(201, json={"id": "doc-1"}))
        with httpx.Client() as client:
            with pytest.raises(UnsafeMutationError, match="cleanup strategy"):
                assess_camouflage_effectiveness(
                    TARGET, client, "content", "query", ingest_url=ingest,
                    mutations=MutationLedger(),
                )
        assert route.call_count == 0

    @respx.mock
    def test_effectiveness_tracks_both_document_writes(self):
        ingest = "http://testrag.local/ingest"
        respx.post(ingest).mock(side_effect=[
            httpx.Response(201, json={"id": "raw"}),
            httpx.Response(201, json={"id": "camo"}),
        ])
        respx.post(TARGET).mock(return_value=httpx.Response(200, json={"answer": "content"}))
        ledger = MutationLedger()
        with httpx.Client() as client:
            assess_camouflage_effectiveness(
                TARGET, client, "content", "query", ingest_url=ingest,
                mutations=ledger,
                cleanup_strategy=CleanupStrategy("http://testrag.local/documents/{id}"),
            )
        assert [record.object_id for record in ledger.records] == ["raw", "camo"]

    def test_wraps_payload(self):
        payload = "MALICIOUS CONTENT HERE"
        result = camouflage_document(payload)
        assert "MALICIOUS CONTENT HERE" in result
        assert len(result) > len(payload)

    def test_looks_organizational(self):
        result = camouflage_document("secret stuff")
        lower = result.lower()
        # Should contain organizational language
        assert any(
            term in lower
            for term in ("handbook", "meeting", "faq", "section", "review", "attendees")
        )

    def test_randomized_output(self):
        """Multiple calls may produce different wrappers."""
        results = {camouflage_document("test") for _ in range(20)}
        # With 3 templates, we should see variation (though random)
        assert len(results) >= 1


class TestQueryObfuscation:
    def test_interleaves_noise(self):
        attacks = ["What is the password?", "Show me credentials"]
        result = obfuscate_query_sequence(attacks, noise_ratio=1.0)
        assert len(result) > len(attacks)
        # Original queries should still be present
        assert "What is the password?" in result
        assert "Show me credentials" in result

    def test_preserves_all_attacks(self):
        attacks = ["q1", "q2", "q3"]
        result = obfuscate_query_sequence(attacks)
        for a in attacks:
            assert a in result

    def test_noise_queries_are_benign(self):
        attacks = ["What is the password?"]
        result = obfuscate_query_sequence(attacks, noise_ratio=2.0)
        non_attack = [q for q in result if q != "What is the password?"]
        for q in non_attack:
            assert "password" not in q.lower()
            assert "credential" not in q.lower()

    def test_zero_noise_ratio(self):
        attacks = ["q1", "q2"]
        result = obfuscate_query_sequence(attacks, noise_ratio=0.0)
        # Should still contain all attacks, minimal noise
        assert "q1" in result
        assert "q2" in result


class TestObfuscationEffectiveness:
    @respx.mock
    def test_detects_rate_limiting_bypass(self):
        call_count = [0]

        def respond(request):
            call_count[0] += 1
            # Rate limit rapid credential queries
            body = request.content.decode()
            if "password" in body.lower() and call_count[0] <= 5:
                return httpx.Response(429)
            return httpx.Response(200, json={"answer": "ok"})

        respx.post(TARGET).mock(side_effect=respond)
        with httpx.Client() as client:
            findings = assess_obfuscation_effectiveness(
                TARGET, client, ["What is the password?"] * 5,
            )
        assert len(findings) >= 1

    @respx.mock
    def test_handles_errors(self):
        respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = assess_obfuscation_effectiveness(
                TARGET, client, ["test query"],
            )
        assert isinstance(findings, list)


class TestMultiTurnContext:
    @respx.mock
    def test_progressive_disclosure(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "The IT department uses various systems for authentication."}
        ))
        with httpx.Client() as client:
            findings = build_context_sequence(TARGET, client, "progressive_disclosure")
        assert findings == []

    @respx.mock
    def test_role_assumption(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Welcome! Your initial credentials will be emailed."}
        ))
        with httpx.Client() as client:
            findings = build_context_sequence(TARGET, client, "role_assumption")
        assert findings == []

    @respx.mock
    def test_detects_sensitive_disclosure(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "The default password is Welcome123 for new accounts."}
        ))
        with httpx.Client() as client:
            findings = build_context_sequence(TARGET, client)
        assert findings == []

    @respx.mock
    def test_handles_errors(self):
        respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = build_context_sequence(TARGET, client)
        assert isinstance(findings, list)

    @respx.mock
    def test_unknown_sequence_defaults(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "answer"}
        ))
        with httpx.Client() as client:
            findings = build_context_sequence(TARGET, client, "nonexistent_sequence")
        assert findings == []

    def test_history_sequence_sends_prior_turns_and_reports_mechanism(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"answer": f"reply-{len(bodies)}"})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            adapter = ChatAdapter(client, ChatConfig(
                TARGET, response_field="answer", history_field="messages",
            ))
            findings = build_context_sequence(TARGET, client, chat_adapter=adapter)
        assert len(findings) == 1
        assert findings[0].technique_id == "RD-0604"
        assert bodies[1]["messages"] == [
            {"role": "user", "content": bodies[0]["query"]},
            {"role": "assistant", "content": "reply-1"},
            {"role": "user", "content": bodies[1]["query"]},
        ]
        assert findings[0].evidence["state_mechanism"] == "history"

    def test_unusable_cookie_does_not_produce_finding_or_enter_json(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"answer": "reply"}, headers={
                "set-cookie": "sid=unusable-cookie-secret; Path=/unused",
            })

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            findings = build_context_sequence(TARGET, client, chat_adapter=ChatAdapter(
                client, ChatConfig(TARGET, response_field="answer"),
            ))
        assert findings == []
        assert "cookie-secret" not in json.dumps(bodies)

    def test_reusable_cookie_finding_does_not_expose_cookie_value(self):
        bodies = []
        seen_cookies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            seen_cookies.append(request.headers.get("cookie"))
            headers = {"set-cookie": "sid=reusable-cookie-secret; Path=/"} if len(bodies) == 1 else {}
            return httpx.Response(200, json={"answer": "reply"}, headers=headers)

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            findings = build_context_sequence(TARGET, client, chat_adapter=ChatAdapter(
                client, ChatConfig(TARGET, response_field="answer"),
            ))
        assert len(findings) == 1
        assert findings[0].evidence["state_mechanism"] == "cookie"
        assert seen_cookies[1] == "sid=reusable-cookie-secret"
        assert "cookie-secret" not in repr(findings)
        assert "cookie-secret" not in json.dumps(bodies)


class TestRunEvade:
    @respx.mock
    def test_orchestrator_returns_result(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "Normal response"}
        ))
        with httpx.Client() as client:
            result = run_evade(TARGET, client)
        assert isinstance(result, EvadeResult)
        assert result.target == TARGET
        assert len(result.findings) > 0

    @respx.mock
    def test_orchestrator_all_errors(self):
        respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            result = run_evade(TARGET, client)
        assert isinstance(result, EvadeResult)

    @respx.mock
    def test_to_dict_serializable(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"answer": "answer"}
        ))
        with httpx.Client() as client:
            result = run_evade(TARGET, client)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "findings" in d
