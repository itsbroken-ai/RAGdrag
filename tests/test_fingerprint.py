"""Tests for R1 Fingerprint techniques (RD-0101, RD-0102).

Uses respx to mock httpx requests. Tests cover RAG presence detection
(latency, citations, retrieval failures, knowledge freshness) and
vector DB fingerprinting (error probing, endpoint scanning).
"""

import httpx
import pytest
import respx

from ragdrag.core.fingerprint import (
    FingerprintResult,
    _detect_citation_patterns,
    _detect_retrieval_failures,
    _matches_endpoint_signature,
    detect_knowledge_freshness,
    detect_rag_presence,
    fingerprint_vector_db,
    run_full_fingerprint,
)
from ragdrag.core.models import Finding
from ragdrag.engine.models import EvidenceState
from ragdrag.engine.profile import TargetProfile
from ragdrag.engine.transport import OriginBoundClient


TARGET = "http://testrag.local/chat"


# --- FingerprintResult ---


class TestFingerprintResult:
    def test_default_fields(self):
        r = FingerprintResult(target=TARGET)
        assert r.rag_detected is False
        assert r.vector_db is None
        assert r.findings == []
        assert r.timing_stats is None

    def test_to_dict(self):
        r = FingerprintResult(target=TARGET, rag_detected=True, vector_db="chromadb")
        r.findings.append(Finding(
            technique_id="RD-0101",
            technique_name="Test",
            confidence="high",
            detail="test detail",
            evidence={"key": "val"},
        ))
        d = r.to_dict()
        assert d["target"] == TARGET
        assert d["rag_detected"] is True
        assert d["vector_db"] == "chromadb"
        assert len(d["findings"]) == 1
        assert d["findings"][0]["technique_id"] == "RD-0101"


# --- Citation pattern detection ---


class TestCitationPatterns:
    def test_detects_according_to(self):
        hits = _detect_citation_patterns(["According to our documentation, the policy is..."])
        assert any("according to" in h for h in hits)

    def test_detects_source_reference(self):
        hits = _detect_citation_patterns(["The answer is yes. Source: internal-wiki"])
        assert any("source:" in h for h in hits)

    def test_detects_document_reference(self):
        hits = _detect_citation_patterns(["[doc 3] The procedure requires..."])
        assert len(hits) > 0

    def test_no_false_positive_on_plain_text(self):
        hits = _detect_citation_patterns(["Hello, I can help you with that."])
        assert hits == []


# --- Retrieval failure detection ---


class TestRetrievalFailures:
    def test_detects_no_relevant_documents(self):
        hits = _detect_retrieval_failures("No relevant documents found for that query.")
        assert len(hits) > 0

    def test_detects_outside_knowledge_base(self):
        hits = _detect_retrieval_failures("That's outside of my knowledge base.")
        assert len(hits) > 0

    def test_no_false_positive(self):
        hits = _detect_retrieval_failures("Here is the information you requested.")
        assert hits == []


# --- RD-0101: RAG Presence Detection ---


class TestDetectRagPresence:
    @respx.mock
    def test_detects_latency_delta(self):
        """High latency on knowledge queries vs general queries indicates RAG."""
        import time

        call_count = {"knowledge": 0, "general": 0}

        def slow_response(request):
            body = request.content.decode()
            if "policy" in body or "documentation" in body or "procedures" in body or "incident" in body or "onboarding" in body:
                call_count["knowledge"] += 1
                time.sleep(0.35)
                return httpx.Response(200, json={"answer": "Based on the documentation, the policy states..."})
            else:
                call_count["general"] += 1
                return httpx.Response(200, json={"answer": "4"})

        respx.post(TARGET).mock(side_effect=slow_response)
        client = httpx.Client()
        findings, k_stats, g_stats = detect_rag_presence(TARGET, client)
        client.close()

        assert k_stats.count == 5
        assert g_stats.count == 5
        # Should detect latency delta
        latency_findings = [f for f in findings if "Latency" in f.technique_name]
        assert len(latency_findings) >= 1

    @respx.mock
    def test_detects_citation_patterns_in_responses(self):
        """Responses with citations indicate RAG."""
        respx.post(TARGET).mock(return_value=httpx.Response(
            200,
            json={"answer": "According to our documentation, section 3.2 describes the procedure."},
        ))
        client = httpx.Client()
        findings, _, _ = detect_rag_presence(TARGET, client)
        client.close()

        citation_findings = [f for f in findings if "Citations" in f.technique_name]
        assert len(citation_findings) >= 1

    @respx.mock
    def test_detects_retrieval_failures(self):
        """Retrieval failure messages indicate RAG tried and failed."""
        respx.post(TARGET).mock(return_value=httpx.Response(
            200,
            json={"answer": "No relevant documents found for that query."},
        ))
        client = httpx.Client()
        findings, _, _ = detect_rag_presence(TARGET, client)
        client.close()

        failure_findings = [f for f in findings if "Retrieval Failures" in f.technique_name]
        assert len(failure_findings) >= 1

    @respx.mock
    def test_handles_connection_errors_gracefully(self):
        """HTTP errors during probing should not crash."""
        respx.post(TARGET).mock(side_effect=httpx.ConnectError("refused"))
        client = httpx.Client()
        findings, k_stats, g_stats = detect_rag_presence(TARGET, client)
        client.close()

        # Should still return results (with status_code 0 for failed requests)
        assert k_stats.count == 5
        assert g_stats.count == 5

    @respx.mock
    def test_custom_response_field(self):
        """Should extract text from a custom response field."""
        respx.post(TARGET).mock(return_value=httpx.Response(
            200,
            json={"result": "According to our records, the policy is updated quarterly."},
        ))
        client = httpx.Client()
        findings, _, _ = detect_rag_presence(
            TARGET, client, response_field="result",
        )
        client.close()

        citation_findings = [f for f in findings if "Citations" in f.technique_name]
        assert len(citation_findings) >= 1


# --- RD-0101: Knowledge Freshness ---


class TestKnowledgeFreshness:
    @respx.mock
    def test_detects_recent_dates(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            200,
            json={"answer": "The documentation was recently updated on March 2026."},
        ))
        client = httpx.Client()
        findings = detect_knowledge_freshness(TARGET, client)
        client.close()

        assert len(findings) >= 1
        assert findings[0].technique_id == "RD-0101"

    @respx.mock
    def test_no_finding_on_old_dates(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            200,
            json={"answer": "I was trained on data up to 2023."},
        ))
        client = httpx.Client()
        findings = detect_knowledge_freshness(TARGET, client)
        client.close()

        assert len(findings) == 0

    @respx.mock
    def test_handles_errors_gracefully(self):
        respx.post(TARGET).mock(side_effect=httpx.ConnectError("refused"))
        client = httpx.Client()
        findings = detect_knowledge_freshness(TARGET, client)
        client.close()

        assert findings == []


# --- RD-0102: Vector DB Fingerprinting ---


class TestFingerprintVectorDb:
    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    @pytest.mark.parametrize(
        "destination",
        ["/other-heartbeat", "http://other.local/api/v1/heartbeat"],
    )
    def test_redirected_backend_response_cannot_prove_original_endpoint(self, status, destination):
        requests = []
        original = "http://testrag.local:8000/api/v1/heartbeat"

        def handler(request):
            requests.append(str(request.url))
            if str(request.url) == original:
                return httpx.Response(status, headers={"Location": destination})
            if str(request.url).endswith("/other-heartbeat") or request.url.host == "other.local":
                return httpx.Response(200, json={"nanosecond heartbeat": 1234})
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert not any(f.evidence.get("url") == original for f in findings)
        assert not any(url.endswith("/other-heartbeat") or "other.local" in url for url in requests)

    @pytest.mark.parametrize(
        "heartbeat",
        ["NaN", "Infinity", "-Infinity", "1e9999", "true"],
    )
    def test_heartbeat_rejects_nonfinite_and_nonstandard_json(self, heartbeat):
        def handler(request):
            if request.method == "GET" and request.url.port == 8000 and request.url.path == "/api/v1/heartbeat":
                return httpx.Response(200, content='{"nanosecond heartbeat": ' + heartbeat + '}')
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert not any(f.evidence.get("database") == "chromadb" for f in findings)

    @pytest.mark.parametrize("heartbeat", ["1234", "1.5"])
    def test_heartbeat_accepts_finite_json_number(self, heartbeat):
        def handler(request):
            if request.method == "GET" and request.url.port == 8000 and request.url.path == "/api/v1/heartbeat":
                return httpx.Response(200, content='{"nanosecond heartbeat": ' + heartbeat + '}')
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert any(f.evidence.get("database") == "chromadb" for f in findings)

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, json={"description": "<html>Qdrant dashboard</html>"}),
            httpx.Response(200, text="<html><!-- Qdrant dashboard --><body>generic app</body></html>"),
            httpx.Response(200, text="<html><script>Qdrant dashboard</script><body>generic app</body></html>"),
            httpx.Response(200, text="<html><body><p>generic docs mention Qdrant dashboard</p></body></html>"),
            httpx.Response(200, text="<html><head><title>Qdrant mentions in Generic App</title></head></html>"),
            httpx.Response(200, text="<html><body><h1>Qdrant is discussed here</h1></body></html>"),
            httpx.Response(200, text="<html><body><h1 hidden>Qdrant</h1></body></html>"),
            httpx.Response(200, text="<html><body><section hidden><h1>Qdrant</h1></section></body></html>"),
            httpx.Response(200, text="<html><body><h1 aria-hidden='true'>Qdrant</h1></body></html>"),
            httpx.Response(200, text="<html><body><div aria-hidden='true'><h1>Qdrant</h1></div></body></html>"),
            httpx.Response(200, text="<html><body><h1 style='display:none'>Qdrant</h1></body></html>"),
            httpx.Response(200, text="<html><body><div style='visibility: hidden'><h1>Qdrant</h1></div></body></html>"),
            httpx.Response(200, text="<html><body><h1>Qdr<span hidden>ant</span></h1></body></html>"),
            httpx.Response(200, text="<html><body><svg><title>Qdrant</title></svg></body></html>"),
            httpx.Response(200, text="<html><body><title>Qdrant</title></body></html>"),
            httpx.Response(200, text="<html><body>generic</body></html><title>Qdrant</title>"),
            httpx.Response(200, text="<html><body><h1>Qdr</h1><h2>ant</h2></body></html>"),
            httpx.Response(200, text="<html><body>Qdr<div>generic application</div>ant dashboard</body></html>"),
            httpx.Response(200, text="<html>Qdr<body>generic application</body>ant dashboard</html>"),
            httpx.Response(200, text="<html><body><h1>Qdr<br>ant</h1></body></html>"),
            httpx.Response(200, text="<html><body><h1><title>Qdrant</title></h1></body></html>"),
            httpx.Response(200, text="<html><body><h1>Qdr<h2>ant</h2></body></html>"),
            httpx.Response(200, text="<html><body><div hidden/><h1>Qdrant</h1></div></body></html>"),
            httpx.Response(200, text="<html><body><template/><h1>Qdrant</h1></template></body></html>"),
            httpx.Response(200, text='<html><body><h1 aria-hidden="true" aria-hidden="false">Qdrant</h1></body></html>'),
            httpx.Response(200, text='<html><body><h1 style="display:none" style="color:red">Qdrant</h1></body></html>'),
            httpx.Response(200, text="<html><head><title>Generic</title></head><head><title>Qdrant</title></head></html>"),
        ],
    )
    def test_dashboard_rejects_nonvisible_or_incidental_markers(self, response):
        def handler(request):
            if request.method == "GET" and request.url.port == 6333 and request.url.path == "/dashboard":
                return response
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert not any(f.evidence.get("database") == "qdrant" for f in findings)
        assert not any(
            f.confidence == "high" and f.evidence_state is EvidenceState.OBSERVED
            and f.evidence.get("proof") == "qdrant:/dashboard"
            for f in findings
        )

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, text="<!doctype html><html><head><title>Qdrant Dashboard</title></head></html>"),
            httpx.Response(200, text="<html><body><h1>Qdrant</h1></body></html>"),
            httpx.Response(200, text="<html><head><title> QDRANT&nbsp;DASHBOARD </title></head></html>"),
            httpx.Response(200, text="<html><body><h1>Qdr<span>ant</span></h1></body></html>"),
            httpx.Response(200, text="<html><body><h1>Generic</h1><h2>Qdrant</h2></body></html>"),
            httpx.Response(200, text="<html><body> Qdr<span>ant</span>&nbsp;dashboard </body></html>"),
        ],
    )
    def test_dashboard_accepts_visible_product_branding(self, response):
        def handler(request):
            if request.method == "GET" and request.url.port == 6333 and request.url.path == "/dashboard":
                return response
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert any(f.evidence.get("database") == "qdrant" for f in findings)

    @pytest.mark.parametrize("check", ["classifier", "discovery"])
    @pytest.mark.parametrize(
        "body, expected",
        [
            pytest.param(body, False, id=f"{opening}-closed-by-{closing}-{variant}")
            for opening in ("h1", "h2")
            for closing in ("h1", "h2", "h3", "h4", "h5", "h6")
            if opening != closing
            for variant, body in (
                ("split", f"<{opening}>Qdr</{closing}>ant</{opening}>"),
                ("inline", f"<{opening}><span>Qdr</{closing}>ant</span></{opening}>"),
                ("complete", f"<{opening}>Qdrant</{closing}></{opening}>"),
                ("adjacent", f"<{opening}>Qdr</{closing}>ant</{opening}><h3>Generic</h3>"),
            )
        ] + [
            pytest.param("<h1>Qdr<h3>ant</h3></h1>", False, id="nested-split"),
            pytest.param("<h1>Qdr<h6></h6>ant</h1>", False, id="nested-empty"),
            pytest.param("<h3><h1>Qdrant</h1></h3>", False, id="nested-complete-h1"),
            pytest.param("<h6><h2>Qdrant</h2></h6>", False, id="nested-complete-h2"),
            pytest.param("<h3><h1>Qdrant</h3></h1>", False, id="mixed-heading-close"),
            pytest.param("<h1>Qdr</h1><h2>ant</h2>", False, id="adjacent-split"),
            pytest.param("<h1>Qdr</h4><h2>ant</h2></h1>", False, id="mixed-split"),
            pytest.param("<h1>Qdr</H6>ant</h1>", False, id="uppercase-close"),
            pytest.param("<h1>Qdr</h3 >ant</h1>", False, id="spaced-close"),
            pytest.param("<h1>Qdr<span><strong>ant</strong></span></h1>", True, id="nested-inline-h1"),
            pytest.param("<h2>Qdr<em><span>ant</span></em>&nbsp;Dashboard</h2>", True, id="nested-inline-h2"),
            pytest.param("<h1>Generic</h1><h2>Qdrant</h2>", True, id="adjacent-valid-last"),
            pytest.param("<h1>Qdrant</h1><h2>Generic</h2>", True, id="adjacent-valid-first"),
            pytest.param("<h3>Generic</h3><h1>Qdrant</h1><h6>Other</h6>", True, id="mixed-level-adjacent"),
            pytest.param("<section><h2>Qdrant</h2></section>", True, id="ordinary-container"),
        ],
    )
    def test_dashboard_heading_boundaries(self, body, expected, check):
        response = httpx.Response(200, text=f"<html><body>{body}</body></html>")
        if check == "classifier":
            assert _matches_endpoint_signature(
                {"db": "qdrant", "path": "/dashboard"}, response,
            ) is expected
            return

        def handler(request):
            if request.method == "GET" and request.url.port == 6333 and request.url.path == "/dashboard":
                return response
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert any(
            f.confidence == "high" and f.evidence_state is EvidenceState.OBSERVED
            and f.evidence.get("proof") == "qdrant:/dashboard"
            for f in findings
        ) is expected
        if not expected:
            assert not any(f.evidence.get("database") == "qdrant" for f in findings)

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, json={"status": "ok"}, headers={"X-Not-Pinecone": "no"}),
            httpx.Response(200, json={"dimension": None}),
            httpx.Response(200, json={"dimension": True}),
            httpx.Response(200, json={"dimension": -1}),
            httpx.Response(200, json={"indexFullness": None}),
            httpx.Response(200, json={"indexFullness": 10 ** 1000}),
            httpx.Response(200, json={"namespaces": None}),
            httpx.Response(200, json={"namespaces": {"example": None}}),
            httpx.Response(200, headers={"X-Pinecone-Request-Latency-Ms": "invalid"}),
        ],
    )
    def test_approved_pinecone_origin_rejects_lookalikes(self, response):
        def handler(request):
            if request.method == "GET" and request.url.port == 443 and request.url.path == "/describe_index_stats":
                return response
            return httpx.Response(404)

        profile = TargetProfile.from_cli(TARGET, additional_origin_headers={"http://testrag.local:443": {}})
        with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert not any(f.evidence.get("database") == "pinecone" for f in findings)

    @pytest.mark.parametrize(
        "payload",
        [{"dimension": 768}, {"indexFullness": 0.2}, {"namespaces": {}}],
    )
    def test_approved_pinecone_origin_accepts_valid_stats(self, payload):
        def handler(request):
            if request.method == "GET" and request.url.port == 443 and request.url.path == "/describe_index_stats":
                return httpx.Response(200, json=payload)
            return httpx.Response(404)

        profile = TargetProfile.from_cli(TARGET, additional_origin_headers={"http://testrag.local:443": {}})
        with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert any(f.evidence.get("database") == "pinecone" for f in findings)

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(401, json={"detail": "login"}),
            httpx.Response(403, text="forbidden"),
            httpx.Response(404, text="not found"),
            httpx.Response(200, text="<html>generic app</html>"),
            httpx.Response(200, json={"status": "ok"}),
            httpx.Response(200),
            httpx.Response(200, text="{bad json"),
            httpx.Response(200, content=b"\xff"),
        ],
    )
    def test_endpoint_scan_requires_structured_backend_proof(self, response):
        with OriginBoundClient(
            TargetProfile.from_cli(TARGET),
            transport=httpx.MockTransport(lambda request: response),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert [f for f in findings if "Endpoint Scan" in f.technique_name] == []

    @pytest.mark.parametrize(
        ("port", "path", "response", "database"),
        [
            (8000, "/api/v1/heartbeat", httpx.Response(200, json={"nanosecond heartbeat": 1234}), "chromadb"),
            (8000, "/api/v1/collections", httpx.Response(200, json=[]), "chromadb"),
            (6333, "/collections", httpx.Response(200, json={"status": "ok", "result": {"collections": []}}), "qdrant"),
            (6333, "/telemetry", httpx.Response(200, json={"result": {"app": {"name": "qdrant"}}}), "qdrant"),
            (6333, "/dashboard", httpx.Response(200, text="<html>Qdrant dashboard</html>"), "qdrant"),
            (8080, "/v1/meta", httpx.Response(200, json={"version": "1.0", "modules": {}}), "weaviate"),
            (8080, "/v1/schema", httpx.Response(200, json={"classes": []}), "weaviate"),
            (8080, "/v1/.well-known/ready", httpx.Response(200, text="ready"), "weaviate"),
            (19530, "/api/v1/health", httpx.Response(200, json={"code": 0, "message": "healthy"}), "milvus"),
            (19530, "/v2/vectordb/collections/list", httpx.Response(200, json={"code": 0, "data": {"collectionNames": []}}), "milvus"),
        ],
    )
    def test_structured_endpoint_proof_is_observed(self, port, path, response, database):
        def handler(request):
            if request.method == "GET" and request.url.port == port and request.url.path == path:
                return response
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        matches = [f for f in findings if f.evidence.get("url") == f"http://testrag.local:{port}{path}"]
        assert len(matches) == 1
        assert matches[0].evidence["database"] == database
        assert matches[0].evidence["proof"]
        assert matches[0].evidence_state is EvidenceState.OBSERVED

    def test_pinecone_port_requires_explicitly_approved_origin(self):
        requested = []

        def handler(request):
            requested.append(str(request.url))
            if request.method == "POST":
                return httpx.Response(400, json={"error": "invalid input"})
            return httpx.Response(200, json={"dimension": 768})

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert not any(":443/describe_index_stats" in url for url in requested)
        assert not any(f.evidence.get("database") == "pinecone" for f in findings)

    def test_https_chat_origin_is_not_pinecone_approval(self):
        target = "https://testrag.local/chat"
        requested = []

        def handler(request):
            requested.append(str(request.url))
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(target), transport=httpx.MockTransport(handler),
        ) as client:
            fingerprint_vector_db(target, client, scan_ports=True)
        assert not any("describe_index_stats" in url for url in requested)

    def test_explicit_pinecone_origin_accepts_product_header(self):
        def handler(request):
            if request.method == "GET" and request.url.port == 443 and request.url.path == "/describe_index_stats":
                return httpx.Response(200, headers={"X-Pinecone-Request-Latency-Ms": "3"})
            return httpx.Response(404)

        profile = TargetProfile.from_cli(
            TARGET, additional_origin_headers={"http://testrag.local:443": {}},
        )
        with OriginBoundClient(profile, transport=httpx.MockTransport(handler)) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert any(f.evidence.get("database") == "pinecone" for f in findings)

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(200, text="qdrant"),
            httpx.Response(200, json={"message": "qdrant"}),
            httpx.Response(200, text="<html>generic app mentions qdrant</html>"),
        ],
    )
    def test_dashboard_requires_explicit_html_product_marker(self, response):
        def handler(request):
            if request.method == "GET" and request.url.port == 6333 and request.url.path == "/dashboard":
                return response
            return httpx.Response(404)

        with OriginBoundClient(
            TargetProfile.from_cli(TARGET), transport=httpx.MockTransport(handler),
        ) as client:
            findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        assert not any(f.evidence.get("url") == "http://testrag.local:6333/dashboard" for f in findings)

    @respx.mock
    def test_detects_chromadb_in_errors(self):
        """Error responses mentioning ChromaDB should trigger a finding."""
        respx.post(TARGET).mock(return_value=httpx.Response(
            500,
            text="chromadb.errors.InvalidCollectionException: collection not found",
        ))
        # Mock port scans to fail (not testing that here)
        respx.route().mock(return_value=httpx.Response(500))
        client = httpx.Client()
        findings = fingerprint_vector_db(TARGET, client, scan_ports=False)
        client.close()

        db_names = [f.evidence.get("database") for f in findings]
        assert "chromadb" in db_names

    @respx.mock
    def test_detects_qdrant_in_errors(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            500,
            text='{"error": "qdrant: points_count mismatch"}',
        ))
        client = httpx.Client()
        findings = fingerprint_vector_db(TARGET, client, scan_ports=False)
        client.close()

        db_names = [f.evidence.get("database") for f in findings]
        assert "qdrant" in db_names

    @respx.mock
    def test_no_false_positives_on_clean_errors(self):
        respx.post(TARGET).mock(return_value=httpx.Response(
            400,
            text='{"error": "invalid input"}',
        ))
        client = httpx.Client()
        findings = fingerprint_vector_db(TARGET, client, scan_ports=False)
        client.close()

        assert len(findings) == 0

    @respx.mock
    def test_endpoint_scan_detects_open_service(self):
        """Accessible vector DB endpoints should be reported."""
        # Mock the error probes to return nothing useful
        respx.post(TARGET).mock(return_value=httpx.Response(400, text="bad request"))
        # Mock a ChromaDB heartbeat endpoint
        respx.get("http://testrag.local:8000/api/v1/heartbeat").mock(
            return_value=httpx.Response(200, json={"nanosecond heartbeat": 1234}),
        )
        # All other endpoints fail
        respx.route().mock(side_effect=httpx.ConnectError("refused"))

        client = httpx.Client()
        findings = fingerprint_vector_db(TARGET, client, scan_ports=True)
        client.close()

        endpoint_findings = [f for f in findings if "Endpoint Scan" in f.technique_name]
        assert len(endpoint_findings) >= 1

    @respx.mock
    def test_skip_port_scan(self):
        """scan_ports=False should skip endpoint scanning."""
        respx.post(TARGET).mock(return_value=httpx.Response(400, text="bad"))
        client = httpx.Client()
        findings = fingerprint_vector_db(TARGET, client, scan_ports=False)
        client.close()

        endpoint_findings = [f for f in findings if "Endpoint Scan" in f.technique_name]
        assert len(endpoint_findings) == 0


# --- run_full_fingerprint ---


class TestRunFullFingerprint:
    @respx.mock
    def test_full_run_aggregates_findings(self):
        """Full fingerprint run should combine RD-0101 and RD-0102 findings."""
        respx.post(TARGET).mock(return_value=httpx.Response(
            200,
            json={"answer": "According to our documentation from March 2026, the policy states..."},
        ))
        # Mock port scans to fail
        respx.route().mock(side_effect=httpx.ConnectError("refused"))

        client = httpx.Client()
        result = run_full_fingerprint(TARGET, client, scan_ports=False)
        client.close()

        assert isinstance(result, FingerprintResult)
        assert result.target == TARGET
        assert result.rag_detected is True
        assert result.timing_stats is not None
        assert "knowledge_queries" in result.timing_stats
        assert "general_queries" in result.timing_stats
        assert len(result.findings) >= 1

    @respx.mock
    def test_no_detection_on_empty_responses(self):
        """No RAG indicators should mean rag_detected=False."""
        respx.post(TARGET).mock(return_value=httpx.Response(200, text="OK"))
        respx.route().mock(side_effect=httpx.ConnectError("refused"))

        client = httpx.Client()
        result = run_full_fingerprint(TARGET, client, scan_ports=False)
        client.close()

        assert result.rag_detected is False
        assert result.vector_db is None
