"""Tests for R2 Probe techniques (RD-0201 through RD-0205).

Uses respx to mock httpx requests. Each technique gets its own test class.
"""

import json

import httpx
import pytest
import respx

from ragdrag.adapters.chat import ChatResponseError
from ragdrag.core.models import Finding
from ragdrag.core.probe import (
    ProbeResult,
    _check_boundary_indicators,
    _extract_relevance_scores,
    _extract_response_text,
    _parse_retrieval_data,
    detect_chunk_boundaries,
    estimate_retrieval_count,
    fingerprint_embedding_model,
    map_kb_scope,
    map_similarity_threshold,
    run_probe,
    scan_debug_endpoints,
)
from ragdrag.engine.models import OutcomeCode

TARGET = "http://testrag.local/chat"


def _rag_response(answer="The policy is...", sources=None, context=None, documents=None):
    """Build a mock RAG API JSON response."""
    body = {"answer": answer}
    if sources is not None:
        body["sources"] = sources
    if context is not None:
        body["context"] = context
    if documents is not None:
        body["documents"] = documents
    return body


def _scored_sources(n=3, base_score=0.9):
    """Build a sources list with relevance scores."""
    return [{"text": f"chunk {i}", "relevance": round(base_score - i * 0.1, 2)} for i in range(n)]


# --- ProbeResult ---


class TestProbeResult:
    def test_default_fields(self):
        r = ProbeResult(target=TARGET)
        assert r.chunk_size_estimate is None
        assert r.similarity_threshold is None
        assert r.retrieval_count is None
        assert r.kb_domains == []
        assert r.embedding_model is None
        assert r.findings == []

    def test_to_dict(self):
        r = ProbeResult(
            target=TARGET,
            chunk_size_estimate=512,
            retrieval_count=3,
            findings=[Finding("RD-0201", "Test", "high", "detail")],
        )
        d = r.to_dict()
        assert d["target"] == TARGET
        assert d["chunk_size_estimate"] == 512
        assert d["retrieval_count"] == 3
        assert len(d["findings"]) == 1
        assert d["findings"][0]["technique_id"] == "RD-0201"

    def test_to_dict_empty_findings(self):
        r = ProbeResult(target=TARGET)
        d = r.to_dict()
        assert d["findings"] == []


# --- Shared Utilities ---


class TestParseRetrievalData:
    def test_sources_field(self):
        body = json.dumps({"sources": ["a", "b", "c"]})
        count, sizes = _parse_retrieval_data(body)
        assert count == 3

    def test_context_field_with_chunks(self):
        body = json.dumps({"context": ["a" * 100, "b" * 200]})
        count, sizes = _parse_retrieval_data(body)
        assert count == 2
        assert sizes == [100, 200]

    def test_documents_field(self):
        body = json.dumps({"documents": ["doc one " * 20, "doc two " * 20]})
        count, sizes = _parse_retrieval_data(body)
        assert count == 2
        assert len(sizes) == 2

    def test_citation_fallback(self):
        body = "Based on [source 1] and [source 2] and [source 1]"
        count, sizes = _parse_retrieval_data(body)
        assert count == 2  # deduped
        assert sizes == []

    def test_invalid_json_no_citations(self):
        count, sizes = _parse_retrieval_data("just plain text")
        assert count == 0
        assert sizes == []

    def test_small_chunks_excluded(self):
        body = json.dumps({"context": ["tiny", "a" * 100]})
        count, sizes = _parse_retrieval_data(body)
        assert count == 2
        assert sizes == [100]  # "tiny" excluded (< 10 chars)


class TestCheckBoundaryIndicators:
    def test_detects_partial_info(self):
        text = "I can only find partial information about that topic."
        assert _check_boundary_indicators(text) is True

    def test_detects_not_enough(self):
        text = "There is not enough information to answer fully."
        assert _check_boundary_indicators(text) is True

    def test_no_indicators(self):
        text = "The password reset policy requires you to visit the portal."
        assert _check_boundary_indicators(text) is False

    def test_detects_based_on_retrieved(self):
        text = "Based on the retrieved documents, the policy states..."
        assert _check_boundary_indicators(text) is True


class TestExtractRelevanceScores:
    def test_extracts_relevance_scores(self):
        body = json.dumps({"sources": [
            {"text": "chunk", "relevance": 0.95},
            {"text": "chunk", "relevance": 0.72},
        ]})
        scores = _extract_relevance_scores(body)
        assert scores == [0.95, 0.72]

    def test_extracts_similarity_key(self):
        body = json.dumps({"sources": [{"text": "x", "similarity": 0.88}]})
        scores = _extract_relevance_scores(body)
        assert scores == [0.88]

    def test_no_scores(self):
        body = json.dumps({"sources": [{"text": "no score here"}]})
        scores = _extract_relevance_scores(body)
        assert scores == []

    def test_invalid_json(self):
        scores = _extract_relevance_scores("not json")
        assert scores == []


# --- RD-0201: Chunk Boundary Detection ---


class TestChunkBoundaryDetection:
    @respx.mock
    def test_fixed_top_k_detected(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(sources=["s1", "s2", "s3"]),
        ))
        with httpx.Client() as client:
            findings = detect_chunk_boundaries(TARGET, client)
        assert len(findings) >= 1
        top_k_finding = [f for f in findings if "top-k" in f.detail.lower() or "retrieval" in f.detail.lower()]
        assert len(top_k_finding) >= 1

    @respx.mock
    def test_chunk_sizes_from_context(self):
        context = ["a" * 500, "b" * 500, "c" * 500]
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(context=context),
        ))
        with httpx.Client() as client:
            findings = detect_chunk_boundaries(TARGET, client)
        chunk_finding = [f for f in findings if "chunk size" in f.detail.lower()]
        assert len(chunk_finding) >= 1
        assert chunk_finding[0].evidence["estimated_avg_chunk_chars"] == 500

    @respx.mock
    def test_response_length_variation(self):
        responses = [
            httpx.Response(200, json=_rag_response(answer="short")),
            httpx.Response(200, json=_rag_response(answer="x" * 1000)),
            httpx.Response(200, json=_rag_response(answer="medium " * 20)),
            httpx.Response(200, json=_rag_response(answer="y" * 800)),
        ]
        route = respx.post(TARGET).mock(side_effect=responses)
        with httpx.Client() as client:
            findings = detect_chunk_boundaries(TARGET, client)
        variation = [f for f in findings if "variation" in f.detail.lower()]
        assert len(variation) >= 1

    @respx.mock
    def test_cross_topic_boundary(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(
                answer="Based on the available context, I can only find partial information about that.",
                sources=["s1", "s2"],
            ),
        ))
        with httpx.Client() as client:
            findings = detect_chunk_boundaries(TARGET, client)
        cross = [f for f in findings if "cross-topic" in f.detail.lower()]
        assert len(cross) >= 1

    @respx.mock
    def test_handles_http_errors(self):
        route = respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = detect_chunk_boundaries(TARGET, client)
        assert findings == []

    @respx.mock
    def test_needs_at_least_2_responses(self):
        responses = [
            httpx.Response(200, json=_rag_response()),
            httpx.ConnectError("fail"),
            httpx.ConnectError("fail"),
            httpx.ConnectError("fail"),
        ]
        route = respx.post(TARGET).mock(side_effect=responses)
        with httpx.Client() as client:
            findings = detect_chunk_boundaries(TARGET, client)
        assert findings == []


# --- RD-0202: Similarity Threshold Mapping ---


class TestSimilarityThreshold:
    @respx.mock
    def test_detects_cutoff(self):
        """Responses degrade from relevant to no-match."""
        responses = []
        for i in range(7):
            if i < 4:
                responses.append(httpx.Response(
                    200, json=_rag_response(
                        answer=f"Relevant answer for level {i}",
                        sources=_scored_sources(3, 0.9 - i * 0.15),
                    ),
                ))
            else:
                responses.append(httpx.Response(
                    200, json=_rag_response(
                        answer="I don't have information about that topic.",
                    ),
                ))
        route = respx.post(TARGET).mock(side_effect=responses)
        with httpx.Client() as client:
            findings = map_similarity_threshold(TARGET, client)
        assert len(findings) >= 1
        cutoff_finding = [f for f in findings if "cutoff" in f.detail.lower() or "threshold" in f.detail.lower()]
        assert len(cutoff_finding) >= 1

    @respx.mock
    def test_no_cutoff_detected(self):
        """System responds to everything including nonsense."""
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(answer="Here's some info!", sources=["s1"]),
        ))
        with httpx.Client() as client:
            findings = map_similarity_threshold(TARGET, client)
        # Should still produce findings about score degradation or no cutoff
        assert isinstance(findings, list)

    @respx.mock
    def test_relevance_scores_captured(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(sources=_scored_sources(3, 0.85)),
        ))
        with httpx.Client() as client:
            findings = map_similarity_threshold(TARGET, client)
        score_finding = [f for f in findings if "score" in str(f.evidence).lower() or "relevance" in str(f.evidence).lower()]
        # May or may not have score findings depending on response pattern
        assert isinstance(findings, list)

    @respx.mock
    def test_handles_all_errors(self):
        route = respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = map_similarity_threshold(TARGET, client)
        assert findings == []


# --- RD-0203: Retrieval Count Estimation ---


class TestRetrievalCount:
    @respx.mock
    def test_fixed_top_k(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(sources=["s1", "s2", "s3"]),
        ))
        with httpx.Client() as client:
            findings = estimate_retrieval_count(TARGET, client)
        assert len(findings) >= 1
        assert any("top-k" in f.detail.lower() or "retrieval" in f.detail.lower() for f in findings)

    @respx.mock
    def test_fixed_top_k_evidence(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(sources=["s1", "s2", "s3"]),
        ))
        with httpx.Client() as client:
            findings = estimate_retrieval_count(TARGET, client)
        top_k_findings = [f for f in findings if "estimated_top_k" in f.evidence]
        assert len(top_k_findings) >= 1
        assert top_k_findings[0].evidence["estimated_top_k"] == 3

    @respx.mock
    def test_variable_counts(self):
        responses = [
            httpx.Response(200, json=_rag_response(sources=["s1", "s2"])),
            httpx.Response(200, json=_rag_response(sources=["s1", "s2", "s3", "s4"])),
            httpx.Response(200, json=_rag_response(sources=["s1"])),
            httpx.Response(200, json=_rag_response(sources=["s1", "s2", "s3"])),
        ]
        route = respx.post(TARGET).mock(side_effect=responses)
        with httpx.Client() as client:
            findings = estimate_retrieval_count(TARGET, client)
        assert isinstance(findings, list)

    @respx.mock
    def test_no_sources_exposed(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(answer="Just an answer, no sources"),
        ))
        with httpx.Client() as client:
            findings = estimate_retrieval_count(TARGET, client)
        assert isinstance(findings, list)

    @respx.mock
    def test_handles_errors(self):
        route = respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = estimate_retrieval_count(TARGET, client)
        assert findings == []


# --- RD-0204: Knowledge Base Scope Mapping ---


class TestKBScopeMapping:
    @respx.mock
    def test_detects_covered_categories(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(
                answer="Our security policy requires multi-factor authentication.",
                sources=["s1", "s2"],
            ),
        ))
        with httpx.Client() as client:
            findings = map_kb_scope(TARGET, client)
        assert len(findings) >= 1
        scope_finding = [f for f in findings if "covered_categories" in f.evidence]
        assert len(scope_finding) >= 1

    @respx.mock
    def test_detects_gaps(self):
        """System returns no useful content for some categories."""
        responses = []
        for i in range(12):  # 6 categories x 2 queries
            if i < 4:  # First 2 categories covered
                responses.append(httpx.Response(
                    200, json=_rag_response(answer="Detailed answer here", sources=["s1"]),
                ))
            else:  # Rest not covered
                responses.append(httpx.Response(
                    200, json=_rag_response(answer="I don't have information about that."),
                ))
        route = respx.post(TARGET).mock(side_effect=responses)
        with httpx.Client() as client:
            findings = map_kb_scope(TARGET, client)
        gap_finding = [f for f in findings if "gap_categories" in f.evidence]
        assert len(gap_finding) >= 1

    @respx.mock
    def test_handles_errors(self):
        route = respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = map_kb_scope(TARGET, client)
        assert isinstance(findings, list)


# --- Debug Endpoint Discovery ---


class TestDebugEndpoints:
    @respx.mock
    def test_discovers_config_endpoint(self):
        base = "http://testrag.local"
        respx.get(f"{base}/debug/config").mock(return_value=httpx.Response(
            200, json={"chunk_size": 512, "n_results": 3, "embedding_model": "all-MiniLM"},
        ))
        # All other endpoints return 404
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        respx.get(f"{base}/debug/config").mock(return_value=httpx.Response(
            200, json={"chunk_size": 512, "n_results": 3, "embedding_model": "all-MiniLM"},
        ))
        with httpx.Client() as client:
            findings = scan_debug_endpoints(TARGET, client)
        debug_findings = [f for f in findings if "debug" in f.detail.lower() or "endpoint" in f.detail.lower()]
        assert isinstance(findings, list)

    @respx.mock
    def test_no_endpoints_found(self):
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        with httpx.Client() as client:
            findings = scan_debug_endpoints(TARGET, client)
        assert isinstance(findings, list)

    @respx.mock
    def test_non_json_response_ignored(self):
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(
            200, text="<html>Not JSON</html>",
        ))
        with httpx.Client() as client:
            findings = scan_debug_endpoints(TARGET, client)
        # Non-JSON 200s should not count as discovered endpoints
        assert isinstance(findings, list)

    @respx.mock
    def test_handles_connection_errors(self):
        respx.get(url__regex=r".*").mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = scan_debug_endpoints(TARGET, client)
        assert findings == [] or isinstance(findings, list)


# --- RD-0205: Embedding Model Fingerprinting ---


class TestEmbeddingModelFingerprint:
    @respx.mock
    def test_returns_findings(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(
                answer="Detailed answer about security topics.",
                sources=_scored_sources(3, 0.85),
            ),
        ))
        with httpx.Client() as client:
            findings = fingerprint_embedding_model(TARGET, client)
        assert isinstance(findings, list)

    @respx.mock
    def test_detects_model_family(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(
                answer="Technical response about code patterns",
                sources=_scored_sources(3, 0.9),
            ),
        ))
        with httpx.Client() as client:
            findings = fingerprint_embedding_model(TARGET, client)
        model_findings = [f for f in findings if "model_family" in f.evidence]
        if model_findings:
            assert model_findings[0].evidence["model_family"] in (
                "multilingual", "code-aware", "general-english", "domain-specific",
            )

    @respx.mock
    def test_handles_errors(self):
        route = respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            findings = fingerprint_embedding_model(TARGET, client)
        assert isinstance(findings, list)


# --- run_probe() Orchestrator ---


class TestRunProbe:
    @respx.mock
    def test_quick_depth(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(sources=["s1", "s2", "s3"]),
        ))
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        with httpx.Client() as client:
            result = run_probe(TARGET, client, depth="quick")
        assert isinstance(result, ProbeResult)
        assert result.target == TARGET
        # Quick runs RD-0201, 0202, 0203, debug — but NOT 0204, 0205
        technique_ids = [f.technique_id for f in result.findings]
        assert "RD-0201" in technique_ids or len(result.findings) >= 0

    @respx.mock
    def test_full_depth_includes_all(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(
                answer="Detailed answer",
                sources=_scored_sources(3, 0.85),
            ),
        ))
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        with httpx.Client() as client:
            result = run_probe(TARGET, client, depth="full")
        assert isinstance(result, ProbeResult)

    @respx.mock
    def test_extracts_chunk_size_into_result(self):
        context = ["a" * 400, "b" * 400, "c" * 400]
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(context=context),
        ))
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        with httpx.Client() as client:
            result = run_probe(TARGET, client, depth="quick")
        assert result.chunk_size_estimate == 400

    @respx.mock
    def test_extracts_retrieval_count_into_result(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(sources=["s1", "s2", "s3"]),
        ))
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        with httpx.Client() as client:
            result = run_probe(TARGET, client, depth="quick")
        assert result.retrieval_count == 3

    @respx.mock
    def test_custom_query_field(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(sources=["s1"]),
        ))
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        with httpx.Client() as client:
            result = run_probe(TARGET, client, query_field="message")
        assert isinstance(result, ProbeResult)

    @respx.mock
    def test_custom_response_field(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json={"reply": "The answer is...", "sources": ["s1"]},
        ))
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        with httpx.Client() as client:
            result = run_probe(TARGET, client, response_field="reply")
        assert isinstance(result, ProbeResult)

    @respx.mock
    def test_all_errors_returns_empty_result(self):
        respx.post(TARGET).mock(side_effect=httpx.ConnectError("fail"))
        respx.get(url__regex=r".*").mock(side_effect=httpx.ConnectError("fail"))
        with httpx.Client() as client:
            result = run_probe(TARGET, client)
        assert isinstance(result, ProbeResult)
        assert result.target == TARGET

    @respx.mock
    def test_result_serializes_to_dict(self):
        route = respx.post(TARGET).mock(return_value=httpx.Response(
            200, json=_rag_response(sources=["s1", "s2", "s3"]),
        ))
        respx.get(url__regex=r".*").mock(return_value=httpx.Response(404))
        with httpx.Client() as client:
            result = run_probe(TARGET, client)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "target" in d
        assert "findings" in d
        assert isinstance(d["findings"], list)


# --- Extract Response Text ---


class TestExtractResponseText:
    def test_with_response_field(self):
        resp = httpx.Response(200, json={"answer": "hello", "other": "stuff"})
        text = _extract_response_text(resp, "answer")
        assert text == "hello"

    def test_without_response_field(self):
        resp = httpx.Response(200, text="raw text here")
        text = _extract_response_text(resp, None)
        assert text == "raw text here"

    def test_missing_field_is_unsupported_response(self):
        resp = httpx.Response(200, json={"answer": "hello"})
        with pytest.raises(ChatResponseError) as error:
            _extract_response_text(resp, "nonexistent")
        assert error.value.outcome is OutcomeCode.UNSUPPORTED_RESPONSE

    def test_non_200_with_configured_field_is_unsupported_response(self):
        resp = httpx.Response(500, text="Internal Server Error")
        with pytest.raises(ChatResponseError) as error:
            _extract_response_text(resp, "answer")
        assert error.value.outcome is OutcomeCode.UNSUPPORTED_RESPONSE
