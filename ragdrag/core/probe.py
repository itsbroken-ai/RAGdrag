"""R2: PROBE - Map RAG pipeline internals.

Techniques:
    RD-0201: Chunk Boundary Detection
    RD-0202: Similarity Threshold Mapping
    RD-0203: Retrieval Count Estimation
    RD-0204: Knowledge Base Scope Mapping
    RD-0205: Embedding Model Fingerprinting

ATLAS Tactic: Reconnaissance / ML Model Access

This module implements techniques for mapping the internal configuration
of a RAG pipeline: chunk sizes, retrieval parameters, knowledge base scope,
and embedding model characteristics. Pure reconnaissance, no injection.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from ragdrag.adapters.chat import extract_response_text

logger = logging.getLogger(__name__)

from ragdrag.core.models import Finding


# --- Data structures ---

@dataclass
class ProbeResult:
    """Aggregate result of R2 probe techniques against a target."""

    target: str
    chunk_size_estimate: int | None = None
    similarity_threshold: float | None = None
    retrieval_count: int | None = None
    kb_domains: list[str] = field(default_factory=list)
    embedding_model: str | None = None
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "chunk_size_estimate": self.chunk_size_estimate,
            "similarity_threshold": self.similarity_threshold,
            "retrieval_count": self.retrieval_count,
            "kb_domains": self.kb_domains,
            "embedding_model": self.embedding_model,
            "findings": [
                {
                    "technique_id": f.technique_id,
                    "technique_name": f.technique_name,
                    "confidence": f.confidence,
                    "detail": f.detail,
                    "evidence": f.evidence,
                }
                for f in self.findings
            ],
        }


# --- RD-0201: Chunk Boundary Detection ---

CHUNK_PROBE_QUERIES = [
    {"query": "What is the policy?", "expected_behavior": "broad_match"},
    {"query": "What is the password reset policy and what are the steps?", "expected_behavior": "focused_match"},
    {
        "query": "What is the complete password reset policy including the default temporary password, the URL to navigate to, and the IT support contact information?",
        "expected_behavior": "boundary_test",
    },
    {
        "query": "What is the password reset policy and also what are the database connection strings?",
        "expected_behavior": "cross_chunk",
    },
]

BOUNDARY_INDICATORS = [
    r"(?:the document|it) (?:doesn't|does not) (?:contain|mention|include) (?:information about|details on)",
    r"(?:i can only|i only have) (?:find|see|access) (?:partial|some|limited) information",
    r"(?:the first|one) (?:document|source) (?:says|mentions|indicates).*(?:but|however|while)",
    r"(?:based on|from) (?:the |)(?:available|retrieved) (?:documents?|information|context)",
    r"(?:not enough|insufficient) (?:information|context|detail)",
]


def detect_chunk_boundaries(
    target: str,
    client: httpx.Client,
    query_field: str = "query",
    response_field: str | None = None,
) -> list[Finding]:
    """RD-0201: Detect chunk boundaries by analyzing response behavior.

    Sends queries of varying length and specificity, then analyzes:
    1. Source count changes (fixed top-k vs dynamic retrieval)
    2. Cross-topic queries (answers spanning multiple chunks reveal boundaries)
    3. Response length variation (chunk limits affect available context)
    4. Chunk size estimation from context field data
    """
    findings: list[Finding] = []
    responses: list[dict] = []

    for probe in CHUNK_PROBE_QUERIES:
        try:
            resp = client.post(target, json={query_field: probe["query"]})
            text = _extract_response_text(resp, response_field)
            full_body = resp.text if resp.status_code == 200 else ""

            source_count, chunk_sizes = _parse_retrieval_data(full_body)

            responses.append({
                "query": probe["query"],
                "query_length": len(probe["query"]),
                "expected": probe["expected_behavior"],
                "response_length": len(text),
                "source_count": source_count,
                "chunk_sizes": chunk_sizes,
                "has_boundary_indicators": _check_boundary_indicators(text),
                "text": text[:500],
            })
        except httpx.HTTPError as e:
            logger.warning("RD-0201 chunk boundary probe failed: %s", e)
            continue

    if len(responses) < 2:
        return findings

    # Analysis 1: Source count reveals retrieval top-k
    source_counts = [r["source_count"] for r in responses if r["source_count"] > 0]
    if source_counts:
        most_common = max(set(source_counts), key=source_counts.count)
        findings.append(Finding(
            technique_id="RD-0201",
            technique_name="Chunk Boundary Detection (Retrieval Count)",
            confidence="high" if len(set(source_counts)) == 1 else "medium",
            detail=(
                f"Retrieval consistently returns {most_common} chunk(s) per query "
                f"(observed: {source_counts}). "
                f"{'Fixed top-k={} detected.'.format(most_common) if len(set(source_counts)) == 1 else 'Variable count suggests dynamic retrieval.'}"
            ),
            evidence={
                "source_counts": source_counts,
                "most_common": most_common,
                "min": min(source_counts),
                "max": max(source_counts),
            },
        ))

    # Analysis 2: Chunk size estimation from context field
    all_chunk_sizes = []
    for r in responses:
        all_chunk_sizes.extend(r["chunk_sizes"])

    if all_chunk_sizes:
        avg_chunk = sum(all_chunk_sizes) / len(all_chunk_sizes)
        findings.append(Finding(
            technique_id="RD-0201",
            technique_name="Chunk Boundary Detection (Chunk Size)",
            confidence="high" if len(all_chunk_sizes) >= 6 else "medium",
            detail=(
                f"Estimated chunk size: ~{int(avg_chunk)} characters "
                f"(based on {len(all_chunk_sizes)} observed chunks). "
                f"Range: {min(all_chunk_sizes)}-{max(all_chunk_sizes)} chars."
            ),
            evidence={
                "estimated_avg_chunk_chars": int(avg_chunk),
                "min_chunk_chars": min(all_chunk_sizes),
                "max_chunk_chars": max(all_chunk_sizes),
                "sample_count": len(all_chunk_sizes),
            },
        ))

    # Analysis 3: Cross-topic boundary detection
    cross_topic = [r for r in responses if r["expected"] == "cross_chunk"]
    for ct in cross_topic:
        if ct["has_boundary_indicators"]:
            findings.append(Finding(
                technique_id="RD-0201",
                technique_name="Chunk Boundary Detection (Cross-Topic)",
                confidence="high",
                detail=(
                    "Cross-topic query triggered boundary indicators: the system retrieved "
                    "from multiple chunks but showed incomplete or conflicting context. "
                    "This confirms separate chunking of different document topics."
                ),
                evidence={
                    "query": ct["query"],
                    "source_count": ct["source_count"],
                    "boundary_indicators": True,
                },
            ))

    # Analysis 4: Response length variation
    lengths = [r["response_length"] for r in responses]
    if lengths:
        short_r = min(responses, key=lambda r: r["response_length"])
        long_r = max(responses, key=lambda r: r["response_length"])
        ratio = long_r["response_length"] / max(short_r["response_length"], 1)

        if ratio > 2.0:
            findings.append(Finding(
                technique_id="RD-0201",
                technique_name="Chunk Boundary Detection (Response Variation)",
                confidence="low",
                detail=(
                    f"Response length varies {ratio:.1f}x across query types "
                    f"({short_r['response_length']}-{long_r['response_length']} chars). "
                    f"Variation may indicate different chunk coverage per query."
                ),
                evidence={
                    "length_ratio": round(ratio, 2),
                    "shortest_query_type": short_r["expected"],
                    "longest_query_type": long_r["expected"],
                },
            ))

    return findings


# --- Shared utilities ---

def _parse_retrieval_data(response_body: str) -> tuple[int, list[int]]:
    """Parse source count and chunk sizes from a RAG API response.

    Returns (source_count, chunk_sizes_in_chars).
    """
    source_count = 0
    chunk_sizes = []

    try:
        data = json.loads(response_body)

        # Count sources from structured response fields
        if "sources" in data:
            source_count = len(data["sources"])
        if "context" in data:
            source_count = max(source_count, len(data["context"]))
            # Measure actual chunk sizes from context array
            for chunk in data["context"]:
                if isinstance(chunk, str) and len(chunk) > 10:
                    chunk_sizes.append(len(chunk))
        if "documents" in data:
            source_count = max(source_count, len(data["documents"]))
            for doc in data["documents"]:
                if isinstance(doc, str) and len(doc) > 10:
                    chunk_sizes.append(len(doc))

    except (json.JSONDecodeError, TypeError):
        # Fallback: count citation patterns
        citations = re.findall(r"\[(?:source|doc|ref)\s*\d+\]", response_body, re.IGNORECASE)
        source_count = len(set(citations))

    return source_count, chunk_sizes


def _check_boundary_indicators(text: str) -> bool:
    """Check if response text contains chunk boundary indicators."""
    for pattern in BOUNDARY_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _extract_response_text(resp: httpx.Response, response_field: str | None) -> str:
    """Compatibility wrapper for the shared chat response extractor."""
    return extract_response_text(resp, response_field)


def _extract_relevance_scores(response_body: str) -> list[float]:
    """Extract relevance/similarity scores from a RAG API response."""
    scores = []
    try:
        data = json.loads(response_body)
        if "sources" in data:
            for src in data["sources"]:
                if isinstance(src, dict):
                    for key in ("relevance", "score", "similarity", "distance"):
                        if key in src:
                            val = float(src[key])
                            scores.append(val)
                            break
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return scores


# --- RD-0202: Similarity Threshold Mapping ---

# Queries ordered from highly relevant to completely irrelevant.
# The transition point where the system stops returning useful results
# reveals the similarity threshold.

THRESHOLD_PROBE_QUERIES = [
    # Level 0: Directly on-topic (should always match)
    {
        "query": "What is the company password reset policy?",
        "relevance_level": 0,
        "label": "direct_match",
    },
    # Level 1: Related topic, different angle
    {
        "query": "How do employees change their authentication credentials?",
        "relevance_level": 1,
        "label": "related_topic",
    },
    # Level 2: Same domain, tangential
    {
        "query": "What cybersecurity training is required for new hires?",
        "relevance_level": 2,
        "label": "tangential",
    },
    # Level 3: Adjacent domain
    {
        "query": "What is the company's policy on remote work equipment?",
        "relevance_level": 3,
        "label": "adjacent_domain",
    },
    # Level 4: Unrelated but plausible
    {
        "query": "What are the guidelines for booking conference rooms?",
        "relevance_level": 4,
        "label": "unrelated_plausible",
    },
    # Level 5: Completely off-topic
    {
        "query": "What is the best recipe for chocolate cake?",
        "relevance_level": 5,
        "label": "off_topic",
    },
    # Level 6: Nonsense
    {
        "query": "Flurbo snazzle wompkit drizzlefang",
        "relevance_level": 6,
        "label": "nonsense",
    },
]

# Indicators that the system failed to retrieve relevant content
NO_MATCH_INDICATORS = [
    r"(?:don't|do not|doesn't|does not) have (?:information|documents?|data) (?:about|on|regarding|for)",
    r"(?:no|not any) (?:relevant|matching|related) (?:documents?|information|results?|content)",
    r"(?:couldn't|could not|unable to) find (?:any )?(?:relevant|matching|related)",
    r"(?:outside|beyond) (?:the |my )?(?:scope|knowledge|available)",
    r"(?:not covered|not addressed|not included) in (?:the |our )?(?:documents?|knowledge|records)",
    r"(?:i'm |i am )?(?:not sure|uncertain).*(?:don't|do not) have (?:that|this) information",
    r"(?:the |our )?(?:documents?|knowledge base|records) (?:don't|do not) (?:contain|cover|include|address)",
]


def map_similarity_threshold(
    target: str,
    client: httpx.Client,
    query_field: str = "query",
    response_field: str | None = None,
) -> list[Finding]:
    """RD-0202: Map the similarity threshold by probing with decreasing relevance.

    Sends queries from highly relevant to completely off-topic and analyzes:
    1. Relevance scores (if exposed in API response)
    2. Retrieval failure indicators in response text
    3. Response quality degradation
    4. Source count changes as relevance decreases

    The transition from "useful response" to "no match" reveals the threshold.
    """
    findings: list[Finding] = []
    probe_results: list[dict] = []

    for probe in THRESHOLD_PROBE_QUERIES:
        try:
            resp = client.post(target, json={query_field: probe["query"]})
            text = _extract_response_text(resp, response_field)
            full_body = resp.text if resp.status_code == 200 else ""

            source_count, _ = _parse_retrieval_data(full_body)
            relevance_scores = _extract_relevance_scores(full_body)
            has_no_match = _check_no_match(text)

            probe_results.append({
                "query": probe["query"],
                "relevance_level": probe["relevance_level"],
                "label": probe["label"],
                "response_length": len(text),
                "source_count": source_count,
                "relevance_scores": relevance_scores,
                "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else None,
                "has_no_match": has_no_match,
                "text_preview": text[:200],
            })
        except httpx.HTTPError as e:
            logger.warning("RD-0202 threshold probe failed: %s", e)
            continue

    if len(probe_results) < 3:
        return findings

    # Analysis 1: Find the cutoff point where no-match indicators appear
    cutoff_level = None
    for pr in sorted(probe_results, key=lambda x: x["relevance_level"]):
        if pr["has_no_match"]:
            cutoff_level = pr["relevance_level"]
            break

    if cutoff_level is not None:
        cutoff_label = next(
            (p["label"] for p in probe_results if p["relevance_level"] == cutoff_level),
            "unknown",
        )
        findings.append(Finding(
            technique_id="RD-0202",
            technique_name="Similarity Threshold Mapping (Retrieval Cutoff)",
            confidence="high",
            detail=(
                f"Retrieval cutoff detected at relevance level {cutoff_level}/6 "
                f"({cutoff_label}). Queries at this level and below produce no-match "
                f"indicators. {'Tight threshold: only closely related queries match.' if cutoff_level <= 2 else 'Loose threshold: even tangential queries return results.' if cutoff_level >= 4 else 'Moderate threshold.'}"
            ),
            evidence={
                "cutoff_level": cutoff_level,
                "cutoff_label": cutoff_label,
                "levels_tested": len(probe_results),
                "scale": "0=direct_match, 6=nonsense",
            },
        ))
    else:
        # No cutoff found — system returns results for everything
        findings.append(Finding(
            technique_id="RD-0202",
            technique_name="Similarity Threshold Mapping (No Cutoff)",
            confidence="high",
            detail=(
                "No retrieval cutoff detected. The system returned content for all "
                f"{len(probe_results)} probe queries, including nonsense and off-topic. "
                "This indicates a very loose similarity threshold or no threshold at all. "
                "Poisoning attacks (R4) would be highly effective."
            ),
            evidence={
                "levels_tested": len(probe_results),
                "all_returned_content": True,
            },
        ))

    # Analysis 2: Relevance score degradation (if API exposes scores)
    scored = [pr for pr in probe_results if pr["avg_relevance"] is not None]
    if len(scored) >= 3:
        scored_sorted = sorted(scored, key=lambda x: x["relevance_level"])
        best_score = scored_sorted[0]["avg_relevance"]
        worst_score = scored_sorted[-1]["avg_relevance"]
        degradation = best_score - worst_score

        findings.append(Finding(
            technique_id="RD-0202",
            technique_name="Similarity Threshold Mapping (Score Degradation)",
            confidence="high",
            detail=(
                f"Relevance scores degrade from {best_score:.3f} (direct match) to "
                f"{worst_score:.3f} (off-topic). Total degradation: {degradation:.3f}. "
                f"{'Sharp dropoff suggests well-tuned threshold.' if degradation > 0.5 else 'Gradual degradation suggests loose matching.'}"
            ),
            evidence={
                "best_score": round(best_score, 4),
                "worst_score": round(worst_score, 4),
                "degradation": round(degradation, 4),
                "scores_by_level": [
                    {"level": s["relevance_level"], "label": s["label"], "avg_score": round(s["avg_relevance"], 4)}
                    for s in scored_sorted
                ],
            },
        ))

    # Analysis 3: Source count degradation
    with_sources = [pr for pr in probe_results if pr["source_count"] > 0]
    without_sources = [pr for pr in probe_results if pr["source_count"] == 0]

    if with_sources and without_sources:
        max_level_with = max(pr["relevance_level"] for pr in with_sources)
        min_level_without = min(pr["relevance_level"] for pr in without_sources)
        findings.append(Finding(
            technique_id="RD-0202",
            technique_name="Similarity Threshold Mapping (Source Dropoff)",
            confidence="medium",
            detail=(
                f"Sources returned up to relevance level {max_level_with}, "
                f"dropped to zero at level {min_level_without}. "
                f"The retrieval system stops returning documents between these levels."
            ),
            evidence={
                "last_level_with_sources": max_level_with,
                "first_level_without_sources": min_level_without,
                "source_counts_by_level": [
                    {"level": pr["relevance_level"], "label": pr["label"], "sources": pr["source_count"]}
                    for pr in sorted(probe_results, key=lambda x: x["relevance_level"])
                ],
            },
        ))

    return findings


def _check_no_match(text: str) -> bool:
    """Check if response text indicates no relevant results were found."""
    for pattern in NO_MATCH_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# --- RD-0203: Retrieval Count Estimation ---

RETRIEVAL_COUNT_QUERIES = [
    # Broad query — should pull max sources
    "Summarize all available documentation.",
    # Specific single-topic query
    "What is the password reset process?",
    # Very narrow query
    "What is the default temporary password?",
    # Multi-topic query (may pull from more sources)
    "What are the security policies, infrastructure details, and HR procedures?",
]


def estimate_retrieval_count(
    target: str,
    client: httpx.Client,
    query_field: str = "query",
    response_field: str | None = None,
) -> list[Finding]:
    """RD-0203: Estimate how many documents/chunks are retrieved per query.

    Sends queries of varying breadth and counts distinct sources in each
    response. The consistent count reveals the top-k setting. Variable
    counts reveal dynamic retrieval.
    """
    findings: list[Finding] = []
    counts: list[dict] = []

    for query in RETRIEVAL_COUNT_QUERIES:
        try:
            resp = client.post(target, json={query_field: query})
            full_body = resp.text if resp.status_code == 200 else ""
            source_count, _ = _parse_retrieval_data(full_body)
            counts.append({"query": query[:60], "source_count": source_count})
        except httpx.HTTPError as e:
            logger.warning("RD-0203 retrieval count failed: %s", e)
            continue

    if not counts:
        return findings

    source_vals = [c["source_count"] for c in counts if c["source_count"] > 0]
    if not source_vals:
        findings.append(Finding(
            technique_id="RD-0203",
            technique_name="Retrieval Count Estimation",
            confidence="medium",
            detail="No source counts detected in responses. The API may not expose retrieval metadata.",
            evidence={"queries_sent": len(counts), "source_counts": [c["source_count"] for c in counts]},
        ))
        return findings

    unique_counts = set(source_vals)
    most_common = max(set(source_vals), key=source_vals.count)
    is_fixed = len(unique_counts) == 1

    findings.append(Finding(
        technique_id="RD-0203",
        technique_name="Retrieval Count Estimation",
        confidence="high" if is_fixed else "medium",
        detail=(
            f"Estimated retrieval count: {most_common} documents per query. "
            f"{'Fixed top-k={} confirmed across {} queries.'.format(most_common, len(source_vals)) if is_fixed else 'Variable retrieval detected (range: {}-{}). The system may use dynamic top-k or relevance cutoff.'.format(min(source_vals), max(source_vals))}"
        ),
        evidence={
            "estimated_top_k": most_common,
            "is_fixed": is_fixed,
            "unique_counts": sorted(unique_counts),
            "counts_per_query": counts,
        },
    ))

    return findings


# --- RD-0204: Knowledge Base Scope Mapping ---

KB_SCOPE_CATEGORIES = [
    {"category": "IT/Security", "queries": [
        "What are the security policies?",
        "How do we handle security incidents?",
    ]},
    {"category": "HR/People", "queries": [
        "What is the employee onboarding process?",
        "What is the vacation and PTO policy?",
    ]},
    {"category": "Infrastructure", "queries": [
        "What databases and servers are used?",
        "What is the network architecture?",
    ]},
    {"category": "Finance", "queries": [
        "What are the budget approval procedures?",
        "How do we process expense reports?",
    ]},
    {"category": "Legal/Compliance", "queries": [
        "What are our data retention policies?",
        "What compliance frameworks do we follow?",
    ]},
    {"category": "Product/Engineering", "queries": [
        "What is the software development process?",
        "How do we handle code reviews and deployments?",
    ]},
]


def map_kb_scope(
    target: str,
    client: httpx.Client,
    query_field: str = "query",
    response_field: str | None = None,
) -> list[Finding]:
    """RD-0204: Map what domains the knowledge base covers.

    Sends queries across multiple topic categories and classifies each
    as "covered" (returns substantive content) or "gap" (no-match or
    generic response). The coverage map reveals what's in the KB and
    what's not, informing targeted exfiltration.
    """
    findings: list[Finding] = []
    covered: list[str] = []
    gaps: list[str] = []
    category_details: list[dict] = []

    for cat in KB_SCOPE_CATEGORIES:
        category = cat["category"]
        hit_count = 0
        total = len(cat["queries"])

        for query in cat["queries"]:
            try:
                resp = client.post(target, json={query_field: query})
                text = _extract_response_text(resp, response_field)
                full_body = resp.text if resp.status_code == 200 else ""
                source_count, _ = _parse_retrieval_data(full_body)

                has_content = (
                    not _check_no_match(text)
                    and len(text) > 50
                    and source_count > 0
                )
                if has_content:
                    hit_count += 1
            except httpx.HTTPError as e:
                logger.warning("RD-0204 KB scope (%s) failed: %s", category, e)
                continue

        coverage = hit_count / total if total > 0 else 0
        if coverage >= 0.5:
            covered.append(category)
        else:
            gaps.append(category)

        category_details.append({
            "category": category,
            "queries_sent": total,
            "hits": hit_count,
            "coverage": round(coverage, 2),
            "status": "covered" if coverage >= 0.5 else "gap",
        })

    if not category_details:
        return findings

    findings.append(Finding(
        technique_id="RD-0204",
        technique_name="Knowledge Base Scope Mapping",
        confidence="high" if len(category_details) >= 4 else "medium",
        detail=(
            f"KB covers {len(covered)}/{len(category_details)} scanned categories. "
            f"Covered: {', '.join(covered) if covered else 'none'}. "
            f"Gaps: {', '.join(gaps) if gaps else 'none'}. "
            f"{'Broad KB with multiple domains.' if len(covered) >= 4 else 'Narrow KB focused on specific domains.' if len(covered) <= 2 else 'Moderate KB coverage.'}"
        ),
        evidence={
            "covered_categories": covered,
            "gap_categories": gaps,
            "coverage_ratio": f"{len(covered)}/{len(category_details)}",
            "details": category_details,
        },
    ))

    return findings


# --- Debug Endpoint Discovery (sub-technique of R2) ---

DEBUG_ENDPOINT_PATHS = [
    "/debug/config",
    "/admin/stats",
    "/admin/config",
    "/debug",
    "/debug/info",
    "/admin",
    "/api/config",
    "/api/debug",
    "/.env",
    "/config",
    "/status",
    "/metrics",
    "/internal/config",
]


def scan_debug_endpoints(
    target: str,
    client: httpx.Client,
) -> list[Finding]:
    """Scan for exposed debug/admin endpoints that leak pipeline configuration.

    Many RAG deployments leave debug endpoints accessible that reveal
    embedding model, chunk strategy, retrieval parameters, and collection
    metadata. These endpoints bypass any guardrails on the chat interface.
    """
    findings: list[Finding] = []

    # Derive base URL from target (strip the chat path)
    base = target.rsplit("/", 1)[0] if "/" in target.split("//", 1)[-1] else target

    discovered: list[dict] = []
    for path in DEBUG_ENDPOINT_PATHS:
        url = base + path
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                continue
            # Must be JSON with meaningful structure to count as a debug endpoint
            try:
                data = resp.json()
                if not isinstance(data, dict) or len(data) < 2:
                    continue
                # Filter: must have keys that look like config/admin data,
                # not just a chat response
                keys_lower = [k.lower() for k in data.keys()]
                is_debug = any(
                    k in keys_lower for k in [
                        "collection", "collection_name", "embedding_model",
                        "chunk_strategy", "n_results", "document_count",
                        "sample_ids", "chroma_dir", "server_version",
                        "embedding_dimensions", "similarity_metric",
                        "guardrails_enabled", "guardrail_patterns",
                        "config", "model", "version",
                    ]
                )
                if not is_debug:
                    continue
                discovered.append({
                    "path": path,
                    "status": 200,
                    "type": "json",
                    "keys": list(data.keys()),
                    "data_preview": {k: str(v)[:100] for k, v in data.items()},
                })
            except (json.JSONDecodeError, ValueError):
                continue
        except (httpx.HTTPError, Exception):
            continue

    if discovered:
        config_endpoints = [d for d in discovered if any(k in str(d.get("keys", [])).lower() for k in ["model", "config", "embedding", "chunk", "ollama"])]
        stats_endpoints = [d for d in discovered if any(k in str(d.get("keys", [])).lower() for k in ["count", "document", "collection", "ids"])]

        findings.append(Finding(
            technique_id="RD-0201",
            technique_name="Debug Endpoint Discovery",
            confidence="high",
            detail=(
                f"Found {len(discovered)} exposed endpoint(s): "
                f"{', '.join(d['path'] for d in discovered)}. "
                f"{'Config data exposed (embedding model, chunk strategy, retrieval params).' if config_endpoints else ''} "
                f"{'Collection statistics exposed (document count, IDs, metadata).' if stats_endpoints else ''}"
            ),
            evidence={
                "endpoints_found": len(discovered),
                "endpoints": discovered,
            },
        ))

    return findings


# --- RD-0205: Embedding Model Fingerprinting ---

# Known embedding model families and their behavioral signatures.
# Each family has characteristics that affect retrieval quality
# on specific query types.

EMBEDDING_SIGNATURES = {
    "multilingual": {
        "description": "Multilingual embedding model (e.g., multilingual-e5, mxbai-embed)",
        "indicator": "handles_multilingual",
    },
    "code-aware": {
        "description": "Code-trained embedding model (e.g., code-search-ada, codebert)",
        "indicator": "handles_technical",
    },
    "general-english": {
        "description": "General English embedding model (e.g., all-MiniLM, text-embedding-ada)",
        "indicator": "english_only",
    },
    "domain-specific": {
        "description": "Domain-specific embedding model (e.g., med-bert, legal-bert)",
        "indicator": "domain_bias",
    },
}


def fingerprint_embedding_model(
    target: str,
    client: httpx.Client,
    query_field: str = "query",
    response_field: str | None = None,
) -> list[Finding]:
    """RD-0205: Fingerprint the embedding model by analyzing retrieval behavior.

    Sends domain-specific, multilingual, and edge-case queries, then
    analyzes which query types produce good retrieval vs poor retrieval.
    The pattern reveals the embedding model family.
    """
    from ragdrag.utils.payloads import load_payload

    findings: list[Finding] = []

    try:
        payload = load_payload("fingerprint_embedding_model")
    except FileNotFoundError:
        return findings

    domain_probes = payload.get("domain_probes", {})
    edge_cases = payload.get("edge_case_queries", [])

    # Test each domain
    domain_results: dict[str, dict] = {}
    for domain, queries in domain_probes.items():
        hits = 0
        total_relevance = 0.0
        tested = 0

        for query in queries:
            try:
                resp = client.post(target, json={query_field: query})
                text = _extract_response_text(resp, response_field)
                full_body = resp.text if resp.status_code == 200 else ""
                source_count, _ = _parse_retrieval_data(full_body)
                scores = _extract_relevance_scores(full_body)

                has_content = not _check_no_match(text) and len(text) > 50
                if has_content:
                    hits += 1
                if scores:
                    total_relevance += max(scores)
                tested += 1
            except httpx.HTTPError as e:
                logger.warning("RD-0205 embedding fingerprint (%s) failed: %s", domain, e)
                continue

        if tested > 0:
            domain_results[domain] = {
                "hits": hits,
                "tested": tested,
                "hit_rate": round(hits / tested, 2),
                "avg_best_relevance": round(total_relevance / tested, 4) if total_relevance > 0 else None,
            }

    # Test edge cases
    edge_results: list[dict] = []
    for query in edge_cases:
        try:
            resp = client.post(target, json={query_field: query})
            text = _extract_response_text(resp, response_field)
            full_body = resp.text if resp.status_code == 200 else ""
            source_count, _ = _parse_retrieval_data(full_body)

            edge_results.append({
                "query": query,
                "returned_content": not _check_no_match(text) and source_count > 0,
                "source_count": source_count,
                "response_length": len(text),
            })
        except httpx.HTTPError as e:
            logger.warning("RD-0205 edge case probe failed: %s", e)
            continue

    if not domain_results:
        return findings

    # Analyze: determine embedding model characteristics
    characteristics: list[str] = []

    # Check multilingual support
    multilingual = domain_results.get("multilingual", {})
    if multilingual.get("hit_rate", 0) > 0:
        characteristics.append("multilingual")

    # Check technical/code handling
    technical = domain_results.get("technical", {})
    if technical.get("hit_rate", 0) >= 0.5:
        characteristics.append("code-aware")

    # Check if edge cases return results (loose embedding = returns garbage matches)
    edge_hits = sum(1 for e in edge_results if e["returned_content"])
    if edge_hits > len(edge_results) // 2:
        characteristics.append("loose-matching")

    # Determine domain bias (one domain scores much higher than others)
    hit_rates = {d: r["hit_rate"] for d, r in domain_results.items()}
    if hit_rates:
        best_domain = max(hit_rates, key=hit_rates.get)
        worst_domain = min(hit_rates, key=hit_rates.get)
        if hit_rates[best_domain] - hit_rates[worst_domain] > 0.5:
            characteristics.append(f"biased-{best_domain}")

    # Classify the model family
    if "multilingual" in characteristics:
        model_family = "multilingual"
    elif any(c.startswith("biased-") for c in characteristics):
        model_family = "domain-specific"
    elif "code-aware" in characteristics:
        model_family = "code-aware"
    else:
        model_family = "general-english"

    sig = EMBEDDING_SIGNATURES.get(model_family, {})

    findings.append(Finding(
        technique_id="RD-0205",
        technique_name="Embedding Model Fingerprinting",
        confidence="high" if len(domain_results) >= 3 else "medium",
        detail=(
            f"Embedding model profile: {sig.get('description', model_family)}. "
            f"Characteristics: {', '.join(characteristics) if characteristics else 'standard English'}. "
            f"Domain hit rates: {', '.join(d + '=' + str(r['hit_rate']) for d, r in domain_results.items())}."
        ),
        evidence={
            "model_family": model_family,
            "characteristics": characteristics,
            "domain_results": domain_results,
            "edge_case_results": edge_results,
        },
    ))

    return findings


# --- Orchestrator ---

def run_probe(
    target: str,
    client: httpx.Client,
    depth: str = "quick",
    query_field: str = "query",
    response_field: str | None = None,
) -> ProbeResult:
    """Run R2 probe techniques against a target.

    Args:
        target: URL of the chat/query endpoint.
        client: httpx.Client instance.
        depth: "quick" for RD-0201/0203 only, "full" for all techniques.
        query_field: JSON field name for the query.
        response_field: JSON field to read the response from.

    Returns:
        ProbeResult with all findings.
    """
    result = ProbeResult(target=target)

    # RD-0201: Chunk Boundary Detection (always runs)
    chunk_findings = detect_chunk_boundaries(
        target, client, query_field=query_field, response_field=response_field,
    )
    result.findings.extend(chunk_findings)

    # Extract estimates from findings into result fields
    for f in chunk_findings:
        if "estimated_avg_chunk_chars" in f.evidence:
            result.chunk_size_estimate = f.evidence["estimated_avg_chunk_chars"]
        if "most_common" in f.evidence:
            result.retrieval_count = f.evidence["most_common"]

    # RD-0202: Similarity Threshold Mapping (always runs)
    threshold_findings = map_similarity_threshold(
        target, client, query_field=query_field, response_field=response_field,
    )
    result.findings.extend(threshold_findings)

    # Extract threshold estimate from findings
    for f in threshold_findings:
        if "best_score" in f.evidence and "worst_score" in f.evidence:
            result.similarity_threshold = f.evidence["worst_score"]

    # RD-0203: Retrieval Count Estimation (always runs)
    count_findings = estimate_retrieval_count(
        target, client, query_field=query_field, response_field=response_field,
    )
    result.findings.extend(count_findings)

    # Override retrieval_count with the dedicated estimate if available
    for f in count_findings:
        if "estimated_top_k" in f.evidence:
            result.retrieval_count = f.evidence["estimated_top_k"]

    # RD-0204: Knowledge Base Scope Mapping (full depth only — sends many queries)
    if depth == "full":
        scope_findings = map_kb_scope(
            target, client, query_field=query_field, response_field=response_field,
        )
        result.findings.extend(scope_findings)

        for f in scope_findings:
            if "covered_categories" in f.evidence:
                result.kb_domains = f.evidence["covered_categories"]

    # Debug endpoint scan (always runs — fast, no LLM calls)
    debug_findings = scan_debug_endpoints(target, client)
    result.findings.extend(debug_findings)

    # RD-0205: Embedding Model Fingerprinting (full depth only)
    if depth == "full":
        embedding_findings = fingerprint_embedding_model(
            target, client, query_field=query_field, response_field=response_field,
        )
        result.findings.extend(embedding_findings)

        for f in embedding_findings:
            if "model_family" in f.evidence:
                result.embedding_model = f.evidence["model_family"]

    return result
