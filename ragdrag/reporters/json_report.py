"""Schema 1.0 report serialization and the direct-phase compatibility adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from functools import lru_cache
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import click

from ragdrag import __version__
from ragdrag.engine.models import (
    CapabilityResult, CapabilityStatus, CleanupState, EvidenceItem, EvidenceState,
    ExitCode, Finding, ImpactLevel, MutationRecord, OutcomeCode, RunResult,
)
from ragdrag.engine.phases import (
    EngagementOutcome, PHASE_METADATA, _SAFE_EVIDENCE_FIELDS, _SAFE_PAYLOAD_FIELDS,
    normalize_finding,
)
from ragdrag.engine.profile import TargetProfile, canonical_origin
from ragdrag.engine.redaction import REDACTED, _is_sensitive_key, redact

_ERROR = "invalid report data"
_STATUS = {"completed", "partial", "failed", "interrupted"}
_SEVERITY = {"critical", "high", "medium", "low", "info"}
_CONFIDENCE = {"high", "medium", "low"}
_IMPLEMENTATION = {"catalogued", "implemented", "validated", "experimental"}
_DYNAMIC_KEYS = _SAFE_EVIDENCE_FIELDS | _SAFE_PAYLOAD_FIELDS | {
    "phase", "technique_id", "finding", "severity", "confidence", "method", "sensitivity",
    "status_code", "request_count", "response_bytes",
}
_SAFE_ERRORS = {
    "legacy phase request failed", "legacy phase conversion failed",
    "response format is unsupported", "authentication required",
}
_SAFE_EXCEPTION_NAMES = {
    "ValueError", "TypeError", "RuntimeError", "OSError", "TimeoutError",
    "ConnectionError", "HTTPStatusError", "ConnectError", "ReadError",
    "ReadTimeout", "ConnectTimeout", "RequestBudgetExceeded",
    "ResponseTooLarge", "ChatResponseError", "KeyboardInterrupt",
}
_SAFE_PATH_SEGMENTS = {
    "api", "v1", "v2", "v3", "chat", "query", "search", "generate",
    "completions", "messages", "ask", "rag", "health",
}
_REFERENCE_PATTERN = re.compile(
    r"(?:preflight|phase\.r[1-6](?:\.\d+)?|RD-\d{4}|(?:run|fi|ev|mu|trial)-\d+|"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_CONFIG_FIELDS = {
    "query", "prompt", "message", "input", "text", "question",
    "response", "answer", "output", "result", "content",
}


def _invalid() -> None:
    raise ValueError(_ERROR)


def _string(value: object) -> str:
    if type(value) is not str:
        _invalid()
    if len(value) > 1_000_000:
        _invalid()
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        _invalid()
    if len(encoded) > 1_000_000:
        _invalid()
    return value


def _digest(value: object) -> str | None:
    if value is None:
        return None
    text = _string(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        _invalid()
    return text


def _nonnegative(value: object) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        _invalid()
    return value


def _enum(value: object, enum_type: type) -> str:
    if type(value) is not enum_type:
        _invalid()
    return value.value


def _strings(value: object) -> list[str]:
    if type(value) is not list and type(value) is not tuple:
        _invalid()
    return [_string(item) for item in value]


def _reference(value: object) -> str:
    text = _string(value)
    if _REFERENCE_PATTERN.fullmatch(text):
        return text
    return "ref-" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _references(value: object) -> list[str]:
    if type(value) is not list and type(value) is not tuple:
        _invalid()
    return [_reference(item) for item in value]


def _config_field(value: object) -> str:
    text = _string(value)
    return text if text in _CONFIG_FIELDS else REDACTED


def _timestamp(value: object) -> str:
    text = _string(value)
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return REDACTED
    return text


def _controls(value: object) -> list[str]:
    names = _strings(value)
    return [name if name in {"baseline", "negative-control", "cleanup-verification"}
            or re.fullmatch(r"tests/test_[a-z0-9_]+\.py", name) else REDACTED
            for name in names]


def _dynamic_string(key: str, value: str) -> str:
    if key == "phase" and re.fullmatch(r"R[1-6]", value):
        return value
    if key == "technique_id" and re.fullmatch(r"RD-\d{4}", value):
        return value
    if key == "method" and value in {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}:
        return value
    if key == "severity" and value in _SEVERITY:
        return value
    if key == "confidence" and value in _CONFIDENCE:
        return value
    if key == "state_mechanism" and value in {"history", "session", "cookie"}:
        return value
    if key == "sensitivity" and value in {"credential", "internal_doc", "system-prompt", "secret", "evasion", "general"}:
        return value
    return REDACTED


def _json_value(value: object, active: set[int] | None = None, key: str = "") -> Any:
    """Detach JSON data without invoking custom conversion or key hooks."""
    if type(value) is str:
        return _dynamic_string(key, _string(value))
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if value < -(2**63) or value > 2**63 - 1:
            _invalid()
        return value
    if type(value) is float:
        if not isfinite(value):
            _invalid()
        return value
    if type(value) is not dict and type(value) is not list and type(value) is not tuple:
        _invalid()
    if active is None:
        active = set()
    identity = id(value)
    if identity in active or len(active) >= 64:
        _invalid()
    active.add(identity)
    try:
        if type(value) is dict:
            result = {}
            for item_key, item in value.items():
                original = _string(item_key)
                safe_key = original if original in _DYNAMIC_KEYS else "<redacted-key>"
                result[safe_key] = (
                    REDACTED if _is_sensitive_key(original)
                    else _json_value(item, active, original)
                )
            return result
        return [_json_value(item, active, key) for item in value]
    finally:
        active.remove(identity)


def _json_object(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        _invalid()
    return _json_value(value)


def _safe_free_text(value: str) -> str:
    if value in {"", REDACTED, "legacy phase heuristic; inspect referenced evidence"}:
        return value
    if re.fullmatch(r"R[1-6] completed with \d+ finding\(s\)", value):
        return value
    return REDACTED


def _safe_error(value: str) -> str:
    if value in _SAFE_ERRORS:
        return value
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Interrupt))(?::.*)?", value)
    return match.group(1) if match and match.group(1) in _SAFE_EXCEPTION_NAMES else REDACTED


def _finding_dict(item: Finding) -> dict[str, Any]:
    if type(item) is not Finding:
        _invalid()
    severity = _string(item.severity)
    confidence = _string(item.confidence)
    if severity not in _SEVERITY or confidence not in _CONFIDENCE:
        _invalid()
    technique_id = _string(item.technique_id)
    registered = next((metadata.title for metadata in PHASE_METADATA.values()
                       if technique_id in metadata.technique_ids), None)
    return {
        "finding_id": _reference(item.finding_id),
        "technique_id": _reference(technique_id),
        "technique_name": registered or REDACTED,
        "severity": severity,
        "confidence": confidence,
        "evidence_state": _enum(item.evidence_state, EvidenceState),
        "confidence_basis": _safe_free_text(_string(item.confidence_basis)),
        "detail": _safe_free_text(_string(item.detail)),
        "affected_component": _safe_free_text(_string(item.affected_component)),
        "evidence": _json_object(item.evidence),
        "remediation": _safe_free_text(_string(item.remediation)),
    }


def _capability_dict(item: CapabilityResult) -> dict[str, Any]:
    if type(item) is not CapabilityResult or type(item.findings) is not list:
        _invalid()
    for finding in item.findings:
        if type(finding) is not Finding:
            _invalid()
    if type(item.outcomes) is not list:
        _invalid()
    return {
        "capability_id": _reference(item.capability_id),
        "technique_ids": _references(item.technique_ids),
        "status": _enum(item.status, CapabilityStatus),
        "impact": _enum(item.impact, ImpactLevel),
        "started_at": _timestamp(item.started_at),
        "ended_at": _timestamp(item.ended_at),
        "trials": _nonnegative(item.trials),
        "controls": _controls(item.controls),
        "finding_ids": [_reference(finding.finding_id) for finding in item.findings],
        "evidence_ids": _references(item.evidence_ids),
        "outcomes": [_enum(code, OutcomeCode) for code in item.outcomes],
        "errors": [_safe_error(text) for text in _strings(item.errors)],
        "mutation_ids": _references(item.mutation_ids),
        "cleanup_state": _enum(item.cleanup_state, CleanupState),
        "summary": _safe_free_text(_string(item.summary)),
        "payload": _json_object(item.payload),
    }


def _evidence_dict(item: EvidenceItem) -> dict[str, Any]:
    if type(item) is not EvidenceItem:
        _invalid()
    return {
        "evidence_id": _reference(item.evidence_id),
        "capability_id": _reference(item.capability_id),
        "trial_id": _reference(item.trial_id),
        "timestamp": _timestamp(item.timestamp),
        "state": _enum(item.state, EvidenceState),
        "confidence_basis": _safe_free_text(_string(item.confidence_basis)),
        "request_summary": _json_object(item.request_summary),
        "response_summary": _json_object(item.response_summary),
        "artifact_digest": _digest(item.artifact_digest),
    }


def _mutation_dict(item: MutationRecord) -> dict[str, Any]:
    if type(item) is not MutationRecord:
        _invalid()
    return {
        "mutation_id": _reference(item.mutation_id),
        "capability_id": _reference(item.capability_id),
        "target_scope": _safe_free_text(_string(item.target_scope)),
        "operation": _safe_free_text(_string(item.operation)),
        "object_id": _safe_free_text(_string(item.object_id)),
        "cleanup_method": _safe_free_text(_string(item.cleanup_method)),
        "state": _enum(item.state, CleanupState),
        "cleanup_attempts": _nonnegative(item.cleanup_attempts),
        "evidence_ids": _references(item.evidence_ids),
    }


def _implementation_rows() -> list[dict[str, Any]]:
    rows = []
    for phase, metadata in PHASE_METADATA.items():
        maturity = _string(metadata.maturity)
        rows.append({
            "phase": _string(phase),
            "capability": _string(metadata.capability_id),
            "technique_ids": _strings(metadata.technique_ids),
            "impact": _enum(metadata.impact, ImpactLevel),
            "maturity": maturity,
            "status": maturity if maturity in _IMPLEMENTATION else "implemented",
            "release": __version__,
        })
    return rows


def _summary(capabilities: list[dict[str, Any]], findings: list[dict[str, Any]],
             mutations: list[dict[str, Any]]) -> dict[str, int]:
    statuses = [item["status"] for item in capabilities]
    states = [item["evidence_state"] for item in findings]
    return {
        "total_capabilities": len(capabilities),
        "completed": statuses.count("completed"),
        "partial": statuses.count("partial"),
        "skipped": statuses.count("skipped"),
        "blocked": statuses.count("blocked"),
        "failed": statuses.count("failed"),
        "total_findings": len(findings),
        "observed": states.count("observed"),
        "inferred": states.count("inferred"),
        "validated": states.count("validated"),
        "unresolved_mutations": sum(item["state"] in ("active", "unresolved", "unknown")
                                    for item in mutations),
    }


def _safe_target(url: str) -> str:
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        _invalid()
    if type(host) is not str or parts.scheme not in ("http", "https"):
        _invalid()
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    path = "/".join(
        part if not part or part.lower() in _SAFE_PATH_SEGMENTS else "redacted"
        for part in parts.path.split("/")
    )
    return urlunsplit((parts.scheme, netloc, path, "", ""))


def _ensure_json_types(value: object, active: set[int] | None = None) -> None:
    if value is None or type(value) is bool:
        return
    if type(value) is str:
        _string(value)
        return
    if type(value) is int:
        if value < -(2**63) or value > 2**63 - 1:
            _invalid()
        return
    if type(value) is float:
        if not isfinite(value):
            _invalid()
        return
    if type(value) is not dict and type(value) is not list:
        _invalid()
    if active is None:
        active = set()
    identity = id(value)
    if identity in active or len(active) >= 64:
        _invalid()
    active.add(identity)
    try:
        if type(value) is dict:
            for key, item in value.items():
                _string(key)
                _ensure_json_types(item, active)
        else:
            for item in value:
                _ensure_json_types(item, active)
    finally:
        active.remove(identity)


@lru_cache(maxsize=1)
def _report_schema() -> dict[str, Any]:
    try:
        path = Path(__file__).parent / "schemas" / "report-v1.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _invalid()
    if type(schema) is not dict:
        _invalid()
    return schema


def _schema_type_matches(value: Any, name: str) -> bool:
    return {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "integer": type(value) is int,
        "boolean": type(value) is bool,
        "null": value is None,
    }.get(name, False)


def _validate_schema_node(value: Any, spec: dict[str, Any], root: dict[str, Any]) -> None:
    if "$ref" in spec:
        reference = spec["$ref"]
        if type(reference) is not str or not reference.startswith("#/$defs/"):
            _invalid()
        definition = root.get("$defs", {}).get(reference[len("#/$defs/"):])
        if type(definition) is not dict:
            _invalid()
        _validate_schema_node(value, definition, root)
        return
    expected = spec.get("type")
    if expected is not None:
        choices = expected if type(expected) is list else [expected]
        if not any(_schema_type_matches(value, name) for name in choices):
            _invalid()
    if "const" in spec and value != spec["const"]:
        _invalid()
    if "enum" in spec and value not in spec["enum"]:
        _invalid()
    if type(value) is dict:
        required = spec.get("required", [])
        properties = spec.get("properties", {})
        if any(key not in value for key in required):
            _invalid()
        if spec.get("additionalProperties") is False and any(key not in properties for key in value):
            _invalid()
        for key, item in value.items():
            if key in properties:
                _validate_schema_node(item, properties[key], root)
    elif type(value) is list:
        if spec.get("uniqueItems") and any(
            value[index] == value[previous]
            for index in range(len(value)) for previous in range(index)
        ):
            _invalid()
        item_spec = spec.get("items")
        if type(item_spec) is dict:
            for item in value:
                _validate_schema_node(item, item_spec, root)
    elif type(value) is int:
        if ("minimum" in spec and value < spec["minimum"]) or (
            "maximum" in spec and value > spec["maximum"]
        ):
            _invalid()
    elif type(value) is str and "pattern" in spec:
        if re.fullmatch(spec["pattern"], value) is None:
            _invalid()


def _validate_report(report: dict[str, Any]) -> str:
    _ensure_json_types(report)
    _validate_schema_node(report, _report_schema(), _report_schema())
    try:
        serialized = json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
        serialized.encode("utf-8")
    except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError):
        _invalid()
    return serialized


def _configured_secrets(profile: TargetProfile) -> tuple[str, ...]:
    """Read only exact built-in credential strings; never reflect foreign objects."""
    scoped = profile.headers_by_origin
    if type(scoped) is not MappingProxyType and type(scoped) is not dict:
        _invalid()
    secrets: set[str] = set()
    for headers in scoped.values():
        if type(headers) is not MappingProxyType and type(headers) is not dict:
            _invalid()
        for name, value in headers.items():
            header = _string(name).lower()
            text = _string(value)
            if text:
                secrets.add(text)
            if header == "authorization" and text.lower().startswith("bearer "):
                token = text[7:].strip()
                if token:
                    secrets.add(token)
            if header == "cookie":
                for part in text.split(";"):
                    if "=" in part:
                        token = part.split("=", 1)[1].strip()
                        if token:
                            secrets.add(token)
    return tuple(sorted(secrets, key=len, reverse=True))


def _has_secret(value: str, secrets: tuple[str, ...]) -> bool:
    return any(secret in value for secret in secrets)


def _surrogate(value: str, domain: str, secrets: tuple[str, ...], *,
               prefix: str = "") -> str:
    if not _has_secret(value, secrets):
        return value
    if _has_secret(prefix, secrets):
        prefix = ""
    for counter in range(4096):
        source = f"ragdrag-report-v1:{domain}:{counter}:{value}".encode("utf-8")
        candidate = prefix + hashlib.sha256(source).hexdigest()
        if not _has_secret(candidate, secrets):
            return candidate
    _invalid()


def _untrusted_scalar(value: Any, domain: str, secrets: tuple[str, ...]) -> Any:
    if type(value) is str:
        return _surrogate(value, domain, secrets, prefix="safe-")
    if value is None or type(value) is bool or type(value) is int or type(value) is float:
        spelling = json.dumps(value, allow_nan=False)
        if _has_secret(spelling, secrets):
            return _surrogate(spelling, domain, secrets, prefix="safe-")
    return value


def _untrusted_nonnegative(value: int, secrets: tuple[str, ...]) -> int:
    if not _has_secret(str(value), secrets):
        return value
    for candidate in range(4096):
        if not _has_secret(str(candidate), secrets):
            return candidate
    _invalid()


def _untrusted_json(value: Any, secrets: tuple[str, ...]) -> Any:
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            safe_key = _untrusted_scalar(key, "dynamic-key", secrets)
            result[safe_key] = _untrusted_json(item, secrets)
        return result
    if type(value) is list:
        return [_untrusted_json(item, secrets) for item in value]
    return _untrusted_scalar(value, "dynamic-value", secrets)


def _safe_url_secret(value: str, secrets: tuple[str, ...], *, origin: bool = False) -> str:
    try:
        parts = urlsplit(value)
        host = parts.hostname
        port = parts.port
    except ValueError:
        _invalid()
    if type(host) is not str or parts.scheme not in ("http", "https"):
        _invalid()
    if _has_secret(host, secrets):
        safe_host = _surrogate(host, "target-host", secrets)
        host = next((candidate for candidate in (
            "h-" + safe_host[:24] + ".invalid",
            "x-" + safe_host[:24] + ".example",
            safe_host[:24] + ".test",
        ) if not _has_secret(candidate, secrets)), None)
        if host is None:
            _invalid()
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    path = "" if origin else "/".join(
        _surrogate(segment, "target-path", secrets, prefix="p-")
        for segment in parts.path.split("/")
    )
    safe = urlunsplit((parts.scheme, netloc, path, "", ""))
    if _has_secret(safe, secrets):
        _invalid()
    return safe


def _sanitize_untrusted_report(report: dict[str, Any], profile: TargetProfile) -> None:
    """Withhold configured credentials without changing canonical report metadata."""
    secrets = _configured_secrets(profile)
    if not secrets:
        return
    run = report["run"]
    run["run_id"] = _surrogate(run["run_id"], "run-id", secrets, prefix="ref-")
    for field in ("started_at", "ended_at"):
        run[field] = _untrusted_scalar(run[field], field, secrets)
    target = report["target"]
    target["url"] = _safe_url_secret(target["url"], secrets)
    target["approved_origins"] = sorted({
        _safe_url_secret(origin, secrets, origin=True)
        for origin in target["approved_origins"]
    })
    try:
        primary_origin = canonical_origin(target["url"])
    except ValueError:
        _invalid()
    if primary_origin not in target["approved_origins"]:
        _invalid()
    for field in ("query_field", "response_field"):
        if target[field] is not None:
            target[field] = _untrusted_scalar(target[field], "target-field", secrets)
    for capability in report["capabilities"]:
        capability["capability_id"] = _surrogate(capability["capability_id"], "capability-id", secrets, prefix="ref-")
        for field, domain in (("technique_ids", "technique-id"), ("finding_ids", "finding-id"),
                              ("evidence_ids", "evidence-id"), ("mutation_ids", "mutation-id")):
            capability[field] = [_surrogate(item, domain, secrets, prefix="ref-") for item in capability[field]]
        for field in ("started_at", "ended_at", "summary"):
            capability[field] = _untrusted_scalar(capability[field], field, secrets)
        capability["trials"] = _untrusted_nonnegative(capability["trials"], secrets)
        for field in ("controls", "errors"):
            capability[field] = [_untrusted_scalar(item, field, secrets) for item in capability[field]]
        capability["payload"] = _untrusted_json(capability["payload"], secrets)
    for finding in report["findings"]:
        # Classify before credential sanitization can replace the registered ID.
        if not any(finding["technique_id"] in metadata.technique_ids
                   for metadata in PHASE_METADATA.values()):
            finding["technique_name"] = _untrusted_scalar(finding["technique_name"], "technique-name", secrets)
        finding["finding_id"] = _surrogate(finding["finding_id"], "finding-id", secrets, prefix="ref-")
        finding["technique_id"] = _surrogate(finding["technique_id"], "technique-id", secrets, prefix="ref-")
        for field in ("confidence_basis", "detail", "affected_component", "remediation"):
            finding[field] = _untrusted_scalar(finding[field], field, secrets)
        finding["evidence"] = _untrusted_json(finding["evidence"], secrets)
    for evidence in report["evidence"]:
        for field, domain in (("evidence_id", "evidence-id"), ("capability_id", "capability-id"),
                              ("trial_id", "trial-id")):
            evidence[field] = _surrogate(evidence[field], domain, secrets, prefix="ref-")
        for field in ("timestamp", "confidence_basis"):
            evidence[field] = _untrusted_scalar(evidence[field], field, secrets)
        for field in ("request_summary", "response_summary"):
            evidence[field] = _untrusted_json(evidence[field], secrets)
        if evidence["artifact_digest"] is not None:
            evidence["artifact_digest"] = _surrogate(evidence["artifact_digest"], "artifact-digest", secrets)
    for mutation in report["mutations"]:
        mutation["mutation_id"] = _surrogate(mutation["mutation_id"], "mutation-id", secrets, prefix="ref-")
        mutation["capability_id"] = _surrogate(mutation["capability_id"], "capability-id", secrets, prefix="ref-")
        mutation["evidence_ids"] = [_surrogate(item, "evidence-id", secrets, prefix="ref-")
                                    for item in mutation["evidence_ids"]]
        for field in ("target_scope", "operation", "object_id", "cleanup_method"):
            mutation[field] = _untrusted_scalar(mutation[field], field, secrets)
        mutation["cleanup_attempts"] = _untrusted_nonnegative(mutation["cleanup_attempts"], secrets)


def generate_run_report(
    outcome: EngagementOutcome, profile: TargetProfile, output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build one detached, redacted report from a completed or interrupted engagement."""
    if type(outcome) is not EngagementOutcome or type(profile) is not TargetProfile:
        _invalid()
    run = outcome.run
    if type(run) is not RunResult or type(run.capabilities) is not list:
        _invalid()
    if type(outcome.evidence) is not list or type(outcome.mutations) is not list:
        _invalid()
    if type(run.status) is not str or run.status not in _STATUS:
        _invalid()
    if type(run.exit_code) is not ExitCode:
        _invalid()
    if type(profile.approved_origins) is not frozenset:
        _invalid()
    capabilities = [_capability_dict(item) for item in run.capabilities]
    findings = [_finding_dict(finding) for item in run.capabilities for finding in item.findings]
    evidence = [_evidence_dict(item) for item in outcome.evidence]
    mutations = [_mutation_dict(item) for item in outcome.mutations]
    report = {
        "schema_version": "1.0",
        "tool": {"name": "ragdrag", "version": __version__},
        "run": {"run_id": _reference(run.run_id), "started_at": _timestamp(run.started_at),
                "ended_at": _timestamp(run.ended_at), "status": run.status,
                "exit_code": int(run.exit_code)},
        "target": {"url": _safe_target(_string(profile.target_url)),
                   "approved_origins": sorted(_strings(list(profile.approved_origins))),
                   "query_field": _config_field(profile.query_field),
                   "response_field": None if profile.response_field is None else _config_field(profile.response_field)},
        "safety": {"impact_ceiling": _enum(profile.impact_ceiling, ImpactLevel),
                   "write_enabled": profile.impact_ceiling is ImpactLevel.MUTATING},
        "summary": _summary(capabilities, findings, mutations),
        "capabilities": capabilities, "findings": findings, "evidence": evidence,
        "mutations": mutations, "sensitive_artifacts": [],
        "implementation_status": _implementation_rows(),
    }
    report = redact(report)
    _validate_report(report)
    _sanitize_untrusted_report(report, profile)
    serialized = _validate_report(report)
    if output_path is not None:
        if type(output_path) is str:
            path_text = _string(output_path)
        elif type(output_path) is type(Path()):
            path_text = _string(str(output_path))
        else:
            _invalid()
        path = Path(path_text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized, encoding="utf-8")
        except OSError:
            raise OSError("report output failed") from None
    return report


def write_sensitive_artifact(path: str | Path, data: bytes) -> dict[str, Any]:
    """Atomically replace an owner-only artifact through no-follow directories."""
    if type(data) is not bytes:
        _invalid()
    if type(path) is not str and type(path) is not type(Path()):
        _invalid()
    destination = Path(path)
    parts = destination.parts
    if not parts or parts[-1] in ("", ".", "..") or ".." in parts:
        raise OSError("sensitive artifact write failed")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("sensitive artifact write failed")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_fd: int | None = None
    temp_fd: int | None = None
    temp_name: str | None = None
    temp_inode: tuple[int, int] | None = None
    replaced = False
    try:
        directory_fd = os.open("/" if destination.is_absolute() else ".", directory_flags)
        ancestors = parts[1:-1] if destination.is_absolute() else parts[:-1]
        for component in ancestors:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            previous_fd = directory_fd
            directory_fd = next_fd
            os.close(previous_fd)
        leaf = parts[-1]
        try:
            old = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            old = None
        if old is not None and (not stat.S_ISREG(old.st_mode) or old.st_uid != os.getuid()):
            raise OSError("unsafe artifact target")
        temp_name = f".ragdrag-{uuid4().hex}.tmp"
        temp_fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        created = os.fstat(temp_fd)
        temp_inode = (created.st_dev, created.st_ino)
        os.fchmod(temp_fd, 0o600)
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.write(temp_fd, view[written:])
            if count <= 0:
                raise OSError("incomplete artifact write")
            written += count
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = None
        os.replace(temp_name, leaf, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        replaced = True
        os.fsync(directory_fd)
    except Exception:
        raise OSError("sensitive artifact write failed") from None
    finally:
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_inode is not None and temp_name is not None and not replaced and directory_fd is not None:
            try:
                current = os.stat(temp_name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == temp_inode:
                    os.unlink(temp_name, dir_fd=directory_fd)
            except OSError:
                pass
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass
    return {"path": str(destination), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def generate_report(result: object, output_path: str | Path | None = None) -> dict[str, Any]:
    """Wrap an already executed legacy phase result; no phase methods are called."""
    from ragdrag.core.evade import EvadeResult
    from ragdrag.core.exfiltrate import ExfiltrateResult
    from ragdrag.core.fingerprint import FingerprintResult
    from ragdrag.core.hijack import HijackResult
    from ragdrag.core.poison import PoisonResult
    from ragdrag.core.probe import ProbeResult

    try:
        result_type = type(result)
        if type(result_type) is not type:
            _invalid()
        target = _string(result.target)
        originals = result.findings
        if type(originals) is not list:
            _invalid()
        collections = [originals]
        if result_type is ExfiltrateResult:
            bypass = result.guardrail_bypass_findings
            if type(bypass) is not list:
                _invalid()
            collections.append(bypass)
        seen: set[int] = set()
        source_findings = []
        for collection in collections:
            for item in collection:
                if id(item) not in seen:
                    source_findings.append(item)
                    seen.add(id(item))
        findings = [normalize_finding(item) for item in source_findings]
        profile = TargetProfile.from_cli(target)
        phase = None
        for known, name in (
            (FingerprintResult, "R1"), (ProbeResult, "R2"),
            (ExfiltrateResult, "R3"), (PoisonResult, "R4"),
            (HijackResult, "R5"), (EvadeResult, "R6"),
        ):
            if result_type is known:
                phase = name
                break
        if phase is None:
            phase = next((name for name, metadata in PHASE_METADATA.items()
                          if any(f.technique_id in metadata.technique_ids for f in findings)), "R1")
        payload = {}
        if any(result_type is known for known in (
            FingerprintResult, ProbeResult, ExfiltrateResult,
            PoisonResult, HijackResult, EvadeResult,
        )):
            fields = vars(result)
            if type(fields) is not dict:
                _invalid()
            for key in _SAFE_PAYLOAD_FIELDS:
                if key in fields:
                    value = fields[key]
                    if (type(value) is not bool and type(value) is not int
                            and type(value) is not float and value is not None):
                        _invalid()
                    payload[key] = value
        metadata = PHASE_METADATA[phase]
        now = datetime.now(timezone.utc).isoformat()
        capability = CapabilityResult(
            capability_id=metadata.capability_id, technique_ids=metadata.technique_ids,
            status=CapabilityStatus.COMPLETED, impact=metadata.impact,
            started_at=now, ended_at=now, findings=findings, payload=payload,
        )
        run = RunResult(str(uuid4()), target, now, now, [capability],
                        ExitCode.FINDINGS if findings else ExitCode.CLEAN)
    except Exception:
        raise ValueError(_ERROR) from None
    return generate_run_report(EngagementOutcome(run, [], []), profile, output_path)


def format_summary(report: dict, *, color: bool = True) -> str:
    """Render safe aggregate status and finding labels from a report-v1 envelope."""
    summary = report["summary"]
    rows = [
        f"Target: {report['target']['url']}",
        f"Capabilities: {summary['total_capabilities']} ({summary['completed']} completed, "
        f"{summary['partial']} partial, {summary['blocked']} blocked, {summary['failed']} failed)",
        f"Findings: {summary['total_findings']} ({summary['observed']} observed, "
        f"{summary['inferred']} inferred, {summary['validated']} validated)",
        f"Unresolved cleanup: {summary['unresolved_mutations']}",
    ]
    for item in report["capabilities"]:
        label = f"{item['capability_id']}: {item['status']}"
        if item["outcomes"]:
            label += " — " + ", ".join(item["outcomes"])
        rows.append(label)
    for item in report["findings"]:
        label = f"[{item['confidence'].upper()}] {item['technique_id']}: {item['technique_name']}"
        if color:
            label = click.style(label, fg={"high": "red", "medium": "yellow", "low": "green"}.get(
                item["confidence"], "white"))
        rows.append(label)
        if item["detail"] != REDACTED:
            rows.append(item["detail"])
    return "\n".join(rows)
