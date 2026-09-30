"""Evidence keeps useful metadata without retaining protected content."""

from copy import deepcopy
from datetime import datetime, timezone
from uuid import UUID

import pytest

from ragdrag.engine.evidence import EvidenceStore
from ragdrag.engine.models import EvidenceState
from ragdrag.engine.redaction import REDACTED, redact


class _OpaqueKey:
    def __init__(self):
        self.format_calls = []

    def __str__(self):
        self.format_calls.append("str")
        return "harmless"

    def __repr__(self):
        self.format_calls.append("repr")
        return "private-marker"


class _MarkedString(str):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.format_calls = []
        return instance

    def __str__(self):
        self.format_calls.append("str")
        return "private-marker"

    def __repr__(self):
        self.format_calls.append("repr")
        return "private-marker"


def test_redact_removes_nested_secret_values_without_changing_input():
    value = {
        "headers": {"Authorization": "Bearer abc", "X-Api-Key": "key"},
        "items": [{"cookie": "sid=secret"}, {"safe": "value"}],
        "metadata": ("first", {"VECTOR": [0.1, 0.2]}),
    }
    original = deepcopy(value)

    result = redact(value)

    assert result["headers"] == {"Authorization": REDACTED, "X-Api-Key": REDACTED}
    assert result["items"] == [{"cookie": REDACTED}, {"safe": "value"}]
    assert result["metadata"] == ["first", {"VECTOR": REDACTED}]
    assert value == original
    assert result["items"] is not value["items"]
    assert result["items"][1] is not value["items"][1]


def test_redact_recognizes_sensitive_key_tokens_case_insensitively():
    value = {
        "Proxy-Authorization": "Bearer xyz",
        "set-cookie": "sid=secret",
        "api_key": "key",
        "access_token": "token",
        "PASSWORD": "password",
        "raw_response": "full response",
        "response_text": "full response",
        "document_content": "full document",
        "body": "full body",
        "matched_value": "secret match",
    }

    assert redact(value) == {key: REDACTED for key in value}


def test_redact_only_preserves_well_formed_protected_metadata():
    value = {
        "document_length": "private text",
        "document_sha256": "private text",
        "vector_length": -1,
        "raw_response_sha256": "A" * 64,
        "content_length": 12,
    }

    assert redact(value) == {
        "document_length": REDACTED,
        "document_sha256": REDACTED,
        "vector_length": REDACTED,
        "raw_response_sha256": "A" * 64,
        "content_length": 12,
    }


def test_redact_does_not_alias_opaque_mutable_values():
    value = {"payload": bytearray(b"private bytes"), "other": {"private value"}}

    result = redact(value)

    assert result == {"payload": REDACTED, "other": REDACTED}
    assert value["payload"] == bytearray(b"private bytes")
    assert value["other"] == {"private value"}


def test_redact_rejects_nested_opaque_key_without_formatting_it():
    key = _OpaqueKey()

    with pytest.raises(ValueError, match="^summary keys must be built-in strings$"):
        redact({"nested": [{key: "value"}]})

    assert key.format_calls == []


def test_redact_rejects_string_subclass_key_without_formatting_it():
    key = _MarkedString("safe_key")

    with pytest.raises(ValueError, match="^summary keys must be built-in strings$"):
        redact({key: "value"})

    assert key.format_calls == []


def test_redact_never_retains_digest_string_subclass():
    digest = _MarkedString("a" * 64)

    result = redact({"document_sha256": digest})

    assert type(result["document_sha256"]) is str
    assert result["document_sha256"] == REDACTED
    assert result["document_sha256"] is not digest
    assert "private-marker" not in repr(result)
    assert digest.format_calls == []


def test_evidence_store_redacts_both_summaries_and_detaches_nested_values():
    store = EvidenceStore()
    request = {"headers": {"Authorization": "Bearer abc"}, "method": "GET"}
    response = {"document": "private text", "status_code": 200}

    item = store.record(
        capability_id="r1.fingerprint",
        trial_id="trial-1",
        state=EvidenceState.OBSERVED,
        confidence_basis="structured response",
        request_summary=request,
        response_summary=response,
    )

    assert item.request_summary == {"headers": {"Authorization": REDACTED}, "method": "GET"}
    assert item.response_summary == {"document": REDACTED, "status_code": 200}
    assert store.items == [item]
    request["headers"]["Authorization"] = "changed"
    response["status_code"] = 500
    assert item.request_summary["headers"]["Authorization"] == REDACTED
    assert item.response_summary["status_code"] == 200


def test_evidence_store_preserves_safe_artifact_metadata_and_identity():
    store = EvidenceStore()
    item = store.record(
        capability_id="r1.fingerprint",
        trial_id="trial-1",
        state=EvidenceState.VALIDATED,
        confidence_basis="digest comparison",
        request_summary={},
        response_summary={
            "document": "private text",
            "document_length": 12,
            "document_sha256": "a" * 64,
        },
        artifact_digest="a" * 64,
    )

    UUID(item.evidence_id)
    assert datetime.fromisoformat(item.timestamp).tzinfo == timezone.utc
    assert item.state is EvidenceState.VALIDATED
    assert item.response_summary == {
        "document": REDACTED,
        "document_length": 12,
        "document_sha256": "a" * 64,
    }
    assert item.artifact_digest == "a" * 64


def test_evidence_store_rejects_non_digest_artifact_text():
    store = EvidenceStore()

    with pytest.raises(ValueError, match="artifact_digest must be a SHA-256 digest"):
        store.record(
            capability_id="r1.fingerprint",
            trial_id="trial-1",
            state=EvidenceState.OBSERVED,
            confidence_basis="structured response",
            request_summary={},
            response_summary={},
            artifact_digest="private document text",
        )

    assert store.items == []


@pytest.mark.parametrize("state", list(EvidenceState))
def test_evidence_store_accepts_each_canonical_state(state):
    store = EvidenceStore()

    item = store.record(
        capability_id="r1.fingerprint",
        trial_id="trial-1",
        state=state,
        confidence_basis="structured response",
        request_summary={},
        response_summary={},
    )

    assert item.state is state
    assert store.items == [item]


@pytest.mark.parametrize(
    "invalid_state",
    ["observed", "high", "invented", None, [], {"state": "observed"}, 1],
    ids=["plain-valid-string", "confidence", "unknown", "none", "list", "dict", "number"],
)
def test_evidence_store_rejects_invalid_state_atomically(invalid_state):
    store = EvidenceStore()
    existing = store.record(
        capability_id="r1.fingerprint",
        trial_id="first",
        state=EvidenceState.OBSERVED,
        confidence_basis="structured response",
        request_summary={},
        response_summary={},
    )

    with pytest.raises(ValueError, match="^state must be an EvidenceState$"):
        store.record(
            capability_id="r1.fingerprint",
            trial_id="second",
            state=invalid_state,
            confidence_basis="structured response",
            request_summary={},
            response_summary={},
        )

    assert len(store.items) == 1
    assert store.items[0] is existing


def test_evidence_store_rejects_nested_opaque_key_atomically():
    store = EvidenceStore()
    key = _OpaqueKey()

    with pytest.raises(ValueError, match="^summary keys must be built-in strings$"):
        store.record(
            capability_id="r1.fingerprint",
            trial_id="trial-1",
            state=EvidenceState.OBSERVED,
            confidence_basis="structured response",
            request_summary={"nested": [{key: "value"}]},
            response_summary={},
        )

    assert store.items == []
    assert key.format_calls == []


def test_evidence_store_rejects_digest_string_subclass_atomically():
    store = EvidenceStore()
    digest = _MarkedString("a" * 64)

    with pytest.raises(ValueError, match="^artifact_digest must be a SHA-256 digest$"):
        store.record(
            capability_id="r1.fingerprint",
            trial_id="trial-1",
            state=EvidenceState.OBSERVED,
            confidence_basis="structured response",
            request_summary={},
            response_summary={},
            artifact_digest=digest,
        )

    assert store.items == []
    assert digest.format_calls == []
