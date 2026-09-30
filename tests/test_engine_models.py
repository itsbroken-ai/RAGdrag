"""Contracts for the stable engine data models."""

import pytest

from ragdrag.core.models import Finding as LegacyFinding
from ragdrag.engine.models import (
    EvidenceState,
    ExitCode,
    Finding,
    ImpactLevel,
    RequestBudget,
)


def test_legacy_finding_import_is_engine_finding():
    assert LegacyFinding is Finding


def test_new_finding_fields_preserve_legacy_equality():
    first = Finding("RD-0201", "Chunk Boundary Detection", "high", "detail")
    second = Finding("RD-0201", "Chunk Boundary Detection", "high", "detail")
    assert first == second
    assert first.finding_id != second.finding_id
    assert first.evidence_state is EvidenceState.INFERRED


def test_exit_codes_match_public_contract():
    assert [member.value for member in ExitCode] == [0, 1, 2, 3, 4, 5]
    assert ImpactLevel.MUTATING.value == "mutating"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_requests": 0},
        {"timeout_seconds": 0},
        {"max_response_bytes": 0},
        {"max_redirects": -1},
        {"max_concurrency": 0},
    ],
)
def test_request_budget_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        RequestBudget(**kwargs)


@pytest.mark.parametrize(
    "field_name", ("max_requests", "max_response_bytes", "max_redirects", "max_concurrency")
)
@pytest.mark.parametrize("invalid", (True, 1.5, float("nan"), float("inf")))
def test_request_budget_rejects_non_integer_count_fields(field_name, invalid):
    with pytest.raises(ValueError):
        RequestBudget(**{field_name: invalid})


@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), float("-inf")))
def test_request_budget_rejects_non_finite_or_boolean_timeout(invalid):
    with pytest.raises(ValueError):
        RequestBudget(timeout_seconds=invalid)


def test_request_budget_accepts_finite_boundary_values():
    budget = RequestBudget(
        max_requests=1,
        timeout_seconds=0.001,
        max_response_bytes=1,
        max_redirects=0,
        max_concurrency=1,
    )
    assert budget.max_requests == 1
    assert budget.timeout_seconds == 0.001
    assert budget.max_response_bytes == 1
    assert budget.max_redirects == 0
    assert budget.max_concurrency == 1
