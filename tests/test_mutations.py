"""Mutation records remain auditable through cleanup and retries."""

from uuid import UUID

from ragdrag.engine.models import CleanupState
from ragdrag.engine.mutations import MutationLedger


def test_attempt_is_unresolved_until_creation_outcome_is_known():
    ledger = MutationLedger()

    record = ledger.record_attempt(
        "r4.poison", "https://target.test", "insert", "doc-1", "DELETE /documents/doc-1"
    )

    UUID(record.mutation_id)
    assert ledger.records == [record]
    assert record.state is CleanupState.UNKNOWN
    assert record.cleanup_attempts == 0
    assert ledger.has_unresolved is True


def test_successful_cleanup_moves_active_record_to_removed_once():
    ledger = MutationLedger()
    record = ledger.record_attempt(
        "r4.poison", "https://target.test", "insert", "doc-1", "DELETE /documents/doc-1"
    )
    ledger.mark_created(record.mutation_id)
    calls = []

    def remove():
        calls.append("removed")
        return CleanupState.REMOVED

    ledger.register_cleanup(record.mutation_id, remove)
    ledger.cleanup_all()
    ledger.cleanup_all()

    assert record.state is CleanupState.REMOVED
    assert record.cleanup_attempts == 1
    assert calls == ["removed"]
    assert ledger.has_unresolved is False


def test_failed_cleanup_is_unresolved_and_can_be_retried():
    ledger = MutationLedger()
    record = ledger.record_attempt(
        "r4.poison", "https://target.test", "insert", "doc-1", "DELETE /documents/doc-1"
    )
    ledger.mark_created(record.mutation_id)
    attempts = []

    def fail_then_restore():
        attempts.append("attempt")
        if len(attempts) == 1:
            raise RuntimeError("delete failed")
        return CleanupState.RESTORED

    ledger.register_cleanup(record.mutation_id, fail_then_restore)
    ledger.cleanup_all()
    assert record.state is CleanupState.UNRESOLVED
    assert record.cleanup_attempts == 1
    assert ledger.has_unresolved is True

    ledger.cleanup_all()
    assert record.state is CleanupState.RESTORED
    assert record.cleanup_attempts == 2
    assert ledger.has_unresolved is False


def test_unknown_attempt_without_callback_becomes_unresolved():
    ledger = MutationLedger()
    record = ledger.record_attempt(
        "r4.poison", "https://target.test", "insert", "doc-1", "DELETE /documents/doc-1"
    )

    ledger.cleanup_all()

    assert record.state is CleanupState.UNRESOLVED
    assert record.cleanup_attempts == 1
    assert ledger.has_unresolved is True


def test_not_created_attempt_is_skipped_during_cleanup():
    ledger = MutationLedger()
    record = ledger.record_attempt(
        "r4.poison", "https://target.test", "insert", "doc-1", "DELETE /documents/doc-1"
    )
    ledger.mark_not_created(record.mutation_id)

    ledger.cleanup_all()

    assert record.state is CleanupState.NOT_CREATED
    assert record.cleanup_attempts == 0
    assert ledger.has_unresolved is False


def test_cleanup_continues_in_record_order_after_callback_failure():
    ledger = MutationLedger()
    first = ledger.record_attempt("r4.poison", "target", "insert", "one", "delete one")
    second = ledger.record_attempt("r4.poison", "target", "insert", "two", "delete two")
    ledger.mark_created(first.mutation_id)
    ledger.mark_created(second.mutation_id)
    order = []

    def fail():
        order.append("one")
        raise RuntimeError("delete failed")

    def remove():
        order.append("two")
        return CleanupState.REMOVED

    ledger.register_cleanup(first.mutation_id, fail)
    ledger.register_cleanup(second.mutation_id, remove)
    ledger.cleanup_all()

    assert order == ["one", "two"]
    assert [record.state for record in ledger.records] == [
        CleanupState.UNRESOLVED,
        CleanupState.REMOVED,
    ]
    assert ledger.has_unresolved is True


def test_non_terminal_cleanup_result_stays_unresolved():
    ledger = MutationLedger()
    record = ledger.record_attempt("r4.poison", "target", "insert", "one", "delete one")
    ledger.mark_created(record.mutation_id)
    ledger.register_cleanup(record.mutation_id, lambda: CleanupState.ACTIVE)

    ledger.cleanup_all()

    assert record.state is CleanupState.UNRESOLVED
    assert record.cleanup_attempts == 1
