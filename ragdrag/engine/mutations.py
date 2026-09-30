"""Track write attempts and invoke their registered cleanup callbacks."""

from __future__ import annotations

from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable
from concurrent.futures import CancelledError as FuturesCancelledError
from uuid import uuid4

from ragdrag.engine.models import CleanupState, MutationRecord

CleanupCallback = Callable[[], CleanupState]
_NEEDS_CLEANUP = {CleanupState.ACTIVE, CleanupState.UNKNOWN, CleanupState.UNRESOLVED}
_CLEANED = {CleanupState.REMOVED, CleanupState.RESTORED}


class MutationLedger:
    def __init__(self) -> None:
        self.records: list[MutationRecord] = []
        self._cleanup: dict[str, CleanupCallback] = {}
        self._cleanup_interruption: BaseException | None = None

    def clear(self) -> None:
        """Start a new engagement while preserving this ledger's identity."""
        self.records.clear()
        self._cleanup.clear()
        self._cleanup_interruption = None

    @property
    def cleanup_interruption(self) -> BaseException | None:
        return self._cleanup_interruption

    @property
    def has_unresolved(self) -> bool:
        return any(record.state in _NEEDS_CLEANUP for record in self.records)

    def record_attempt(
        self,
        capability_id: str,
        target_scope: str,
        operation: str,
        object_id: str,
        cleanup_method: str,
    ) -> MutationRecord:
        record = MutationRecord(
            mutation_id=str(uuid4()),
            capability_id=capability_id,
            target_scope=target_scope,
            operation=operation,
            object_id=object_id,
            cleanup_method=cleanup_method,
            state=CleanupState.UNKNOWN,
        )
        self.records.append(record)
        return record

    def _record(self, mutation_id: str) -> MutationRecord:
        for record in self.records:
            if record.mutation_id == mutation_id:
                return record
        raise KeyError(mutation_id)

    def mark_created(self, mutation_id: str) -> None:
        self._record(mutation_id).state = CleanupState.ACTIVE

    def mark_not_created(self, mutation_id: str) -> None:
        self._record(mutation_id).state = CleanupState.NOT_CREATED

    def register_cleanup(self, mutation_id: str, callback: CleanupCallback) -> None:
        self._record(mutation_id)
        self._cleanup[mutation_id] = callback

    def cleanup_all(self) -> None:
        for record in self.records:
            if record.state not in _NEEDS_CLEANUP:
                continue
            record.cleanup_attempts += 1
            callback = self._cleanup.get(record.mutation_id)
            if callback is None:
                record.state = CleanupState.UNRESOLVED
                continue
            try:
                final_state = callback()
            except BaseException as exc:
                record.state = CleanupState.UNRESOLVED
                if self._cleanup_interruption is None and (
                    not isinstance(exc, Exception)
                    or isinstance(exc, FuturesCancelledError)
                    or isinstance(exc, AsyncCancelledError)
                ):
                    self._cleanup_interruption = exc
                continue
            record.state = (
                final_state
                if isinstance(final_state, CleanupState) and final_state in _CLEANED
                else CleanupState.UNRESOLVED
            )
