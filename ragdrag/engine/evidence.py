"""In-memory evidence records containing only redacted summaries."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ragdrag.engine.models import EvidenceItem, EvidenceState
from ragdrag.engine.redaction import redact


class EvidenceStore:
    def __init__(self) -> None:
        self.items: list[EvidenceItem] = []

    def record(
        self,
        *,
        capability_id: str,
        trial_id: str,
        state: EvidenceState,
        confidence_basis: str,
        request_summary: dict[str, Any],
        response_summary: dict[str, Any],
        artifact_digest: str | None = None,
    ) -> EvidenceItem:
        if type(state) is not EvidenceState:
            raise ValueError("state must be an EvidenceState")
        if artifact_digest is not None and (
            type(artifact_digest) is not str
            or re.fullmatch(r"[0-9a-fA-F]{64}", artifact_digest) is None
        ):
            raise ValueError("artifact_digest must be a SHA-256 digest")
        item = EvidenceItem(
            evidence_id=str(uuid4()),
            capability_id=capability_id,
            trial_id=trial_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            state=state,
            confidence_basis=confidence_basis,
            request_summary=redact(request_summary),
            response_summary=redact(response_summary),
            artifact_digest=artifact_digest,
        )
        self.items.append(item)
        return item
