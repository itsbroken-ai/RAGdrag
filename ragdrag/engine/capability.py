"""Generic capability contracts and fail-closed execution policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from ragdrag.engine.evidence import EvidenceStore
from ragdrag.engine.models import CapabilityMetadata, CapabilityResult, ImpactLevel
from ragdrag.engine.mutations import MutationLedger
from ragdrag.engine.profile import TargetProfile
from ragdrag.engine.transport import HTTPClient


class InvalidConfiguration(ValueError):
    """The caller supplied an invalid engagement configuration."""


@dataclass
class CapabilityContext:
    profile: TargetProfile
    client: HTTPClient
    evidence: EvidenceStore
    mutations: MutationLedger
    started_at: str


class Capability(Protocol):
    metadata: CapabilityMetadata

    def execute(self, context: CapabilityContext) -> CapabilityResult: ...


_IMPACT_ORDER = {
    ImpactLevel.PASSIVE: 0,
    ImpactLevel.ACTIVE: 1,
    ImpactLevel.MUTATING: 2,
}


class SafetyPolicy:
    def __init__(
        self,
        ceiling: ImpactLevel,
        *,
        authorized_controls: Iterable[str] = (),
    ) -> None:
        if type(ceiling) is not ImpactLevel:
            raise InvalidConfiguration("impact ceiling must be a known ImpactLevel")
        controls = frozenset(authorized_controls)
        if any(type(item) is not str or not item for item in controls):
            raise InvalidConfiguration("authorized controls must be nonempty strings")
        self.ceiling = ceiling
        self.authorized_controls = controls

    def authorize(self, metadata: CapabilityMetadata) -> tuple[bool, str]:
        if type(metadata) is not CapabilityMetadata:
            return False, "capability metadata is invalid"
        if type(metadata.capability_id) is not str or not metadata.capability_id:
            return False, "capability identifier is invalid"
        if type(metadata.impact) is not ImpactLevel:
            return False, f"{metadata.capability_id} has unknown impact"
        if type(metadata.creates_mutations) is not bool:
            return False, f"{metadata.capability_id} has invalid mutation metadata"
        if metadata.creates_mutations and metadata.impact is not ImpactLevel.MUTATING:
            return False, f"{metadata.capability_id} has inconsistent mutation metadata"
        if _IMPACT_ORDER[metadata.impact] > _IMPACT_ORDER[self.ceiling]:
            return False, f"{metadata.capability_id} requires {metadata.impact.value} impact"
        if type(metadata.required_controls) is not tuple or any(
            type(item) is not str or not item for item in metadata.required_controls
        ):
            return False, f"{metadata.capability_id} has invalid control metadata"
        if type(metadata.validation_test) not in {str, type(None)}:
            return False, f"{metadata.capability_id} has invalid validation metadata"
        required = set(metadata.required_controls)
        if metadata.validation_test is not None:
            if not metadata.validation_test:
                return False, f"{metadata.capability_id} has invalid validation metadata"
            required.add(metadata.validation_test)
        missing = required - self.authorized_controls
        if missing:
            return False, f"{metadata.capability_id} requires explicit authorization"
        return True, "authorized"


class CapabilityRegistry:
    """Insertion-ordered registry of caller-supplied inert capability objects."""

    def __init__(self, capabilities: Iterable[Capability] = ()) -> None:
        self._capabilities: dict[str, Capability] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: Capability) -> None:
        metadata = getattr(capability, "metadata", None)
        capability_id = getattr(metadata, "capability_id", None)
        if type(metadata) is not CapabilityMetadata or type(capability_id) is not str or not capability_id:
            raise InvalidConfiguration("capability metadata must include a nonempty identifier")
        if not callable(getattr(capability, "execute", None)):
            raise InvalidConfiguration(f"{capability_id} has no execute method")
        if capability_id in self._capabilities:
            raise InvalidConfiguration(f"duplicate capability identifier: {capability_id}")
        self._capabilities[capability_id] = capability

    def select(self, capability_ids: Iterable[str] | None = None) -> list[Capability]:
        if capability_ids is None:
            return list(self._capabilities.values())
        selected: list[Capability] = []
        seen: set[str] = set()
        for capability_id in capability_ids:
            if capability_id in seen:
                raise InvalidConfiguration(f"duplicate capability selection: {capability_id}")
            if capability_id not in self._capabilities:
                raise InvalidConfiguration(f"unknown capability identifier: {capability_id}")
            selected.append(self._capabilities[capability_id])
            seen.add(capability_id)
        return selected
