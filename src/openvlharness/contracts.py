"""Small shared contracts; no provider, backend, or evaluation dependencies."""
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class VirtualCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ArtifactRef:
    kind: str
    id: str


@dataclass
class CapabilityResult:
    ok: bool
    data: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifact_refs: list[ArtifactRef] = field(default_factory=list)
    error: dict[str, Any] | None = None
    # Private trace/reporting fields never enter the model's task arguments.
    payload: dict[str, Any] = field(default_factory=dict)
    prepared: dict[str, Any] = field(default_factory=dict)
    processor_info: dict[str, Any] = field(default_factory=dict)
    detection: dict[str, Any] | None = None


class Capability(Protocol):
    name: str

    def public_schema(self) -> dict[str, Any]: ...

    def invoke(self, call: VirtualCall, memory: Any) -> CapabilityResult: ...
