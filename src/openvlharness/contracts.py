"""Small shared contracts; no provider, backend, or evaluation dependencies."""
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class VirtualCall:
    """A model-requested capability invocation.

    Attributes:
        call_id: Identifier used to correlate the call and its result.
        name: Public capability name advertised in the model's tool schema.
        arguments: Parsed task arguments before memory references are resolved.
    """
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ArtifactRef:
    """An artifact registered in per-question memory.

    Attributes:
        kind: Registry category, such as image or binding.
        id: Identifier used to retrieve the artifact from that registry.
    """
    kind: str
    id: str


@dataclass
class ServiceResult:
    """A backend response with metadata that survives router serialization.

    Attributes:
        value: Tool output to place in the execute envelope's result field.
        metadata: Structured provenance and diagnostics for the envelope's meta field.
    """
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityResult:
    """A capability outcome shared by the agent and model adapter.

    Attributes:
        ok: Whether validation, execution and result processing succeeded.
        data: Canonical structured output, or None on failure.
        evidence: Text and image content parts presented to the model.
        artifact_refs: References to values registered in question memory.
        error: Structured failure details, or None on success.
        payload: Execute envelope preserved for reporting and protocol rendering.
        prepared: Rendered text, image attachments and their registry references.
        processor_info: Evidence-processing metadata for trajectory reports.
        detection: Optional boxes and confidence scores for evaluation sidecars.
    """
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
    """Protocol implemented by model-facing capabilities."""
    name: str

    def public_schema(self) -> dict[str, Any]:
        """Return the function-tool schema advertised to the model."""
        ...

    def invoke(self, call: VirtualCall, memory: Any) -> CapabilityResult:
        """Execute a call against question memory and return its structured outcome."""
        ...
