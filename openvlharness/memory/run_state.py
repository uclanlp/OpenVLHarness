"""Mutable registry state and descriptive metadata for one question."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

@dataclass
class FieldDescriptor:
    """Human-readable metadata for one field in a Python binding.

    Attributes:
        dtype: Type description, including array dimensions when available.
        description: Capability-supplied meaning of the field.
        shape_hint: Optional additional shape information.
        preview: Bounded text preview suitable for code-generation context.
    """
    dtype: str = ""
    description: str = ""
    shape_hint: str = ""
    preview: str = ""

@dataclass
class BindingDescriptor:
    """Provenance and field descriptions for a stored Python binding.

    Attributes:
        key: Unique variable name in the execution context.
        source_tool: Capability that produced the value.
        source_args: Compact task arguments with large image payloads removed.
        fields: Field names mapped to descriptive metadata.
    """
    key: str = ""
    source_tool: str = ""
    source_args: Dict[str, Any] = field(default_factory=dict)
    fields: Dict[str, FieldDescriptor] = field(default_factory=dict)

@dataclass
class ImagesState:
    """Image entries, alias indexes and inventory bookkeeping for one question.

    Registry entries contain either image data or a URL to fetch on demand.
    tool_generated_counter allocates IDs; last_snapshot tracks inventory emission.
    """
    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alias_to_ref: Dict[str, str] = field(default_factory=dict)
    url_to_ref: Dict[str, str] = field(default_factory=dict)
    tool_generated_counter: int = 0
    last_snapshot: Optional[Tuple[Tuple[str, str], ...]] = None

@dataclass
class BindingsState:
    """Python values, unique-name counters and their descriptive metadata.

    last_descriptors_count tracks the most recently emitted binding inventory.
    """
    bindings: Dict[str, Any] = field(default_factory=dict)
    binding_counters: Dict[str, int] = field(default_factory=dict)
    descriptors: Dict[str, BindingDescriptor] = field(default_factory=dict)
    last_descriptors_count: int = -1

@dataclass
class UrlsState:
    """Webpage references, reverse URL lookup and inventory counters.

    Each registry entry records its URL, domain, source and first-seen turn.
    """
    webpage_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    url_to_ref: Dict[str, str] = field(default_factory=dict)
    webpage_counter: int = 0
    last_webpage_counter: int = -1

@dataclass
class MemoryRunState:
    """Aggregate mutable state owned by one MemoryStore.

    Attributes:
        images: Image registry and aliases.
        bindings: Python execution values and descriptions.
        urls: Webpage references and provenance.
        scratch: Temporary state shared by inventory builders.
    """
    images: ImagesState = field(default_factory=ImagesState)
    bindings: BindingsState = field(default_factory=BindingsState)
    urls: UrlsState = field(default_factory=UrlsState)
    scratch: Dict[str, Dict[str, Any]] = field(default_factory=dict)

