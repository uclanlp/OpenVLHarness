from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FieldDescriptor:
    dtype: str = ""
    description: str = ""
    shape_hint: str = ""
    preview: str = ""


@dataclass
class BindingDescriptor:
    key: str = ""
    source_tool: str = ""
    source_args: Dict[str, Any] = field(default_factory=dict)
    fields: Dict[str, FieldDescriptor] = field(default_factory=dict)


@dataclass
class ImagesState:
    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alias_to_ref: Dict[str, str] = field(default_factory=dict)
    url_to_ref: Dict[str, str] = field(default_factory=dict)
    tool_generated_counter: int = 0
    last_snapshot: Optional[Tuple[Tuple[str, str], ...]] = None


@dataclass
class BindingsState:
    bindings: Dict[str, Any] = field(default_factory=dict)
    binding_counters: Dict[str, int] = field(default_factory=dict)
    descriptors: Dict[str, BindingDescriptor] = field(default_factory=dict)
    last_descriptors_count: int = -1


@dataclass
class UrlsState:
    webpage_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    url_to_ref: Dict[str, str] = field(default_factory=dict)
    webpage_counter: int = 0
    last_webpage_counter: int = -1


@dataclass
class ToolOutputsState:
    processor_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_infra_src_path_bootstrapped: bool = False


@dataclass
class MemoryRunState:
    images: ImagesState = field(default_factory=ImagesState)
    bindings: BindingsState = field(default_factory=BindingsState)
    urls: UrlsState = field(default_factory=UrlsState)
    tool_outputs: ToolOutputsState = field(default_factory=ToolOutputsState)
    scratch: Dict[str, Dict[str, Any]] = field(default_factory=dict)
