"""Per-question artifact storage and inventory emission, independent of tools."""
from .run_state import MemoryRunState
from .images import (
    _snapshot, build_image_registry_from_inputs, next_tool_generated_image_ref_id,
    register_image_data_url, register_image_url_lazy, resolve_registry_image_data_url,
)
from .bindings import store_binding, get_binding_descriptors_payload, copy_bindings_for_execution
from .urls import register_webpage_url, _resolve_urls_in_value
from .environment_update import (
    EnvironmentUpdateBuilder, add_image_inventory, add_binding_inventory, add_url_inventory,
)


class MemoryStore:
    """Own the artifacts and model-visible inventories for a single question.

    Image bytes and Python arrays stay in registry storage. Environment updates
    expose references and concise descriptions that the model can use in tool calls.
    """
    def __init__(self, *, modules=("image", "tool_context", "url"), emit_policy="always",
                 webpage_env_update_limit=20):
        """Initialize an empty store with an inventory emission policy.

        Args:
            modules: Unique enabled modules in image, tool_context, url order.
            emit_policy: Emit inventories always or only on_change.
            webpage_env_update_limit: Maximum webpage references shown per update,
                or None to show all references. Stored URLs are not discarded.

        Raises:
            ValueError: Module order, names, emission policy or limit is invalid.
        """
        if emit_policy not in {"always", "on_change"}:
            raise ValueError("memory.emit_policy must be always or on_change")
        if len(set(modules)) != len(modules) or any(m not in {"image", "tool_context", "url"} for m in modules):
            raise ValueError("memory.modules must contain unique image, tool_context, or url entries")
        if list(modules) != [m for m in ("image", "tool_context", "url") if m in modules]:
            raise ValueError("Memory processing order must be image, tool_context, url")
        if webpage_env_update_limit is not None and (
            type(webpage_env_update_limit) is not int or webpage_env_update_limit <= 0
        ):
            raise ValueError("webpage_env_update_limit must be a positive integer or null")
        self.modules = tuple(modules)
        self.emit_policy = emit_policy
        self.webpage_env_update_limit = webpage_env_update_limit
        self.state = MemoryRunState()

    def start(self, question):
        """Reset all question state, register input images and return an inventory message.

        Args:
            question: Normalized image/text input items.

        Returns:
            A user-role environment update, or None if no inventory is emitted.
        """
        self.state = MemoryRunState()
        if "image" in self.modules:
            build_image_registry_from_inputs(self.state.images, question)
        return self.build_environment_update()

    def resolve_image(self, ref):
        """Resolve an image reference to (data_url, registry_id, resolution_note).

        Accepts IDs, aliases, inline image data and HTTP(S) URLs. Lazy references may
        trigger a download. Unresolvable inputs return a tuple of None values.
        """
        return resolve_registry_image_data_url(self.state.images, ref)

    def register_image(self, data_url, *, aliases=(), provenance="runtime"):
        """Store an image data URL and return its deduplicated registry ID.

        Args:
            data_url: MIME-prefixed image data URL; other values return None.
            aliases: Additional names that resolve to this image.
            provenance: Source description included in image inventories.
        """
        if not isinstance(data_url, str) or not data_url.strip().startswith("data:image/"):
            return None
        data_url = data_url.strip()
        images = self.state.images
        existing = images.url_to_ref.get(data_url)
        proposed = existing if existing in images.registry else next_tool_generated_image_ref_id(images)
        return register_image_data_url(images, proposed, data_url, source=provenance, aliases=list(aliases))

    def register_lazy_image(self, url, *, provenance="", turn_idx=-1):
        """Register a remote image URL without downloading it and return its ID.

        provenance records the source tool. turn_idx is accepted but not stored by
        the image registry. Empty or non-string URLs return None.
        """
        return register_image_url_lazy(self.state.images, url, source_tool=provenance, turn_idx=turn_idx)

    def register_url(self, url, *, provenance="", turn_idx=-1):
        """Register a webpage URL with source and turn metadata; return its stable ID."""
        return register_webpage_url(self.state.urls, url, source_tool=provenance, turn_idx=turn_idx)

    def resolve_url(self, ref):
        """Resolve webpage references recursively in strings, lists and mappings.

        Unknown references and non-reference strings pass through unchanged.
        """
        return _resolve_urls_in_value(self.state, ref, rewrites=[], path="url")

    def store_binding(self, **kwargs):
        """Store a described tool value and return a unique Python variable name.

        Keyword arguments are those of memory.bindings.store_binding: prefix, query,
        value, source_tool, source_args and per-field descriptions.
        """
        return store_binding(self.state, **kwargs)

    def describe_bindings(self):
        """Return JSON-compatible descriptors for stored Python bindings."""
        return get_binding_descriptors_payload(self.state)

    def copy_bindings_for_execution(self):
        """Return deep copies of all bindings for isolated execution inputs."""
        return copy_bindings_for_execution(self.state)

    def build_environment_update(self, *, successful=True):
        """Build the enabled inventories and update emission bookkeeping.

        Args:
            successful: Whether the tool turn succeeded. False skips emission and
                leaves inventory snapshots untouched.

        Returns:
            A user-role inventory message, or None when empty or suppressed by policy.
            on_change compares image sources, URL count and binding-descriptor count.
        """
        if not successful:
            return None
        state = self.state
        builder = EnvironmentUpdateBuilder.reset(state)
        for module in self.modules:
            if module == "image":
                add_image_inventory(state)
            elif module == "tool_context":
                add_binding_inventory(state)
            else:
                add_url_inventory(state, self.webpage_env_update_limit)
        snapshot = _snapshot(state.images)
        changed = (snapshot != state.images.last_snapshot
                   or state.urls.webpage_counter != state.urls.last_webpage_counter
                   or len(state.bindings.descriptors) != state.bindings.last_descriptors_count)
        state.images.last_snapshot = snapshot
        state.urls.last_webpage_counter = state.urls.webpage_counter
        state.bindings.last_descriptors_count = len(state.bindings.descriptors)
        if builder.is_empty() or (self.emit_policy == "on_change" and not changed):
            return None
        return builder.build_message()
