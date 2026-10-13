"""Common capability lifecycle; concrete contracts live in the flat tool files."""
import copy
import json
from jsonschema import Draft7Validator

from ..contracts import ArtifactRef, CapabilityResult
from ..services.client import ToolClient
from ..memory.urls import rewrite_search_result_text, _resolve_urls_in_value
from ._images import _resolve_image_ids_in_args, prepare_images


def failure(call, code, message, *, details=None, meta=None):
    """Build a failed CapabilityResult with correlated call identifiers and optional diagnostics."""
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    payload = {"ok": False, "tool_name": call.name, "tool_call_id": call.call_id,
               "result": None, "error": error, "meta": meta or {}}
    return CapabilityResult(ok=False, error=error, payload=payload)


def store_selected_binding(memory, call, result, *, prefix, fields, descriptions, convert=None):
    """Register selected result fields as one described Python binding.

    Args:
        memory: Current question's MemoryStore.
        call: Public tool call used for query naming and provenance.
        result: Backend result mapping.
        prefix: Binding-name prefix.
        fields: Result fields eligible for storage.
        descriptions: Field meanings supplied by the capability.
        convert: Optional callable transforming the selected field mapping.

    Returns:
        A list containing the binding ID, or an empty list when no fields or
        call arguments are available. Values are selected without deep-copying.
    """
    if not isinstance(result, dict) or not call.arguments:
        return []
    value = {key: result[key] for key in fields if key in result}
    if not value:
        return []
    if convert:
        value = convert(value)
    query = next((call.arguments[key].strip() for key in ("query", "prompt", "text")
                  if isinstance(call.arguments.get(key), str) and call.arguments[key].strip()), "")
    return [memory.store_binding(prefix=prefix, query=query, value=value,
        source_tool=call.name, source_args=call.arguments, descriptions=descriptions)]


class BaseCapability:
    """Validate task calls, resolve artifacts, execute tools and render model evidence.

    Subclasses define name, backends, schema and process_output, plus canonical
    result normalization. Optional hooks translate arguments, validate backend
    results and store structured bindings before evidence is rendered.
    """
    rewrite_urls = False
    resolve_urls = False

    def __init__(self, *, backend, client_config, service_config=None, client=None):
        """Select a supported backend and prepare schema validation and transport.

        Args:
            backend: Implementation key declared by the subclass.
            client_config: ToolClient connection settings.
            service_config: Backend-specific deployment settings.
            client: Optional injected transport with an execute method.

        Raises:
            ValueError: The backend or client configuration is unsupported.
        """
        if backend not in self.backends:
            raise ValueError(f"Unsupported backend {backend!r} for {self.name}")
        self.backend = backend
        self.concrete_name = self.backends[backend]
        self.service_config = service_config or {}
        self.client = client or ToolClient(**client_config)
        self.validator = Draft7Validator(self.schema["function"]["parameters"])

    def public_schema(self):
        """Return an independent copy of the model-facing function schema."""
        return copy.deepcopy(self.schema)

    def translate_args(self, args, memory):
        """Return resolved task arguments; subclasses may add service-specific inputs."""
        return args

    def execute(self, call, args, memory):
        """Invoke the concrete backend through the tool client using the public call ID."""
        return self.client.execute(self.concrete_name, args, tool_call_id=call.call_id)

    def store_result(self, call, result, memory):
        """Return binding IDs created from a result; the base implementation stores none."""
        return []

    def normalize_result(self, call, result, prepared, binding_refs):
        """Convert a backend result into the capability's canonical output mapping.

        Subclasses must implement this hook. prepared contains rendered image/text
        data; binding_refs identifies structured values already stored in memory.
        """
        raise NotImplementedError

    def validate_result(self, payload):
        """Validate a backend envelope before registering artifacts.

        Subclasses may raise ValueError, TypeError or KeyError for malformed results;
        invoke converts those exceptions to invalid_tool_result failures.
        """

    def invoke(self, call, memory):
        """Run a public call through validation, execution and evidence preparation.

        Args:
            call: VirtualCall with a public name, correlation ID and parsed arguments.
            memory: Current question's MemoryStore, updated by successful results.

        Returns:
            A CapabilityResult containing canonical data, model evidence and report
            metadata. Invalid arguments, unresolved images, execution exceptions and
            invalid backend structures produce explicit failure results.

        Image and URL references are resolved in a copy of the task arguments before
        translation. Image resolution can download lazy references.
        """
        allowed = self.schema["function"]["parameters"].get("properties", {})
        hidden = set(call.arguments) - set(allowed)
        errors = sorted(self.validator.iter_errors(call.arguments), key=lambda e: str(e.path))
        if hidden or errors:
            message = (f"Unsupported task arguments: {', '.join(sorted(hidden))}." if hidden
                       else errors[0].message)
            return failure(call, "invalid_tool_arguments", message)
        args = copy.deepcopy(call.arguments)
        resolution = {"rewrites": [], "errors": []}
        if "image" in memory.modules:
            _resolve_image_ids_in_args(memory.state.images, args, path="args", **resolution)
        if self.resolve_urls and "url" in memory.modules:
            for key in ("urls", "url"):
                if key in args:
                    args[key] = _resolve_urls_in_value(memory.state, args[key],
                        rewrites=resolution["rewrites"], path=f"args.{key}")
        if resolution["errors"]:
            return failure(call, "unresolved_image_reference",
                "Failed to map one or more image_id arguments to runtime images.",
                details=resolution["errors"], meta={"image_resolution": resolution})
        try:
            args = self.translate_args(args, memory)
            payload = self.execute(call, args, memory)
        except Exception as exc:
            return failure(call, "tool_execution_error", str(exc))
        payload = dict(payload)
        payload["tool_name"] = self.name
        payload["tool_call_id"] = call.call_id
        payload["meta"] = dict(payload.get("meta") or {})
        payload["meta"]["backend"] = self.backend
        if resolution["rewrites"]:
            payload["meta"]["image_resolution"] = resolution
        if payload.get("ok") is not True:
            return CapabilityResult(ok=False, error=payload.get("error"), payload=payload)
        try:
            self.validate_result(payload)
        except (ValueError, TypeError, KeyError) as exc:
            return failure(call, "invalid_tool_result", str(exc), meta=payload["meta"])
        return self.consume_result(call, payload, memory)

    def consume_result(self, call, payload, memory):
        """Register successful results in image, binding, then URL order and render evidence.

        Args:
            call: Public tool call providing task arguments and correlation ID.
            payload: Successful, structurally validated execute envelope.
            memory: Question memory receiving artifacts and bindings.

        Returns:
            A CapabilityResult with canonical output, model evidence and artifact
            references. Processor exceptions are recorded and use default evidence
            rendering; normalization and memory registration errors propagate.
        """
        compact = {key: payload.get(key) for key in ("ok", "result", "error")}
        try:
            text = json.dumps(compact, ensure_ascii=False)
        except Exception:
            text = str(compact)
        prepared = {"text_payload": text, "compact_payload": compact,
                    "compact_result": None, "image_parts": [], "image_refs": [],
                    "service_meta": payload.get("meta") or {}}
        if "image" in memory.modules:
            prepared = prepare_images(payload, prepared, memory.state)
        binding_refs = []
        if "tool_context" in memory.modules:
            binding_refs = self.store_result(call, payload.get("result"), memory)
        if self.rewrite_urls and "url" in memory.modules:
            prepared["text_payload"] = rewrite_search_result_text(memory.state,
                prepared["text_payload"], source_tool=self.name)
        custom = None
        info = {"mode": "custom", "status": "ok", "module": type(self).__module__}
        try:
            custom = self.process_output(payload, prepared)
        except Exception as exc:
            info.update(status="fallback", fallback_reason=f"processor_exception:{type(exc).__name__}: {exc}")
        if isinstance(custom, dict):
            info.update(custom.get("processor_info") or {})
        if isinstance(custom, dict) and isinstance(custom.get("user_message"), dict):
            content = custom["user_message"].get("content")
        else:
            lines = [f"Tool `{self.name}` (tool_call_id={call.call_id}) returned:"]
            if prepared["image_refs"]:
                lines.append(f"Image refs: {', '.join(prepared['image_refs'])}")
            lines.append(prepared["text_payload"])
            content = [{"type": "text", "text": "\n".join(lines)}] + prepared["image_parts"]
        evidence = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        artifacts = [ArtifactRef("image", ref) for ref in prepared["image_refs"]]
        artifacts.extend(ArtifactRef("binding", ref) for ref in binding_refs)
        return CapabilityResult(ok=True, data=self.normalize_result(call, payload.get("result"), prepared, binding_refs),
            evidence=evidence, artifact_refs=artifacts, payload=payload, prepared=prepared,
            processor_info=info, detection=custom.get("tool_detection") if isinstance(custom, dict) else None)
