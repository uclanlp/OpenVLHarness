"""Common capability lifecycle; concrete contracts live in the flat tool files."""
import copy
import json
from jsonschema import Draft7Validator

from ..contracts import ArtifactRef, CapabilityResult
from ..services.client import ToolClient
from ..memory.urls import rewrite_search_result_text, _resolve_urls_in_value
from ._images import _resolve_image_ids_in_args, prepare_images


def failure(call, code, message, *, details=None, meta=None):
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    payload = {"ok": False, "tool_name": call.name, "tool_call_id": call.call_id,
               "result": None, "error": error, "meta": meta or {}}
    return CapabilityResult(ok=False, error=error, payload=payload)


def store_selected_binding(memory, call, result, *, prefix, fields, descriptions, convert=None):
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
    rewrite_urls = False
    resolve_urls = False

    def __init__(self, *, backend, client_config, service_config=None, client=None):
        if backend not in self.backends:
            raise ValueError(f"Unsupported backend {backend!r} for {self.name}")
        self.backend = backend
        self.concrete_name = self.backends[backend]
        self.service_config = service_config or {}
        self.client = client or ToolClient(**client_config)
        self.validator = Draft7Validator(self.schema["function"]["parameters"])

    def public_schema(self):
        return copy.deepcopy(self.schema)

    def translate_args(self, args, memory):
        return args

    def execute(self, call, args, memory):
        return self.client.execute(self.concrete_name, args, tool_call_id=call.call_id)

    def store_result(self, call, result, memory):
        return []

    def normalize_result(self, call, result, prepared, binding_refs):
        raise NotImplementedError

    def validate_result(self, payload):
        """Validate backend structure before registering result artifacts."""

    def invoke(self, call, memory):
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
        """Register once in image → binding → URL order, then render evidence."""
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
