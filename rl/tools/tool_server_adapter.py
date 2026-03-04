"""verl BaseTool adapter for integrate-verl FastAPI tool server."""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
import re
import time
from typing import Any, Optional
from uuid import uuid4

import requests
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

_IMAGE_REF_PATTERN = re.compile(r"^image_(\d+)$")
_IMAGE_RESULT_HINTS = {
    "image_base64",
    "rendered_image_b64",
    "cropped_image_base64",
    "overlay_image_base64",
    "mask_image_base64",
    "depth_image_base64",
}


class MiniVLMHTTPTool(BaseTool):
    """Forward verl tool calls to integrate-verl tool server `/execute`."""

    def __init__(self, config: dict, tool_schema):
        super().__init__(config=config, tool_schema=tool_schema)
        self.tool_base = str(config.get("tool_base", "http://127.0.0.1:8101")).rstrip("/")
        self.tool_name = str(config.get("tool_name") or self.name)
        self.request_timeout_sec = float(config.get("request_timeout_sec", 120))
        self.emit_metrics_log_path = config.get("emit_metrics_log_path")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        del kwargs
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return await asyncio.to_thread(self._execute_sync, instance_id, parameters, kwargs)

    def _execute_sync(self, instance_id: str, parameters: dict[str, Any], kwargs: dict[str, Any]) -> tuple[ToolResponse, float, dict]:
        start = time.perf_counter()
        agent_data = kwargs.get("agent_data")

        tool_args = copy.deepcopy(parameters) if isinstance(parameters, dict) else {}
        rewrite_info = self._rewrite_image_path_refs(tool_args, agent_data)
        self._inject_default_image_if_needed(tool_args, agent_data)

        tool_call_id = f"{instance_id}_{uuid4().hex[:8]}"
        request_payload = {
            "tool_name": self.tool_name,
            "args": tool_args,
            "tool_call_id": tool_call_id,
        }

        status_code = None
        try:
            response = requests.post(
                f"{self.tool_base}/execute",
                json=request_payload,
                timeout=self.request_timeout_sec,
            )
            status_code = response.status_code
            try:
                payload = response.json()
            except Exception:
                payload = {
                    "ok": False,
                    "tool_name": self.tool_name,
                    "result": None,
                    "error": {
                        "code": "invalid_json_response",
                        "message": response.text[:1000],
                    },
                    "tool_call_id": tool_call_id,
                    "meta": {},
                }
        except Exception as exc:
            payload = {
                "ok": False,
                "tool_name": self.tool_name,
                "result": None,
                "error": {
                    "code": "tool_request_error",
                    "message": str(exc),
                },
                "tool_call_id": tool_call_id,
                "meta": {},
            }

        normalized = self._normalize_execute_payload(payload, tool_call_id=tool_call_id)
        if rewrite_info["rewrites"] or rewrite_info["errors"]:
            meta = normalized.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["image_resolution"] = rewrite_info
            normalized["meta"] = meta

        tool_response = self._convert_payload_to_tool_response(normalized)

        latency_ms = (time.perf_counter() - start) * 1000.0
        exec_ok = bool(normalized.get("ok"))
        tool_metrics = {
            "tool_name": self.tool_name,
            "exec_ok": 1.0 if exec_ok else 0.0,
            "latency_ms": float(latency_ms),
        }
        if status_code is not None:
            tool_metrics["http_status"] = int(status_code)

        self._maybe_log_metrics(tool_metrics)
        return tool_response, 0.0, tool_metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        del instance_id, kwargs
        return 0.0

    def _maybe_log_metrics(self, tool_metrics: dict[str, Any]) -> None:
        path = self.emit_metrics_log_path
        if not path:
            return
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(tool_metrics, ensure_ascii=False) + "\n")
        except Exception:
            # Metric logging failure should not block rollout.
            return

    def _normalize_execute_payload(self, payload: Any, tool_call_id: str) -> dict[str, Any]:
        if isinstance(payload, dict):
            ok = payload.get("ok")
            if isinstance(ok, bool):
                normalized = {
                    "ok": ok,
                    "tool_name": payload.get("tool_name") or self.tool_name,
                    "result": payload.get("result"),
                    "error": payload.get("error"),
                    "tool_call_id": payload.get("tool_call_id") or tool_call_id,
                    "meta": payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
                }
                if not normalized["ok"] and not isinstance(normalized["error"], dict):
                    normalized["error"] = {
                        "code": "tool_execution_error",
                        "message": str(normalized["error"] or "tool failed"),
                    }
                return normalized

            # Fallback: unwrapped payload from server.
            return {
                "ok": False,
                "tool_name": self.tool_name,
                "result": None,
                "error": {
                    "code": "invalid_execute_envelope",
                    "message": "Tool server response missing boolean 'ok'.",
                    "detail": payload,
                },
                "tool_call_id": tool_call_id,
                "meta": {},
            }

        return {
            "ok": False,
            "tool_name": self.tool_name,
            "result": None,
            "error": {
                "code": "invalid_execute_response_type",
                "message": f"Unexpected tool response type: {type(payload).__name__}",
            },
            "tool_call_id": tool_call_id,
            "meta": {},
        }

    def _convert_payload_to_tool_response(self, payload: dict[str, Any]) -> ToolResponse:
        if not payload.get("ok"):
            error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            code = str(error.get("code") or "tool_execution_error")
            message = str(error.get("message") or "Tool execution failed")
            text = f"Error: {code}: {message}"
            return ToolResponse(text=text)

        result = payload.get("result")
        images, compact_result = self._extract_images_and_scrub(result)

        text_payload = {
            "ok": True,
            "result": compact_result,
            "tool_name": payload.get("tool_name"),
            "tool_call_id": payload.get("tool_call_id"),
        }
        if payload.get("meta"):
            text_payload["meta"] = payload["meta"]

        try:
            text = json.dumps(text_payload, ensure_ascii=False)
        except Exception:
            text = str(text_payload)

        if images:
            return ToolResponse(text=text, image=images)
        return ToolResponse(text=text)

    def _rewrite_image_path_refs(self, args: Any, agent_data: Any) -> dict[str, Any]:
        rewrites = []
        errors = []

        def _walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                # Recurse first for nested payloads.
                for key, value in list(node.items()):
                    if isinstance(value, (dict, list)):
                        _walk(value, f"{path}.{key}")

                for key in list(node.keys()):
                    if not self._is_image_path_key(key):
                        continue
                    raw = node.get(key)
                    if not isinstance(raw, str):
                        continue

                    image = self._resolve_image_ref(raw, agent_data)
                    if image is None:
                        # keep absolute files untouched; server may access local paths.
                        if os.path.isfile(raw):
                            continue
                        errors.append({
                            "field": f"{path}.{key}",
                            "value": raw,
                            "reason": "unresolved_image_reference",
                        })
                        continue

                    base64_key = f"{key[:-5]}_base64" if key.endswith("_path") else f"{key}_base64"
                    node[base64_key] = self._pil_to_data_url(image)
                    node.pop(key, None)
                    rewrites.append(
                        {
                            "field": f"{path}.{key}",
                            "target_field": f"{path}.{base64_key}",
                            "original": raw,
                        }
                    )

            elif isinstance(node, list):
                for idx, item in enumerate(node):
                    if isinstance(item, (dict, list)):
                        _walk(item, f"{path}[{idx}]")

        if isinstance(args, dict):
            _walk(args, "args")
        return {"rewrites": rewrites, "errors": errors}

    def _inject_default_image_if_needed(self, args: dict[str, Any], agent_data: Any) -> None:
        if not isinstance(args, dict):
            return
        if self._has_image_input(args):
            return
        if not self._schema_accepts_image():
            return

        image = self._resolve_image_ref("image_1", agent_data)
        if image is None and agent_data is not None and getattr(agent_data, "image_data", None):
            image = agent_data.image_data[0]
        if image is None:
            return
        args["image_base64"] = self._pil_to_data_url(image)

    def _schema_accepts_image(self) -> bool:
        schema = self.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
        function = schema.get("function") if isinstance(schema, dict) else None
        parameters = function.get("parameters") if isinstance(function, dict) else None
        properties = parameters.get("properties") if isinstance(parameters, dict) else {}
        if not isinstance(properties, dict):
            return False
        keys = {str(k) for k in properties.keys()}
        return any("image" in key and (key.endswith("_path") or key.endswith("_base64")) for key in keys)

    def _has_image_input(self, args: Any) -> bool:
        if isinstance(args, dict):
            for key, value in args.items():
                key_l = str(key).lower()
                if "image" in key_l and (key_l.endswith("_path") or key_l.endswith("_base64")) and value:
                    return True
                if self._has_image_input(value):
                    return True
        elif isinstance(args, list):
            return any(self._has_image_input(item) for item in args)
        return False

    def _resolve_image_ref(self, value: str, agent_data: Any) -> Optional[Image.Image]:
        if not isinstance(value, str):
            return None
        match = _IMAGE_REF_PATTERN.match(value.strip())
        if not match:
            return None
        idx = int(match.group(1)) - 1
        if idx < 0:
            return None

        if agent_data is None:
            return None
        images = getattr(agent_data, "image_data", None)
        if not isinstance(images, list):
            return None
        if idx >= len(images):
            return None
        img = images[idx]
        if isinstance(img, Image.Image):
            return img
        return None

    def _is_image_path_key(self, key: str) -> bool:
        key = str(key).lower()
        return key.endswith("_path") and "image" in key

    def _pil_to_data_url(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _extract_images_and_scrub(self, value: Any) -> tuple[list[Image.Image], Any]:
        images: list[Image.Image] = []

        def _walk(node: Any, key_hint: str = "") -> Any:
            if isinstance(node, dict):
                compact = {}
                for key, item in node.items():
                    compact_item = _walk(item, key_hint=str(key))
                    if compact_item is not _Drop:
                        compact[key] = compact_item
                return compact

            if isinstance(node, list):
                compact_list = []
                for item in node:
                    compact_item = _walk(item, key_hint=key_hint)
                    if compact_item is not _Drop:
                        compact_list.append(compact_item)
                return compact_list

            if isinstance(node, str):
                maybe_image = self._decode_image_string(node, key_hint)
                if maybe_image is not None:
                    images.append(maybe_image)
                    return _Drop

            return node

        compact = _walk(copy.deepcopy(value), key_hint="result")
        if compact is _Drop:
            compact = None
        return images, compact

    def _decode_image_string(self, value: str, key_hint: str = "") -> Optional[Image.Image]:
        text = value.strip()
        key_hint = str(key_hint).lower()

        is_data_url = text.startswith("data:image/")
        is_key_hint = key_hint in _IMAGE_RESULT_HINTS or key_hint.endswith("_base64") or key_hint.endswith("_b64")
        if not (is_data_url or is_key_hint):
            return None

        raw = text
        if is_data_url:
            if "," not in text:
                return None
            raw = text.split(",", 1)[1]
        elif raw.startswith("base64:"):
            raw = raw.split(":", 1)[1]

        try:
            data = base64.b64decode(raw, validate=True)
        except Exception:
            return None

        try:
            image = Image.open(io.BytesIO(data))
            return image.convert("RGB")
        except Exception:
            return None


class _DropType:
    pass


_Drop = _DropType()
