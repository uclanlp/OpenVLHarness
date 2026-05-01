"""Client for the Visual_Grounding_Tool routed through the tool server.

The tool server at `tool_base` (default http://127.0.0.1:8102) is the same
endpoint the agent uses (configured as `tool_base` in the agent YAML, e.g.
configs/bryan-testing/v3_instruct.yaml). It exposes `/execute` with the
contract defined in mini_vlm_toolkit/tool_infra/src/server/app.py:499:
    POST /execute  body: {"tool_name": str, "args": dict, "tool_call_id": str}
    response envelope: {"ok": bool, "tool_name": ..., "result": <tool out>,
                        "error": {...}, "tool_call_id": ..., "meta": {...}}

The tool server may be SSH-forwarded from another machine, so we must send
the image as base64 (not a local filesystem path) — same trick the agent's
ImageStateMemory uses (mini_vlm_toolkit/memory/image_state_memory.py:639).
"""

import base64
import mimetypes
import os
import time
import uuid

import requests

DEFAULT_TOOL_BASE = "http://127.0.0.1:8102"
TOOL_NAME = "Visual_Grounding_Tool"


def _file_to_data_url(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class SAM3Client:
    def __init__(
        self,
        tool_base: str | None = None,
        timeout_seconds: int = 300,
        retries: int = 3,
        backoff: float = 2.0,
    ):
        self.tool_base = (tool_base or os.getenv("TOOL_BASE", DEFAULT_TOOL_BASE)).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff = backoff

    def health_check(self) -> None:
        resp = requests.get(f"{self.tool_base}/health", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        tools = body.get("tools") or []
        if TOOL_NAME not in tools and TOOL_NAME.lower() not in [t.lower() for t in tools]:
            raise RuntimeError(
                f"{TOOL_NAME} not registered on tool server {self.tool_base}. "
                f"Available tools: {sorted(tools)[:8]}..."
            )

    def segment(self, prompt: str, image_path: str) -> dict:
        # Encode locally — the tool server may be on a remote host that
        # cannot read this filesystem path.
        data_url = _file_to_data_url(image_path)
        body = {
            "tool_name": TOOL_NAME,
            "args": {
                "query": prompt,
                "image_base64": data_url,
            },
            "tool_call_id": f"sam3-baseline-{uuid.uuid4().hex[:12]}",
        }

        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    f"{self.tool_base}/execute",
                    json=body,
                    timeout=self.timeout_seconds,
                )
                envelope = resp.json()
                if envelope.get("ok"):
                    result = envelope.get("result")
                    if not isinstance(result, dict):
                        raise RuntimeError(f"Tool server returned ok=true but result is {type(result).__name__}")
                    return result
                err = envelope.get("error") or {}
                msg = err.get("message") or str(err) or f"HTTP {resp.status_code}"
                raise RuntimeError(f"Tool execute failed: {msg}")
            except Exception as exc:
                last_err = exc
                if attempt < self.retries - 1:
                    time.sleep(self.backoff * (attempt + 1))
        raise RuntimeError(f"SAM3 segment failed after {self.retries} attempts: {last_err}")
