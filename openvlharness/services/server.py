"""Configured tool router: python -m openvlharness.services.server --config ..."""
import argparse
import asyncio
import importlib
import sys

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from ..bootstrap import load_agent_config, build_capabilities, RAW_BACKENDS
from ..contracts import ServiceResult
from . import degradation
from .models import ModelAdapter, AuxiliaryModel
from .tools.base import ToolArgumentError
from .envelope import _normalize_result_payload, _build_error_payload, _build_success_payload

BACKEND_GROUPS = {
    "sam3_segment": "sam3", "grounding_dino": "dino", "ocr": "ocr",
    "depth_estimation": "depth", "camera_trajectory": "depth",
    "serper_text_search": "search", "serper_image_search": "search",
    "serper_webpage_visit": "webpage", "python_coding_agent": "coding",
}
DEFAULT_LIMITS = {"depth": 1, "sam3": 1, "dino": 2, "ocr": 10,
                  "webpage": 8, "search": 4, "coding": 10}


class ExecuteRequest(BaseModel):
    """Validated router request containing tool_name, tool_call_id and task args.

    Unknown top-level fields are rejected; args defaults to an empty mapping.
    """
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    tool_call_id: str
    args: dict = Field(default_factory=dict)


def redact(value):
    """Copy nested settings, masking keys that indicate credentials or secrets."""
    if isinstance(value, dict):
        return {key: "<redacted>" if any(word in key.lower() for word in ("key", "secret", "password", "token"))
                and key not in {"max_tokens", "return_token_ids"} else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def create_app(config, *, implementations=None):
    """Construct the configured FastAPI tool router and backend concurrency limits.

    Args:
        config: Agent-system configuration mapping or YAML path.
        implementations: Optional mapping from backend names to tool instances.
            When omitted, only selected backend modules are imported and built.

    Returns:
        A FastAPI application exposing execution, tool schemas, redacted
        environment settings and health/status endpoints.

    Raises:
        ValueError: Configuration or backend concurrency limits are invalid.
        KeyError: An injected implementation is missing for a selected backend.

    Construction does not start an HTTP server or model worker. Tool execution
    runs in a thread pool after acquiring the appropriate backend semaphore.
    """
    config = load_agent_config(config)
    capabilities = build_capabilities(config)
    settings = config["services"]
    limits = {**DEFAULT_LIMITS, **settings["server"].get("limits", {})}
    if set(limits) != set(DEFAULT_LIMITS) or any(type(n) is not int or n <= 0 for n in limits.values()):
        raise ValueError("Server limits must name known backend groups and use positive integers")
    registry = {}
    for cap in capabilities.values():
        service = settings["tools"][cap.backend]
        if implementations is not None:
            tool = implementations[cap.backend]
        else:
            module = importlib.import_module(f"openvlharness.services.tools.{cap.backend}.tool")
            cls = getattr(module, cap.concrete_name)
            if cap.backend.startswith("serper_"):
                model_name = service.get("model_service", "auxiliary")
                tool = cls(llm_engine=AuxiliaryModel(settings["models"][model_name]))
            elif cap.backend == "python_coding_agent":
                model_name = service.get("model_service", config["agent"]["model_service"])
                # Code-generation requests use a dedicated 600-second HTTP timeout.
                tool = cls(model_adapter=ModelAdapter({**settings["models"][model_name], "timeout": 600}))
            else:
                tool = cls()
        registry[cap.concrete_name] = (tool, cap.backend, service)
    semaphores = {name: asyncio.Semaphore(limit) for name, limit in limits.items()}
    active = dict.fromkeys(limits, 0)
    app = FastAPI(title="OpenVLHarness tool router")
    app.state.registry = registry
    app.state.backend_groups = BACKEND_GROUPS
    app.state.semaphores = semaphores

    @app.get("/health")
    async def health():
        """Report router readiness and the number of registered tools."""
        return {"status": "ok", "tools": len(registry)}

    @app.get("/tools")
    async def tools():
        """List configured concrete tool names and their backend identifiers."""
        return {name: {"tool_name": name, "backend": backend} for name, (_, backend, _) in registry.items()}

    @app.get("/llm-schema")
    async def schemas():
        """Return public function schemas for the configured capabilities."""
        return [cap.public_schema() for cap in capabilities.values()]

    @app.get("/env")
    async def environment():
        """Return redacted service settings and fixed search/extraction policy."""
        return {"config": redact(settings), "scope": "configured services; credentials redacted",
                "search": {"summarize": True, "multimodal": True, "num_results": 10,
                           "keep_k": 3, "max_tokens": 4096, "thinking": False},
                "webpage": {"max_urls": 5, "max_tokens": 16384}}

    @app.get("/backend-status")
    async def status():
        """Report configured concurrency limits and in-flight counts by backend group."""
        return {name: {"limit": limit, "in_flight": active[name]} for name, limit in limits.items()}

    def execute(request):
        """Execute one resolved request and return a JSON envelope with its HTTP status."""
        tool, backend, service = registry[request.tool_name]
        args = dict(request.args)
        try:
            # Only resolved task data crosses this boundary; endpoints come from deployment config,
            # and images arrive as base64 so callers cannot make tools open router-host files.
            forbidden = {"server_url", "timeout_seconds", "llm_model", "llm_base_url", "llm_api_key",
                         "box_threshold", "text_threshold", "num_results", "gl", "hl",
                         "image_path", "image_paths"} & set(args)
            if forbidden:
                raise ToolArgumentError(f"Runtime/service fields cannot be supplied as task arguments: {sorted(forbidden)}")
            if backend in RAW_BACKENDS:
                args["server_url"] = service["server_url"]
            result = tool.execute(**args)
            extra = {}
            if isinstance(result, ServiceResult):
                extra, result = result.metadata, result.value
            result, meta = _normalize_result_payload(result)
            meta.update(extra)
            return _build_success_payload(request.tool_name, request.tool_call_id, result, meta), 200
        except Exception as exc:
            # Invalid caller arguments are client errors; everything else is a backend failure.
            status = 400 if isinstance(exc, ToolArgumentError) else 500
            return _build_error_payload(request.tool_name, request.tool_call_id, "tool_execution_error",
                str(exc), {"exception_type": type(exc).__name__}), status

    @app.post("/execute")
    async def call(request: ExecuteRequest):
        """Dispatch an HTTP tool request under its shared backend concurrency limit."""
        if request.tool_name not in registry:
            return JSONResponse(status_code=404, content=_build_error_payload(request.tool_name,
                request.tool_call_id, "tool_not_found", f"Tool not found: {request.tool_name}"))
        backend = registry[request.tool_name][1]
        group = BACKEND_GROUPS.get(backend)
        if group:
            # Acquire before occupying a threadpool worker, including shared depth/camera capacity.
            async with semaphores[group]:
                active[group] += 1
                try:
                    payload, code = await run_in_threadpool(execute, request)
                finally:
                    active[group] -= 1
        else:
            payload, code = await run_in_threadpool(execute, request)
        return JSONResponse(status_code=code, content=payload)

    return app


def main():
    """Load the CLI configuration and serve the router with Uvicorn."""
    parser = argparse.ArgumentParser(prog="python -m openvlharness.services.server", description=__doc__)
    parser.add_argument("--config", required=True, help="Agent-system YAML; use the same file as the agent")
    args = parser.parse_args()
    try:
        config = load_agent_config(args.config)
    except (ValueError, OSError) as exc:
        sys.exit(f"error: {exc}")
    settings = config["services"]["server"]
    import uvicorn
    uvicorn.run(create_app(config), host=settings.get("host", "127.0.0.1"), port=settings.get("port", 8101))


if __name__ == "__main__":
    main()
