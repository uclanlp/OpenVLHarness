"""Load the agent-system configuration and wire explicit runtime dependencies."""
import copy
import importlib
import os
import re
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse
import yaml

from .memory.store import MemoryStore
from .services.models import ModelAdapter
from .services.client import ToolClient

# Explicit allowlist; the final registry is sorted by public name like the old router.
PAPER_CAPABILITIES = ("grounding", "ocr", "depth", "camera_pose", "zoom",
                      "text_search", "image_search", "web_visit", "coding")
PAPER_BACKENDS = dict(zip(PAPER_CAPABILITIES, (
    "sam3_segment", "ocr", "depth_estimation", "camera_trajectory", "zoom_in_image_by_bbox",
    "serper_text_search", "serper_image_search", "serper_webpage_visit", "python_coding_agent")))
RAW_BACKENDS = {"sam3_segment", "ocr", "depth_estimation", "camera_trajectory", "grounding_dino"}


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        def env(match):
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Missing environment variable {name} required by agent configuration")
            return os.environ[name]
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", env, value)
    return value


def load_agent_config(config):
    if isinstance(config, (str, Path)):
        with open(config) as stream:
            config = yaml.safe_load(stream)
    config = _expand(copy.deepcopy(config))
    if not isinstance(config, dict) or config.get("agent_name") != "MainAgent":
        raise ValueError("Use the OpenVLHarness agent-system format with agent_name: MainAgent")
    allowed = {"agent_name", "agent_config_version", "agent", "memory", "capabilities", "services"}
    if set(config) - allowed:
        raise ValueError(f"Unknown agent-system sections: {sorted(set(config) - allowed)}")
    for section in ("agent", "memory", "capabilities", "services"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
    agent = config["agent"]
    if set(agent) - {"model_service", "max_iterations", "max_context_length", "tool_choice", "paired_sampling_seed"}:
        raise ValueError("Unknown agent options")
    if type(agent.get("max_iterations", 15)) is not int or agent.get("max_iterations", 15) <= 0:
        raise ValueError("agent.max_iterations must be a positive integer")
    if type(agent.get("paired_sampling_seed", False)) is not bool:
        raise ValueError("agent.paired_sampling_seed must be boolean")
    caps = config["capabilities"]
    if set(caps) - {"registries", "tools"}:
        raise ValueError("Unknown capability options")
    groups = caps.get("registries", [])
    if not isinstance(groups, list) or "paper" not in groups or set(groups) - {"paper", "extensions"}:
        raise ValueError("capabilities.registries must include paper; extensions explicitly enables DINO")
    selected = caps.get("tools")
    if not isinstance(selected, dict) or set(selected) - set(PAPER_CAPABILITIES):
        raise ValueError("capabilities.tools must map flat paper capability names to implementations")
    services = config["services"]
    if set(services) - {"client", "server", "tools", "models"}:
        raise ValueError("Unknown service options")
    for name in ("client", "server", "tools", "models"):
        if not isinstance(services.get(name), dict):
            raise ValueError(f"services.{name} must be a mapping")
    client = services["client"]
    if set(client) - {"base_url", "timeout_seconds", "retry"} or not client.get("base_url"):
        raise ValueError("services.client requires base_url and accepts timeout_seconds/retry")
    ToolClient(**client)
    server = services["server"]
    if set(server) - {"host", "port", "registration", "limits"}:
        raise ValueError("Unknown tool server options")
    if server.get("registration", "configured_only") != "configured_only":
        raise ValueError("services.server.registration must be configured_only")
    if type(server.get("port", 8101)) is not int or not 0 < server.get("port", 8101) < 65536:
        raise ValueError("services.server.port must be an integer in 1..65535")
    for name, settings in selected.items():
        if not isinstance(settings, dict):
            raise ValueError(f"capabilities.tools.{name} must be a mapping")
        if name == "coding":
            if set(settings) != {"generator", "executor"} or settings != {
                "generator": "python_coding_agent", "executor": "python_executor_local"}:
                raise ValueError("coding requires python_coding_agent generation and python_executor_local execution")
            backend = settings["generator"]
        else:
            if set(settings) != {"backend"}:
                raise ValueError(f"{name} accepts only backend selection; task/tuning values do not belong here")
            backend = settings["backend"]
            if backend != PAPER_BACKENDS[name] and not (name == "grounding" and backend == "grounding_dino" and "extensions" in groups):
                raise ValueError(f"Unsupported {name} backend {backend!r}; no automatic fallback")
        service = services["tools"].get(backend)
        if not isinstance(service, dict):
            raise ValueError(f"Missing services.tools.{backend} configuration")
        allowed_settings = {"server_url"} if backend in RAW_BACKENDS else (
            {"model_service", "context"} if backend == "python_coding_agent" else
            {"model_service"} if backend.startswith("serper_") else set())
        if set(service) - allowed_settings:
            raise ValueError(f"Unsupported settings for {backend}; algorithmic controls are service constants")
        if backend in RAW_BACKENDS and not service.get("server_url"):
            raise ValueError(f"services.tools.{backend}.server_url is required")
        if backend in RAW_BACKENDS:
            url = urlparse(service["server_url"])
            if (url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment
                    or url.path.rstrip("/").endswith(("/v1", "/v1/depth", "/v1/pose", "/v1/segment",
                        "/v1/grounding_dino", "/chat/completions", "/responses", "/execute"))):
                raise ValueError(f"services.tools.{backend}.server_url must be an HTTP(S) service base, without an endpoint suffix")
        if "model_service" in allowed_settings:
            model_service = service.get("model_service", "auxiliary" if backend.startswith("serper_") else agent["model_service"])
            if model_service not in services["models"]:
                raise ValueError(f"Missing model service {model_service!r} for {backend}")
        if backend == "python_coding_agent" and service.get("context", "fresh") != "fresh":
            raise ValueError("Coding generation requires context: fresh")
    if agent.get("model_service") not in services["models"]:
        raise ValueError("agent.model_service must name an entry in services.models")
    # Resolve every model/options before any inference or backend import.
    models = {name: ModelAdapter(model) for name, model in services["models"].items()}
    if "coding" in selected:
        coder_name = services["tools"]["python_coding_agent"].get("model_service", agent["model_service"])
        coder, orchestrator = models[coder_name], models[agent["model_service"]]
        if (coder.model, coder.protocol) != (orchestrator.model, orchestrator.protocol):
            raise ValueError("Coding must use the same backbone and native protocol as the orchestrator")
        executor = services["tools"].get("python_executor_local", {})
        if executor.get("transport", "local") != "local" or set(executor) - {"transport", "timeout_seconds"}:
            raise ValueError("The paper coding executor supports local transport only")
        timeout = executor.get("timeout_seconds", 30)
        if type(timeout) not in (int, float) or timeout <= 0:
            raise ValueError("Local executor timeout_seconds must be positive")
    memory_factory(config["memory"])()
    return config


def memory_factory(config):
    if set(config) - {"modules", "emit_policy"}:
        raise ValueError("Unknown memory options")
    entries = config.get("modules")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) or "name" not in entry for entry in entries):
        raise ValueError("memory.modules must be a list of named mappings")
    limit = 20
    for entry in entries:
        if set(entry) - ({"name", "webpage_env_update_limit"} if entry["name"] == "url" else {"name"}):
            raise ValueError(f"Unknown options for memory module {entry['name']}")
        if entry["name"] == "url":
            limit = entry.get("webpage_env_update_limit", 20)
    return lambda: MemoryStore(modules=[entry["name"] for entry in entries],
        emit_policy=config.get("emit_policy", "always"), webpage_env_update_limit=limit)


def build_capabilities(config):
    selected = config["capabilities"]["tools"]
    services = config["services"]
    result = {}
    for name in PAPER_CAPABILITIES:
        if name not in selected:
            continue
        cls = importlib.import_module(f"openvlharness.capabilities.{name}").Capability
        backend = selected[name].get("backend", selected[name].get("generator"))
        settings = dict(services["tools"][backend])
        if name == "coding":
            executor = services["tools"].get("python_executor_local", {})
            if executor.get("transport", "local") != "local" or set(executor) - {"transport", "timeout_seconds"}:
                raise ValueError("The paper coding executor supports local transport only")
            settings["execution_timeout_seconds"] = executor.get("timeout_seconds", 30)
        capability = cls(backend=backend, client_config=services["client"], service_config=settings)
        result[capability.name] = capability
    return dict(sorted(result.items()))


def resolved_model_id(config):
    config = load_agent_config(config)
    model = config["services"]["models"][config["agent"]["model_service"]]
    return ModelAdapter(model).model


def build_agent(config):
    from .agent.agent import MainAgent
    config = load_agent_config(config)
    options = config["agent"]
    adapter = ModelAdapter(config["services"]["models"][options["model_service"]],
                           max_context_length=options.get("max_context_length"))
    adapter.load_tokenizer()
    resources = files("openvlharness.prompts")
    system = resources.joinpath(adapter.profile["prompt_bundle"] + "_system.txt").read_text()
    prefix = resources.joinpath("user_prefix.txt").read_text()
    return MainAgent(adapter, build_capabilities(config), memory_factory(config["memory"]),
        system, prefix, max_iterations=options.get("max_iterations", 15),
        tool_choice=options.get("tool_choice", "auto"), paired_sampling_seed=options.get("paired_sampling_seed", False))
