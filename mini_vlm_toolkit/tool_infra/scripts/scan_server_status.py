#!/usr/bin/env python3
import json
import socket
import urllib.request


HOST = "127.0.0.1"
START_PORT = 8000
END_PORT = 9000
CONNECT_TIMEOUT_SEC = 0.15
HTTP_TIMEOUT_SEC = 0.4
MAX_MODEL_DETAIL_ITEMS = 8
def _port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_SEC):
            return True
    except OSError:
        return False


def _get_json(host, port, path):
    url = f"http://{host}:{port}{path}"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SEC) as response:
            payload = response.read()
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None


def _openai_models_info(host, port):
    models = _get_json(host, port, "/v1/models")
    if not isinstance(models, dict):
        return None
    data = models.get("data")
    if not isinstance(data, list):
        return None

    model_ids = []
    owned_by = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if model_id:
            model_ids.append(str(model_id))
        owner = item.get("owned_by")
        if owner:
            owned_by.append(str(owner))

    if not model_ids:
        return None

    return {
        "model_ids": sorted(set(model_ids)),
        "owned_by": sorted(set(owned_by)),
    }


def _is_vllm_model_info(info):
    for owner in info.get("owned_by", []):
        if "vllm" in owner.lower():
            return True
    return False


def _tool_metadata(host, port):
    tools = _get_json(host, port, "/tools")
    if not isinstance(tools, dict):
        return None
    return tools


def _identify_subtool(host, port):
    health = _get_json(host, port, "/health")
    health_ok = isinstance(health, dict) and health.get("status") == "ok"

    if health_ok:
        if "languages" in health:
            return "OCR", health
        if "colormap_default" in health:
            return "Depth Anything", health
        if "aspect_ratio_default" in health:
            return "Qwen Image", health
        if "model_id" in health and "device" in health:
            if "paddle" in health.get("model_id", "").lower():
                return "OCR", health
            return "Grounding DINO", health

    models_info = _openai_models_info(host, port)
    if models_info:
        if "sam3-image" in models_info.get("model_ids", []):
            return "SAM3 Segmentation", health if health_ok else models_info
        if _is_vllm_model_info(models_info):
            return "vLLM", models_info

    if health_ok:
        return "Unknown (health ok)", health

    return None


def _format_tools_line(tool_names, llm_direct_suffix=None):
    if not tool_names:
        return "(no tools reported)"
    formatted = []
    for name in tool_names:
        if name == "LLM_Direct_Generator_Tool" and llm_direct_suffix:
            name = name + llm_direct_suffix
        formatted.append(name)
    return ", ".join(formatted)


def _format_list(items, limit=MAX_MODEL_DETAIL_ITEMS):
    if not items:
        return ""
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", +{len(items) - limit} more"


def _llm_direct_suffix_from_metadata(tool_metadata):
    if not isinstance(tool_metadata, dict):
        return None
    llm_meta = tool_metadata.get("LLM_Direct_Generator_Tool")
    if not isinstance(llm_meta, dict):
        return None
    description = llm_meta.get("tool_description", "")
    marker = "Configured models:"
    if marker not in description:
        return None
    tail = description.split(marker, 1)[1].strip()
    if not tail:
        return " (models: none)"
    if tail.startswith("(") and tail.endswith(")"):
        return f" (models: {tail[1:-1].strip()})"
    models = [item.strip() for item in tail.split(",") if item.strip()]
    if not models:
        return " (models: none)"
    return " (models: " + _format_list(models) + ")"


def _format_subtool_details(name, details):
    if name == "OCR":
        parts = []
        model_id = details.get("model_id")
        device = details.get("device")
        if model_id:
            parts.append(f"model_id: {model_id}")
        if device:
            parts.append(f"device: {device}")
        return " | " + " | ".join(parts) if parts else ""


    if name == "Depth Anything":
        parts = []
        model_id = details.get("model_id")
        device = details.get("device")
        colormap = details.get("colormap_default")
        if model_id:
            parts.append(f"model_id: {model_id}")
        if device:
            parts.append(f"device: {device}")
        if colormap:
            parts.append(f"colormap_default: {colormap}")
        return " | " + " | ".join(parts) if parts else ""

    if name == "Grounding DINO":
        parts = []
        model_id = details.get("model_id")
        device = details.get("device")
        if model_id:
            parts.append(f"model_id: {model_id}")
        if device:
            parts.append(f"device: {device}")
        return " | " + " | ".join(parts) if parts else ""

    if name == "Qwen Image":
        parts = []
        model_id = details.get("model_id")
        device = details.get("device")
        aspect = details.get("aspect_ratio_default")
        if model_id:
            parts.append(f"model_id: {model_id}")
        if device:
            parts.append(f"device: {device}")
        if aspect:
            parts.append(f"aspect_ratio_default: {aspect}")
        return " | " + " | ".join(parts) if parts else ""

    if name == "vLLM":
        parts = []
        model_ids = details.get("model_ids")
        owners = details.get("owned_by")
        if model_ids:
            parts.append("models: " + _format_list(model_ids))
        if owners:
            parts.append("owned_by: " + _format_list(owners))
        return " | " + " | ".join(parts) if parts else ""

    return ""


def main():
    tool_servers = []
    subtool_servers = []

    for port in range(START_PORT, END_PORT + 1):
        if not _port_open(HOST, port):
            continue

        tool_metadata = _tool_metadata(HOST, port)
        if tool_metadata is not None:
            tool_servers.append((port, tool_metadata))
            continue

        subtool = _identify_subtool(HOST, port)
        if subtool:
            name, details = subtool
            subtool_servers.append((port, name, details))

    print(f"Tool Server Status (ports {START_PORT}-{END_PORT})")
    if tool_servers:
        for port, tool_metadata in tool_servers:
            tool_names = sorted(tool_metadata.keys())
            llm_direct_suffix = _llm_direct_suffix_from_metadata(tool_metadata)
            tools_line = _format_tools_line(tool_names, llm_direct_suffix)
            print(f"- http://{HOST}:{port} | tools: {tools_line}")
    else:
        print("- None found.")

    print("")
    print(f"Subtool Server Status (ports {START_PORT}-{END_PORT})")
    print("- Note: subtool ports are local to this machine. A subtool can still be invoked via the tool server if it appears in the server tool list, even when its subtool port is not visible here (the server may be forwarded from elsewhere).")
    if subtool_servers:
        for port, name, details in subtool_servers:
            extra = _format_subtool_details(name, details)
            print(f"- {name} | http://{HOST}:{port}{extra}")
    else:
        print("- None found.")


if __name__ == "__main__":
    main()
