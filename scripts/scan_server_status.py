#!/usr/bin/env python3
import json
import socket
import urllib.request


HOST = "127.0.0.1"
START_PORT = 8000
END_PORT = 9000
CONNECT_TIMEOUT_SEC = 0.15
HTTP_TIMEOUT_SEC = 0.4


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


def _tool_names(host, port):
    tools = _get_json(host, port, "/tools")
    if not isinstance(tools, dict):
        return None
    return sorted(tools.keys())


def _identify_subtool(host, port):
    health = _get_json(host, port, "/health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        return None

    if "languages" in health:
        return "OCR", health
    if "colormap_default" in health:
        return "Depth Anything", health
    if "model_id" in health and "device" in health:
        return "Grounding DINO", health

    models = _get_json(host, port, "/v1/models")
    if isinstance(models, dict):
        for item in models.get("data", []):
            if isinstance(item, dict) and item.get("id") == "sam3-image":
                return "SAM3 Segmentation", health

    return "Unknown (health ok)", health


def _format_tools_line(tool_names):
    if not tool_names:
        return "(no tools reported)"
    return ", ".join(tool_names)


def _format_subtool_details(name, details):
    if name == "OCR":
        languages = details.get("languages")
        gpu = details.get("gpu")
        parts = []
        if languages is not None:
            if isinstance(languages, list):
                parts.append("languages: " + ", ".join(str(item) for item in languages))
            else:
                parts.append(f"languages: {languages}")
        if gpu is not None:
            parts.append(f"gpu: {gpu}")
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

    return ""


def main():
    tool_servers = []
    subtool_servers = []

    for port in range(START_PORT, END_PORT + 1):
        if not _port_open(HOST, port):
            continue

        tool_names = _tool_names(HOST, port)
        if tool_names is not None:
            tool_servers.append((port, tool_names))

        subtool = _identify_subtool(HOST, port)
        if subtool:
            name, details = subtool
            subtool_servers.append((port, name, details))

    print(f"Tool Server Status (ports {START_PORT}-{END_PORT})")
    if tool_servers:
        for port, tool_names in tool_servers:
            tools_line = _format_tools_line(tool_names)
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
