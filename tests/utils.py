import requests


def ensure_tool_server(server_url, required_tools=None, timeout_seconds=5):
    if not server_url:
        print("TOOL_SERVER_URL is not set; skipping tests.")
        return False
    endpoint = f"{server_url.rstrip('/')}/tools"
    try:
        response = requests.get(endpoint, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"Tool server unavailable at {endpoint}: {exc}")
        return False
    if not isinstance(payload, dict):
        print(f"Tool server returned unexpected payload from {endpoint}")
        return False
    missing = [tool for tool in (required_tools or []) if tool not in payload]
    if missing:
        missing_list = ", ".join(missing)
        print(f"Tool server missing required tools: {missing_list}. Skipping tests.")
        return False
    return True


def ensure_sam3_server(server_url, timeout_seconds=5):
    if not server_url:
        print("SAM3_SERVER_URL is not set; skipping SAM3 tests.")
        return False
    endpoint = f"{server_url.rstrip('/')}/health"
    try:
        response = requests.get(endpoint, timeout=timeout_seconds)
    except Exception as exc:
        print(f"SAM3 server unavailable at {endpoint}: {exc}")
        return False
    if response.status_code != 200:
        print(
            f"SAM3 server unhealthy at {endpoint}: status {response.status_code}"
        )
        return False
    return True
