import requests


class ToolClient:
    def __init__(self, base_url):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")

    def list_tools(self):
        response = requests.get(f"{self.base_url}/tools", timeout=30)
        response.raise_for_status()
        return response.json()

    def execute(self, tool_name, args):
        payload = {"tool_name": tool_name, "args": args}
        response = requests.post(f"{self.base_url}/execute", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
