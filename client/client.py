import requests

from mini_vlm_tool_kit.client.parser import parse_tool_execute_args


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

    def execute_direct(self, tool_instance, args):
        return tool_instance.execute(**args)

    def execute_with_llm(self, tool_name, tool_metadata, user_request, llm_engine):
        prompt = _build_tool_call_prompt(tool_name, tool_metadata, user_request)
        llm_output = llm_engine.generate(prompt)
        args = parse_tool_execute_args(llm_output)
        return self.execute(tool_name, args)


def _build_tool_call_prompt(tool_name, tool_metadata, user_request):
    return (
        "Generate a single Python code block that calls this tool.\n"
        "The code must include exactly one line: execution = tool.execute(...).\n"
        f"Tool name: {tool_name}\n"
        f"Tool metadata: {tool_metadata}\n"
        f"User request: {user_request}\n"
        "Return only the code block."
    )
