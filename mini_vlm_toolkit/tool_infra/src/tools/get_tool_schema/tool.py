from tools.base import BaseTool

TOOL_NAME = "Get_Tool_Schema"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {
            "type": "string",
            "description": "The name of the tool to retrieve the schema for.",
        },
    },
    "required": ["tool_name"],
}


class Get_Tool_Schema(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Get the full parameter schema, usage guide, and examples for a tool. "
                "Call this before using any other tool for the first time."
            ),
            tool_version="0.1.0",
            input_types={"tool_name": "str - name of the tool to look up"},
            input_schema=INPUT_SCHEMA,
            output_type="str - full schema text for the requested tool",
        )
        self._load_fn = None  # injected post-load by server

    def configure(self, load_fn):
        """Called by the server after startup to inject the schema loader."""
        self._load_fn = load_fn

    def execute(self, tool_name: str):
        if self._load_fn is None:
            return (
                f"# {tool_name}\n"
                "Get_Tool_Schema not configured. "
                "Server must call configure() at startup."
            )
        return self._load_fn(tool_name)
