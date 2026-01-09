from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "Ground_Google_Search_Tool"


class Google_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Placeholder for AgentFlow Google Search tool.",
            tool_version="0.0.0",
            input_types={"query": "str", "add_citations": "bool"},
            output_type="str",
        )

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Google_Search_Tool is a placeholder.")
