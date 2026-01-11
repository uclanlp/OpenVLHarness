from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "Serper_Google_Search_Tool"


class Serper_Google_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Placeholder for Serper-backed Google search tool that returns URLs."
            ),
            tool_version="0.0.0",
            input_types={"query": "str", "num_results": "int"},
            output_type="list[str]",
        )

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Serper_Google_Search_Tool is a placeholder.")
