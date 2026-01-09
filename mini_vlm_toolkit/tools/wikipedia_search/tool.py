from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "Wikipedia_RAG_Search_Tool"


class Wikipedia_Search_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Placeholder for AgentFlow Wikipedia RAG tool.",
            tool_version="0.0.0",
            input_types={"query": "str"},
            output_type="dict",
        )

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Wikipedia_Search_Tool is a placeholder.")
