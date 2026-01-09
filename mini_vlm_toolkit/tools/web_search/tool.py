from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "Web_RAG_Search_Tool"


class Web_Search_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Placeholder for AgentFlow Web RAG tool.",
            tool_version="0.0.0",
            input_types={"query": "str", "url": "str"},
            output_type="str",
        )

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Web_Search_Tool is a placeholder.")
