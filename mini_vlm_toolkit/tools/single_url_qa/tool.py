from mini_vlm_toolkit.tools.base import BaseTool

TOOL_NAME = "Single_URL_QA_Tool"


class Single_URL_QA_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Placeholder for single-URL QA tool.",
            tool_version="0.0.0",
            input_types={"query": "str", "url": "str"},
            output_type="str",
        )

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Single_URL_QA_Tool is a placeholder.")
