from mini_vlm_tool_kit.tools.base import BaseTool

TOOL_NAME = "Python_Code_Generator_Tool"


class Python_Coder_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Placeholder for AgentFlow Python Coder tool.",
            tool_version="0.0.0",
            input_types={"query": "str"},
            output_type="dict",
        )

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Python_Coder_Tool is a placeholder.")
