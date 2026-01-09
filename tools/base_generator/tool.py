from mini_vlm_tool_kit.tools.base import BaseTool

TOOL_NAME = "Generalist_Solution_Generator_Tool"


class Base_Generator_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Placeholder for AgentFlow Base Generator tool.",
            tool_version="0.0.0",
            input_types={"query": "str"},
            output_type="str",
        )

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Base_Generator_Tool is a placeholder.")
