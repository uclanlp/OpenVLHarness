class BaseTool:
    require_llm_engine = False

    def __init__(
        self,
        tool_name,
        tool_description,
        tool_version,
        input_types,
        output_type,
        demo_commands=None,
        user_metadata=None,
    ):
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_version = tool_version
        self.input_types = input_types
        self.output_type = output_type
        self.demo_commands = demo_commands or []
        self.user_metadata = user_metadata or {}

    def get_metadata(self):
        return {
            "tool_name": self.tool_name,
            "tool_description": self.tool_description,
            "tool_version": self.tool_version,
            "input_types": self.input_types,
            "output_type": self.output_type,
            "demo_commands": self.demo_commands,
            "user_metadata": self.user_metadata,
            "require_llm_engine": self.require_llm_engine,
        }

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Tool must implement execute().")
