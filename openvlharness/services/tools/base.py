"""Shared metadata and execution interface for concrete tool backends."""
class ToolArgumentError(ValueError):
    """Invalid task arguments supplied by the caller; the router reports HTTP 400."""


class BaseTool:
    """Base interface for backend metadata and execution."""
    require_llm_engine = False

    def __init__(
        self,
        tool_name,
        tool_description,
        tool_version,
        input_types,
        output_type,
        input_schema=None,
        tool_aliases=None,
        demo_commands=None,
        user_metadata=None,
    ):
        """Store a tool's descriptive metadata without starting execution.

        Args:
            tool_name: Concrete name used by the service router.
            tool_description: Human-readable purpose and usage guidance.
            tool_version: Backend metadata version.
            input_types: Descriptions of accepted input fields.
            output_type: Description of the backend result.
            input_schema: Optional structured argument schema.
            tool_aliases: Optional alternate names.
            demo_commands: Optional example invocations.
            user_metadata: Optional additional metadata mapping.
        """
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_version = tool_version
        self.input_types = input_types
        self.output_type = output_type
        self.input_schema = input_schema
        self.tool_aliases = list(tool_aliases or [])
        self.demo_commands = demo_commands or []
        self.user_metadata = user_metadata or {}

    def execute(self, *args, **kwargs):
        """Execute backend-specific arguments; subclasses must implement this method."""
        raise NotImplementedError("Tool must implement execute().")
