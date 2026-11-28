"""Expose one generated tool to the agent under the schema the tool declares."""
from ..capabilities._base import BaseCapability
from .catalog import json_schema


class Capability(BaseCapability):
    """Model-facing capability for a generated tool executed by the tool router."""

    def __init__(self, tool, *, client_config, client=None):
        """Advertise the tool's own name, description and input schema.

        Args:
            tool: GeneratedTool instance providing the public metadata.
            client_config: ToolClient connection settings.
            client: Optional injected transport with an execute method.
        """
        self.name = tool.tool_name
        self.backends = {"toolsmith": tool.tool_name}
        self.schema = {"type": "function", "function": {
            "name": tool.tool_name, "description": tool.tool_description,
            "parameters": json_schema(tool.input_schema)}}  # tuples as lists, as the catalog checked it
        super().__init__(backend="toolsmith", client_config=client_config, client=client)

    @staticmethod
    def process_output(payload, prepared):
        """Leave rendering to the shared evidence format."""

    def normalize_result(self, call, result, prepared, binding_refs):
        """Return the result as the model sees it, with images replaced by memory IDs."""
        return {"result": prepared["compact_payload"]["result"], "images": prepared["image_refs"]}
