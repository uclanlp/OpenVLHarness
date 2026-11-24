"""The class every generated tool inherits."""
import uuid

from ..services.tools.base import BaseTool


class ToolCallError(RuntimeError):
    """A tool reached through the router reported a failure."""


class GeneratedTool(BaseTool):
    """A tool written as Python source and executed by the tool router.

    Subclasses pass tool_name, tool_description and a JSON-schema input_schema to
    __init__ and implement execute(**kwargs), returning a string, mapping or list.
    An image parameter is declared as image_id; the router delivers it to execute
    as image_base64. call_tool reaches the base tools and the other generated
    tools through the router.
    """

    client = None

    def __init__(self, *, tool_name, tool_description, input_schema):
        """Record the model-facing name, description and argument schema."""
        super().__init__(tool_name=tool_name, tool_description=tool_description, tool_version="1.0",
                         input_types=input_schema.get("properties", {}), output_type="str | dict | list",
                         input_schema=input_schema)

    def call_tool(self, name, **args):
        """Run a router tool with task arguments and return its result.

        Raises:
            ToolCallError: The tool returned a failure envelope.
            RuntimeError: The tool was loaded without a router client.
        """
        if self.client is None:
            raise RuntimeError(f"{self.tool_name} has no router client; load it with load_catalog(..., client=...)")
        payload = self.client.execute(name, args, tool_call_id=f"{self.tool_name}-{uuid.uuid4().hex[:8]}")
        if not payload.get("ok"):
            raise ToolCallError(f"{name}: {(payload.get('error') or {}).get('message', 'call failed')}")
        return payload.get("result")
