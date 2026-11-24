"""The agent sees the base tools plus the generated tools retrieved for each query."""
from ..agent.agent import MainAgent
from .capability import Capability
from .retrieval import ENCODER, TOP_K, ToolRetriever
from .worker import catalog_tools


class ToolSmithAgent(MainAgent):
    """MainAgent whose tool list adds the top-K generated tools retrieved per query."""

    def __init__(self, *args, toolsmith, client_config, **kwargs):
        """Configure the base agent, then load the catalog and its retrieval index.

        Args:
            *args: MainAgent positional arguments; capabilities are the base tools.
            toolsmith: Validated toolsmith configuration section.
            client_config: ToolClient connection settings shared with the base tools.
            **kwargs: MainAgent keyword arguments.

        Raises:
            ValueError: The catalog cannot be loaded or a generated tool shadows a base tool.
        """
        super().__init__(*args, **kwargs)
        # Read in a worker like the router's, so catalog code does not run in the evaluator.
        tools = catalog_tools(toolsmith.get("catalog"), settings=toolsmith.get("worker"))
        shadowed = sorted(set(tools) & set(self.capabilities))
        if shadowed:
            raise ValueError(f"Generated tools shadow base tools: {shadowed}")
        self.generated = {name: Capability(tool, client_config=client_config) for name, tool in tools.items()}
        self.retriever = ToolRetriever({name: tool.tool_description for name, tool in tools.items()},
                                       encoder=toolsmith.get("encoder", ENCODER), top_k=toolsmith.get("top_k", TOP_K))

    def tools_for(self, question):
        """Return the base tools plus the generated tools retrieved for the question text."""
        query = "\n".join(item["value"] for item in question if item["type"] == "text")
        selected = self.retriever.retrieve(query)
        return {**self.capabilities, **{name: self.generated[name] for name in selected}}
