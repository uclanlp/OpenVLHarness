"""The agent sees the base tools plus the generated tools retrieved for each query."""
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import requests

from ..agent.agent import MainAgent
from .capability import Capability
from .catalog import DEFAULT_CATALOG, tool_problem
from .errors import CatalogLoadError
from .retrieval import ENCODER, TOP_K, ToolRetriever
from .worker import catalog_snapshot

logger = logging.getLogger(__name__)


def router_catalog(client_config):
    """Return the generated tools a ToolSmith router serves and the digest of their snapshot.

    The router answers GET /toolsmith/catalog with the metadata its worker validated. Any
    other router, one without a toolsmith section or from before this route, answers 404:
    then None, and the caller reads the catalog itself.

    Returns:
        (tools, catalog_sha256), tools mapping names to objects with tool_name,
        tool_description and input_schema; or None.

    Raises:
        CatalogLoadError: The router cannot be reached, or what it serves is no valid catalog.
    """
    base = client_config["base_url"].rstrip("/")  # as ToolClient joins it
    try:
        response = requests.get(f"{base}/toolsmith/catalog", timeout=min(client_config.get("timeout_seconds", 30), 60))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        served = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise CatalogLoadError(f"cannot read the ToolSmith catalog the tool router at {base} serves "
                               f"({type(exc).__name__}); start the router with this configuration first") from exc
    digest = served.get("catalog_sha256") if isinstance(served, dict) else None
    listed = served.get("tools") if isinstance(served, dict) else None
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or not isinstance(listed, list):
        raise CatalogLoadError(f"the tool router at {base} serves no valid ToolSmith catalog: expected "
                               "catalog_sha256 and a list of tools")
    tools = {}
    for tool in listed:
        # The agent offers what it validates itself, as when it loads the catalog: the router may be older.
        problem = tool_problem(tool) if isinstance(tool, dict) else "a tool must be a mapping"
        name = tool.get("tool_name") if isinstance(tool, dict) else None
        if problem or name in tools:
            raise CatalogLoadError(f"the tool router at {base} serves a ToolSmith tool the agent cannot offer "
                                   f"({name!r}): {problem or 'its name is listed twice'}")
        tools[name] = SimpleNamespace(tool_name=name, tool_description=tool["tool_description"],
                                      input_schema=tool["input_schema"])
    return tools, digest


class ToolSmithAgent(MainAgent):
    """MainAgent whose tool list adds the top-K generated tools retrieved per query."""

    def __init__(self, *args, toolsmith, client_config, **kwargs):
        """Configure the base agent, then read the catalog's tools and build their retrieval index.

        The tools come from the router the agent calls (GET /toolsmith/catalog), so an
        evaluator needs neither the catalog's files nor a confined worker of its own. Only
        a router that serves no such list (404) makes the agent read the catalog in a
        local worker, as the router does. catalog_sha256 is the digest of the snapshot the
        tools came from, which run identities record.

        Args:
            *args: MainAgent positional arguments; capabilities are the base tools.
            toolsmith: Validated toolsmith configuration section.
            client_config: ToolClient connection settings shared with the base tools.
            **kwargs: MainAgent keyword arguments.

        Raises:
            ToolSmithStartError: The catalog cannot be read (CatalogLoadError: from the
                router, from the files, or a generated tool shadows a base tool), the local
                worker cannot start, or the encoder cannot load.
        """
        super().__init__(*args, **kwargs)
        served = router_catalog(client_config)
        if served is None:
            logger.warning("The tool router at %s serves no ToolSmith catalog list; reading %s in a local worker",
                           client_config["base_url"], Path(toolsmith.get("catalog") or DEFAULT_CATALOG))
            # Read in a worker like the router's, so catalog code does not run in the evaluator's own process.
            served = catalog_snapshot(toolsmith.get("catalog"), settings=toolsmith.get("worker"))
        tools, self.catalog_sha256 = served
        shadowed = sorted(set(tools) & set(self.capabilities))
        if shadowed:
            catalog = Path(toolsmith.get("catalog") or DEFAULT_CATALOG)
            raise CatalogLoadError(f"the ToolSmith catalog {catalog} has tools that shadow base tools: "
                                   f"{', '.join(shadowed)}; rename them")
        self.generated = {name: Capability(tool, client_config=client_config) for name, tool in tools.items()}
        self.retriever = ToolRetriever({name: tool.tool_description for name, tool in tools.items()},
                                       encoder=toolsmith.get("encoder", ENCODER), top_k=toolsmith.get("top_k", TOP_K))

    def tools_for(self, question):
        """Return the base tools plus the generated tools retrieved for the question text."""
        query = "\n".join(item["value"] for item in question if item["type"] == "text")
        selected = self.retriever.retrieve(query)
        return {**self.capabilities, **{name: self.generated[name] for name in selected}}
