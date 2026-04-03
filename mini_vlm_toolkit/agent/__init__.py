from .base import BaseAgent, BaseAgentWithTools, BaseAgentWithToolsV2, BaseAgentWithToolsV3
from .base_with_code import BaseAgentWithToolsAndCode
from .rag import BaseAgentWithRAG
from .config import load_agent_config

try:
    from .dvd import DVDAgent
except Exception:  # pragma: no cover - optional heavy deps may be missing.
    DVDAgent = None

AVAILABLE_AGENTS = {
    'BaseAgent': BaseAgent,
    'BaseAgentWithTools': BaseAgentWithTools,
    'BaseAgentWithToolsAndCode': BaseAgentWithToolsAndCode,
    'BaseAgentWithToolsV2': BaseAgentWithToolsV2,
    'BaseAgentWithToolsV3': BaseAgentWithToolsV3,
    'BaseAgentWithRAG': BaseAgentWithRAG,
}

if DVDAgent is not None:
    AVAILABLE_AGENTS['DVDAgent'] = DVDAgent

def get_agent_class(agent_name: str):
    """
    Get agent class by name from available agent classes.
    
    Args:
        agent_name: Name of the agent class (e.g., 'BaseAgent')
        module_dict: Dictionary of available classes (typically from globals())
    
    Returns:
        The agent class
    
    Raises:
        ValueError: If agent class not found
    """
    if agent_name not in AVAILABLE_AGENTS:
        available = [name for name in AVAILABLE_AGENTS.keys()]
        raise ValueError(
            f"Agent '{agent_name}' not found. Available agents: {available}"
        )

    return AVAILABLE_AGENTS[agent_name]


def build_agent(config_path: str) -> BaseAgent:
    """
    Build an agent from a YAML config file.
    
    The config file should have the following structure:
    ```yaml
    agent_name: BaseAgent
    agent_kwargs:
      base: "http://localhost:8000"
      model: "gpt-4"
      temperature: 0.7
      max_tokens: 2048
      # ... any other kwargs
    ```
    
    The function automatically:
    - Loads the YAML config
    - Finds the agent class by name
    - Filters kwargs to match the agent's constructor signature
    - Instantiates and returns the agent
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        An instance of the specified agent class
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If agent class not found or config is invalid
    """
    # Load config
    config = load_agent_config(config_path)
    
    agent_name = config.get('agent_name', 'BaseAgent')
    agent_kwargs = config.get('agent_kwargs', {})
    
    # Get agent class from current module
    agent_class = get_agent_class(agent_name)
    
    print(f"Building agent '{agent_name}' with kwargs: {agent_kwargs}")
    # Instantiate and return
    return agent_class(**agent_kwargs)
