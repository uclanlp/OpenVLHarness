"""
Agent configuration and builder utilities.
"""

import yaml
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, Type


def load_agent_config(config_path: str) -> Dict[str, Any]:
    """
    Load agent configuration from a YAML file.
    
    Expected YAML structure:
    ```yaml
    agent_name: BaseAgent  # or other agent class name
    agent_config_version: v1_0215
    agent_kwargs:
      base: "http://localhost:8000"
      model: "gpt-4"
      temperature: 0.7
      # ... any other constructor arguments
    ```
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dictionary with 'agent_name' and 'agent_kwargs' keys
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a dictionary, got {type(config)}")
    
    if 'agent_name' not in config:
        raise ValueError("Config must contain 'agent_name' key")
    
    return config
