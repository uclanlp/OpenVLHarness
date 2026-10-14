import yaml
import os

from openvlharness import bootstrap

def load_judge_config(config_path: str | dict) -> dict:
    """
    Load judge configuration from a YAML file or saved configuration mapping.

    ``${NAME}`` placeholders are resolved from the environment, as in the agent presets.
    
    Expected YAML structure:
    ```yaml
    judge_config_version: v1_0215
    model_name: "gpt-4"
    nproc: 4
    retry: 3
    key: "empty"
    api_base: "http://localhost:8000/v1/chat/completions"
    ```
    
    Args:
        config_path: Path to YAML config file, or a configuration mapping.
    
    Returns:
        Dictionary with judge configuration parameters
    """
    if not isinstance(config_path, dict) and (config_path is None or not os.path.exists(config_path)):
        return {
            'model': 'exact_matching',
            'nproc': 4,
            'retry': 3,
        }
    if isinstance(config_path, dict):
        config = config_path
        source = ""
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        source = f" {config_path}"
    config = bootstrap.expand_environment(config, label="judge configuration", source=source)

    judge_kwargs = {
        'model': config.get('model_name'),
        'key': config.get('key', 'empty'),
        'api_base': config.get('api_base'),
        'nproc': config.get('nproc', 4),
        'retry': config.get('retry', 3),
    }
    # Pass through any additional fields (e.g. enable_llm_judge) so dataset
    # evaluators can read custom flags from the judge config without us
    # having to whitelist each one here.
    _RESERVED = {'judge_config_version', 'model_name', 'key', 'api_base', 'nproc', 'retry'}
    for k, v in config.items():
        if k not in _RESERVED and k not in judge_kwargs:
            judge_kwargs[k] = v
    return judge_kwargs
