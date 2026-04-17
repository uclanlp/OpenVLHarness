import yaml
import os

def load_judge_config(config_path: str) -> dict:
    """
    Load judge configuration from a YAML file.
    
    Expected YAML structure:
    ```yaml
    judge_config_version: v1_0215
    model_name: "gpt-4"
    nproc: 4
    retry: 3
    key: "empty"
    api_base: "http://localhost:8000"
    ```
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dictionary with judge configuration parameters
    """
    if config_path is None or not os.path.exists(config_path):
        return {
            'model': 'exact_matching',
            'nproc': 4,
            'retry': 3,
        }
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    judge_kwargs = {
        'model': config.get('model_name'),
        'key': config.get('key', 'empty'),
        'api_base': config.get('api_base'),
        'nproc': config.get('nproc', 4),
        'retry': config.get('retry', 3),
    }
    return judge_kwargs
