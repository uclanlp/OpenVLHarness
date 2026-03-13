import sys
import os

# Ensure the project root is on sys.path so `rl` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from transformers import AutoProcessor
from verl.tools.utils.tool_registry import initialize_tools_from_config

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
tools = initialize_tools_from_config(os.path.join(os.path.dirname(__file__), "../../rl/configs/tool_config.yaml"))
tool_schemas = [t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in tools]

messages = [
    {"role": "system", "content": "You are a visual counting expert..."},
    {"role": "user", "content": "<image>\nHow many people are in this image?"},
]

print(processor.apply_chat_template(messages, tools=tool_schemas, add_generation_prompt=True, tokenize=False))