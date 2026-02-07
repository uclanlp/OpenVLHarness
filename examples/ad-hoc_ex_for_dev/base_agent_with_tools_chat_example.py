"""
Minimal integration-style example for BaseAgentWithTools using native chat tool-calling.

Prereqs:
- Tool server running at TOOL_SERVER_URL (default: http://127.0.0.1:8101)
- Qwen3-VL OpenAI-compatible server running at MODEL_BASE_URL (default: http://127.0.0.1:8222)
"""

import os

from mini_vlm_toolkit.agent.base import BaseAgentWithTools


def main():
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    model_base_url = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8222")
    tool_server_url = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")

    agent = BaseAgentWithTools(
        base=model_base_url,
        key=os.getenv("MODEL_API_KEY", "EMPTY"),
        model=model_name,
        tool_base=tool_server_url,
        tool_list=["Python_Executor_Tool"],
        max_iterations=4,
        temperature=0.0,
        max_tokens=512,
    )

    question = (
        "Use Python_Executor_Tool to compute 17 * 19. "
        "After using tools if needed, provide the final answer."
    )

    answer = agent.generate(question)
    print("Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
