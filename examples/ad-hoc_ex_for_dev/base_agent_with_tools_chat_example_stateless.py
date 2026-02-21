"""
Stateless integration-style example for BaseAgentWithTools using native chat tool-calling.

Prereqs:
- Tool server running at TOOL_SERVER_URL (default: http://127.0.0.1:8101)
- Qwen3-VL OpenAI-compatible server running at MODEL_BASE_URL (default: http://127.0.0.1:8222)

OpenAI quick start:
- export MODEL_BASE_URL="https://api.openai.com"
- export MODEL_NAME="gpt-5-mini"
- export MODEL_API_KEY="YOUR_OPENAI_API_KEY"
- export TOOL_SERVER_URL="http://127.0.0.1:8101"
- python examples/ad-hoc_ex_for_dev/base_agent_with_tools_chat_example_stateless.py
"""

import os
import json

from mini_vlm_toolkit.agent.base import BaseAgentWithTools


def _preview(value, max_chars=240):
    if value is None:
        return "None"
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    value = value.replace("\n", "\\n")
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}... (truncated)"


def _print_chat_history(chat_history):
    print(f"Turns: {len(chat_history)}")
    for turn in chat_history:
        print(f"\n--- Turn {turn.get('turn')} ---")
        assistant = turn.get("assistant", {})
        print("Assistant perception:", _preview(assistant.get("perception")))
        print("Assistant thinking:", _preview(assistant.get("thinking")))
        print("Assistant content:", _preview(assistant.get("content")))

        tool_calls = assistant.get("tool_calls", [])
        if tool_calls:
            print(f"Tool calls: {len(tool_calls)}")
            for idx, call in enumerate(tool_calls, 1):
                print(
                    f"  [{idx}] id={call.get('id')} name={call.get('name')} "
                    f"args={_preview(call.get('args'))}"
                )

        tool_responses = turn.get("tool_responses", [])
        if tool_responses:
            print(f"Tool responses: {len(tool_responses)}")
            for idx, payload in enumerate(tool_responses, 1):
                print(
                    f"  [{idx}] ok={payload.get('ok')} tool={payload.get('tool_name')} "
                    f"id={payload.get('tool_call_id')}"
                )
                print(f"      result={_preview(payload.get('result'))}")
                print(f"      error={_preview(payload.get('error'))}")
                print(f"      meta={_preview(payload.get('meta'))}")

        if turn.get("final_answer") is not None:
            print("Final answer (turn):", _preview(turn.get("final_answer")))


def _print_message_flow(messages):
    print(f"\nMessage flow ({len(messages)} messages):")
    for idx, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, list):
            image_parts = sum(
                1 for item in content if isinstance(item, dict) and item.get("type") == "image_url"
            )
            text_parts = sum(
                1 for item in content if isinstance(item, dict) and item.get("type") == "text"
            )
            content_desc = f"list(parts={len(content)}, text_parts={text_parts}, image_parts={image_parts})"
        else:
            content_desc = _preview(content)
        has_tool_calls = bool(message.get("tool_calls"))
        print(f"  [{idx + 1}] role={role} tool_calls={has_tool_calls} content={content_desc}")


def main():
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    model_base_url = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8222")
    tool_server_url = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")
    debug_tool_loop = os.getenv("AGENT_DEBUG_TOOL_LOOP", "1") not in {"0", "false", "False"}

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
        "Use Python_Executor_Tool to compute 1717 * 1919. "
        "After using tools if needed, provide the final answer."
    )

    result = agent.generate(
        [{"type": "text", "value": question}],
        debug_tool_loop=debug_tool_loop,
    )
    trajectory = ((result or {}).get("extra_records") or {}).get("trajectory") or {}
    answer = result.get("prediction")
    chat_history = trajectory.get("chat_history") or []
    messages = trajectory.get("messages") or []
    tool_schema_names = [entry.get("function", {}).get("name") for entry in agent.json_schema]
    print("Model:", model_name)
    print("Model base:", model_base_url)
    print("Tool server:", tool_server_url)
    print("Tool schema names:", tool_schema_names)
    print("Question:", question)
    print("Answer:", answer)
    _print_chat_history(chat_history)
    _print_message_flow(messages)


if __name__ == "__main__":
    main()
