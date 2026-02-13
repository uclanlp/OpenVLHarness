"""
Optional integration-style example for image tool responses in BaseAgentWithTools.

Prereqs:
- Tool server at TOOL_SERVER_URL (default: http://127.0.0.1:8101) with SAM3_Segment_Tool enabled.
- Qwen3-VL OpenAI-compatible server at MODEL_BASE_URL (default: http://127.0.0.1:8222).

OpenAI quick start:
- export MODEL_BASE_URL="https://api.openai.com"
- export MODEL_NAME="gpt-5-mini"
- export MODEL_API_KEY="YOUR_OPENAI_API_KEY"
- export TOOL_SERVER_URL="http://127.0.0.1:8101"
- python examples/ad-hoc_ex_for_dev/base_agent_with_tools_image_response_example.py
"""

import json
import os
from pathlib import Path

from mini_vlm_toolkit.agent.base import BaseAgentWithTools


def _preview(value, max_chars=280):
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
                    f"raw_arguments={_preview(call.get('raw_arguments'))}"
                )

        tool_responses = turn.get("tool_responses", [])
        if tool_responses:
            print(f"Tool responses: {len(tool_responses)}")
            for idx, payload in enumerate(tool_responses, 1):
                print(
                    f"  [{idx}] ok={payload.get('ok')} tool={payload.get('tool_name')} "
                    f"id={payload.get('tool_call_id')}"
                )
                print(f"      error={_preview(payload.get('error'))}")
                print(f"      meta={_preview(payload.get('meta'))}")
                result = payload.get("result")
                if isinstance(result, dict):
                    print(f"      result_keys={sorted(result.keys())}")
                    if "all_masks_viz_base64" in result and isinstance(result["all_masks_viz_base64"], str):
                        print(f"      all_masks_viz_base64_len={len(result['all_masks_viz_base64'])}")
                else:
                    print(f"      result={_preview(result)}")

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

    repo_root = Path(__file__).resolve().parents[2]
    default_image = repo_root / "mini_vlm_toolkit/tool_infra/tests/data/sam3_demo.jpg"
    image_path = Path(os.getenv("IMAGE_PATH", str(default_image))).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"IMAGE_PATH does not exist: {image_path}")

    agent = BaseAgentWithTools(
        base=model_base_url,
        key=os.getenv("MODEL_API_KEY", "EMPTY"),
        model=model_name,
        tool_base=tool_server_url,
        tool_list=["SAM3_Segment_Tool"],
        max_iterations=4,
        temperature=0.0,
        max_tokens=1024,
    )

    # HACK (gpt-5-mini): keep this wording strict so the model uses an entity-style
    # SAM3 prompt argument (for example, "children") instead of an instruction.
    # Long-term fix: improve SAM3_Segment_Tool description/schema to enforce this.
    question = (
        "How many children are visible? Reason step by step. "
        "In your first assistant turn, do a step-by-step analysis and then emit exactly one "
        "tool call for SAM3_Segment_Tool with image_path set to the same image and prompt set "
        "EXACTLY to 'children' (single noun only; do not write an instruction sentence). "
        "After receiving the tool response image, continue reasoning step by step based on that "
        "tool image and then give the final numeric answer."
    )

    use_notebook_image_input = os.getenv("USE_NOTEBOOK_IMAGE_INPUT", "1") not in {"0", "false", "False"}
    if use_notebook_image_input:
        try:
            import PIL  # noqa: F401
            inputs = [
                {"type": "image", "value": str(image_path)},
                {"type": "text", "value": question},
            ]
        except Exception as exc:
            print(f"Warning: Pillow unavailable ({exc}); falling back to text-only prompt.")
            question = (
                f"{question}\nUse this exact image_path when calling SAM3_Segment_Tool: {image_path}"
            )
            inputs = [{"type": "text", "value": question}]
    else:
        question = (
            f"{question}\nUse this exact image_path when calling SAM3_Segment_Tool: {image_path}"
        )
        inputs = [{"type": "text", "value": question}]

    answer, chat_history = agent._run(inputs, debug_tool_loop=debug_tool_loop)
    tool_schema_names = [entry.get("function", {}).get("name") for entry in agent.json_schema]
    print("Model:", model_name)
    print("Model base:", model_base_url)
    print("Tool server:", tool_server_url)
    print("Tool schema names:", tool_schema_names)
    print("Image:", image_path)
    print("Question:", question)
    print("Answer:", answer)

    if chat_history and chat_history[0].get("tool_responses"):
        payload = chat_history[0]["tool_responses"][0]
        tool_msg_content = agent._build_tool_message_content(payload)
        print("Tool message content type: text")
        print("Text payload preview:", str(tool_msg_content)[:300])
        tool_image_msgs = [
            m
            for m in (agent.last_run_messages or [])
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(isinstance(part, dict) and part.get("type") == "image_url" for part in m.get("content"))
        ]
        if tool_image_msgs:
            latest_msg = tool_image_msgs[-1]
            image_count = sum(
                1
                for part in latest_msg.get("content", [])
                if isinstance(part, dict) and part.get("type") == "image_url"
            )
            first_text = next(
                (
                    str(part.get("text", ""))
                    for part in latest_msg.get("content", [])
                    if isinstance(part, dict) and part.get("type") == "text"
                ),
                "",
            )
            print("Tool image handoff role: user")
            print("Image parts:", image_count)
            print("User update preview:", first_text[:300])

    _print_chat_history(chat_history)
    _print_message_flow(agent.last_run_messages)


if __name__ == "__main__":
    main()
