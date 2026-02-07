"""
Optional integration-style example for image tool responses in BaseAgentWithTools.

Prereqs:
- Tool server at TOOL_SERVER_URL (default: http://127.0.0.1:8101) with SAM3_Segment_Tool enabled.
- Qwen3-VL OpenAI-compatible server at MODEL_BASE_URL (default: http://127.0.0.1:8222).
"""

import json
import os
from pathlib import Path

from mini_vlm_toolkit.agent.base import BaseAgentWithTools


def main():
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    model_base_url = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8222")
    tool_server_url = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")

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

    question = (
        "Call SAM3_Segment_Tool exactly once with these arguments: "
        f'{{"image_path": "{image_path}", "prompt": "person", '
        '"return_all_masks_viz": true, "return_masks_png_b64": false}}. '
        "Then provide a final answer with the numeric mask count."
    )
    inputs = [{"type": "text", "value": question}]

    answer, chat_history = agent._run(inputs)
    print("Image:", image_path)
    print("Question:", question)
    print("Answer:", answer)

    if chat_history and chat_history[0].get("tool_responses"):
        payload = chat_history[0]["tool_responses"][0]
        tool_msg_content = agent._build_tool_message_content(payload)
        if isinstance(tool_msg_content, list):
            image_count = sum(1 for part in tool_msg_content if part.get("type") == "image_url")
            text_preview = tool_msg_content[0].get("text", "")
            print("Tool message content type: multimodal")
            print("Image parts:", image_count)
            print("Text payload preview:", text_preview[:300])
        else:
            print("Tool message content type: text")
            print("Text payload preview:", str(tool_msg_content)[:300])

    # Dump compact turn info for debugging.
    print("Turns:", len(chat_history))
    for turn in chat_history:
        assistant = turn.get("assistant", {})
        log = {
            "turn": turn.get("turn"),
            "tool_calls": assistant.get("tool_calls", []),
            "final_answer": turn.get("final_answer"),
        }
        print(json.dumps(log, ensure_ascii=False))


if __name__ == "__main__":
    main()
