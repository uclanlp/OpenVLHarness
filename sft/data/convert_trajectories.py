#!/usr/bin/env python3
"""Convert mini-vlm-toolkit trajectories to LlamaFactory SFT format.

Reads trajectories.jsonl from experiment outputs and converts them to
LlamaFactory-compatible ShareGPT format for supervised fine-tuning.

Usage:
    python -m sft.data.convert_trajectories \
        --input outputs/.../trajectories.jsonl \
        --output sft/data/my_dataset.json \
        [--filter-success]            # only keep trajectories with non-empty final answer
        [--skip-env-updates]          # strip "Environment Update" messages
        [--skip-rag-instructions]     # strip "[RAG Step ...]" instruction messages
        [--strip-experience-block]    # remove <learned_principles> block from system prompt
                                      # (default: True; use --no-strip-experience-block to keep it)
        [--max-samples 100]           # limit number of samples

The --strip-experience-block flag is for trajectories produced by the
experience learning pipeline (run_experience_learning.py). When the agent
runs with an experience_file, the loaded principles get appended to the
system prompt and persisted in every saved trajectory. To avoid leaking
those principles into SFT training data (which would create a train/inference
distribution mismatch), this flag strips the appended block before
conversion, leaving only the original base system prompt.

Output: JSON file in ShareGPT format, compatible with LlamaFactory's data loader.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# Matches the experience block appended by mini_vlm_toolkit/agent/base.py
# (see _construct_system_message). The block starts with the literal preamble
# "Additional learned tool-usage principles from prior runs:" and ends at the
# closing </learned_principles> tag. We strip the leading "\n\n" too so the
# stripped system prompt is byte-identical to a baseline (no-experience) prompt.
_EXPERIENCE_BLOCK_RE = re.compile(
    r"\n\nAdditional learned tool-usage principles from prior runs:.*?</learned_principles>",
    re.DOTALL,
)


def _strip_experience_block(text: str) -> str:
    """Remove the appended <learned_principles> block from a system prompt.

    No-op if the text doesn't contain the block (e.g. baseline trajectories).
    """
    if not text:
        return text
    return _EXPERIENCE_BLOCK_RE.sub("", text)


def load_trajectories(path: str) -> List[Dict[str, Any]]:
    """Load records from trajectories.jsonl."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_env_update(msg: Dict[str, Any]) -> bool:
    """Check if a message is an Environment Update."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return "Environment Update" in content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and "Environment Update" in str(part.get("text", "")):
                return True
    return False


def _is_rag_instruction(msg: Dict[str, Any]) -> bool:
    """Check if a message is a RAG step instruction."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip().startswith("[RAG Step")
    return False


def _extract_text_content(content: Any) -> str:
    """Extract text from message content (handles both str and multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    # Skip inline images in text — they'll be in the images field
                    pass
        return "\n".join(parts)
    return str(content)


def _extract_images(record: Dict[str, Any]) -> List[str]:
    """Extract image paths from the trajectory's input_messages."""
    images = []
    traj = record.get("trajectory", {})
    input_msgs = traj.get("input_messages", [])
    for msg in input_msgs:
        if isinstance(msg, dict) and msg.get("type") == "image":
            path = msg.get("value", "")
            if path and os.path.isfile(path):
                images.append(path)
    return images


def _convert_tool_calls_to_text(msg: Dict[str, Any]) -> str:
    """Convert native tool_calls to text representation for SFT."""
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls", [])
    if not tool_calls:
        return _extract_text_content(content)

    parts = []
    if content:
        parts.append(_extract_text_content(content))
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "unknown")
        args = fn.get("arguments", "{}")
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        parts.append(f"<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>")
    return "\n".join(parts)


def convert_record(
    record: Dict[str, Any],
    skip_env_updates: bool = True,
    skip_rag_instructions: bool = False,
    strip_experience_block: bool = True,
) -> Optional[Dict[str, Any]]:
    """Convert one trajectory record to LlamaFactory ShareGPT format.

    Returns None if the record should be skipped.
    """
    traj = record.get("trajectory", {})
    messages = traj.get("messages", [])
    if not messages:
        return None

    conversations = []
    for msg in messages:
        role = msg.get("role", "")

        # Skip Environment Update messages if requested.
        if skip_env_updates and role == "user" and _is_env_update(msg):
            continue

        # Skip RAG instruction messages if requested.
        if skip_rag_instructions and role == "user" and _is_rag_instruction(msg):
            continue

        # Map roles to ShareGPT format.
        if role == "system":
            sharegpt_role = "system"
            value = _extract_text_content(msg.get("content", ""))
            if strip_experience_block:
                value = _strip_experience_block(value)
        elif role == "user":
            sharegpt_role = "human"
            value = _extract_text_content(msg.get("content", ""))
        elif role == "assistant":
            sharegpt_role = "gpt"
            value = _convert_tool_calls_to_text(msg)
        elif role == "tool":
            sharegpt_role = "tool"
            value = _extract_text_content(msg.get("content", ""))
        else:
            continue

        if not value.strip():
            continue

        conversations.append({"from": sharegpt_role, "value": value})

    if not conversations:
        return None

    result: Dict[str, Any] = {"conversations": conversations}

    # Add images if present.
    images = _extract_images(record)
    if images:
        result["images"] = images

    return result


def convert_trajectories(
    input_path: str,
    output_path: str,
    filter_success: bool = False,
    skip_env_updates: bool = True,
    skip_rag_instructions: bool = False,
    strip_experience_block: bool = True,
    max_samples: Optional[int] = None,
) -> int:
    """Convert trajectories.jsonl to LlamaFactory SFT format.

    Returns the number of converted samples.
    """
    records = load_trajectories(input_path)
    converted = []

    for record in records:
        if filter_success:
            final_answer = record.get("trajectory", {}).get("final_answer", "")
            if not final_answer or final_answer == "Error: Unable to get response from the server.":
                continue

        sample = convert_record(
            record,
            skip_env_updates=skip_env_updates,
            skip_rag_instructions=skip_rag_instructions,
            strip_experience_block=strip_experience_block,
        )
        if sample is not None:
            converted.append(sample)

        if max_samples and len(converted) >= max_samples:
            break

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    return len(converted)


def main():
    parser = argparse.ArgumentParser(
        description="Convert mini-vlm-toolkit trajectories to LlamaFactory SFT format."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to trajectories.jsonl",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for LlamaFactory JSON file",
    )
    parser.add_argument(
        "--filter-success", action="store_true",
        help="Only keep trajectories with non-empty final answer",
    )
    parser.add_argument(
        "--skip-env-updates", action="store_true", default=True,
        help="Strip Environment Update messages (default: True)",
    )
    parser.add_argument(
        "--no-skip-env-updates", action="store_false", dest="skip_env_updates",
    )
    parser.add_argument(
        "--skip-rag-instructions", action="store_true",
        help="Strip RAG step instruction messages",
    )
    parser.add_argument(
        "--strip-experience-block", action="store_true", default=True,
        help="Remove the appended <learned_principles> block from system prompts "
             "(default: True). Use --no-strip-experience-block to keep it.",
    )
    parser.add_argument(
        "--no-strip-experience-block", action="store_false", dest="strip_experience_block",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to convert",
    )
    args = parser.parse_args()

    count = convert_trajectories(
        input_path=args.input,
        output_path=args.output,
        filter_success=args.filter_success,
        skip_env_updates=args.skip_env_updates,
        skip_rag_instructions=args.skip_rag_instructions,
        strip_experience_block=args.strip_experience_block,
        max_samples=args.max_samples,
    )
    print(f"Converted {count} trajectories → {args.output}")


if __name__ == "__main__":
    main()
