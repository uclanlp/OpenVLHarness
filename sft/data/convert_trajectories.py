#!/usr/bin/env python3
"""Convert mini-vlm-toolkit full_trajectories/ folders to LlamaFactory SFT format.

Reads a ``full_trajectories/`` directory of per-task ``{task_id}.json`` files
and converts them to LlamaFactory-compatible ShareGPT format for supervised
fine-tuning.

Each input file is expected to have:
  - ``prediction`` and ``ground_truth`` (top-level)
  - ``messages``: a list of OpenAI-style messages where user content is a
    list of parts, each either ``{"type": "text", "text": ...}`` or
    ``{"type": "image_url", "image_url": {"url": ...}}``. Image urls are
    relative paths into a sibling ``images/`` folder (produced by
    ``run_benchmark.py --save-images files``); inline ``data:image/...`` URIs
    are also tolerated but skipped (no on-disk path to reference).

Usage:
    python -m sft.data.convert_trajectories \
        --input outputs/.../full_trajectories \
        --output sft/data/my_dataset.json \
        [--filter-success]                # only keep correct samples
        [--success-criteria exact_match]  # exact_match | llm_judge | mean_iou_point_5
        [--skip-env-updates]              # strip "Environment Update" messages
        [--skip-rag-instructions]         # strip "[RAG Step ...]" instruction messages
        [--strip-experience-block]        # remove <learned_principles> block from system prompt
                                          # (default: True; use --no-strip-experience-block to keep it)
        [--max-samples 100]               # limit number of samples

The --strip-experience-block flag is for trajectories produced by the
experience learning pipeline (run_experience_learning.py). When the agent
runs with an experience_file, the loaded principles get appended to the
system prompt and persisted in every saved trajectory. To avoid leaking
those principles into SFT training data (which would create a train/inference
distribution mismatch), this flag strips the appended block before
conversion, leaving only the original base system prompt.

Output: JSON file in LlamaFactory ShareGPT format. Each user message that
contains images is prefixed with one ``<image>`` placeholder per image, and
the resolved on-disk paths are appended to a per-sample ``images`` list in
the same order — this is the alignment LlamaFactory's data loader expects.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    """Remove the appended <learned_principles> block from a system prompt."""
    if not text:
        return text
    return _EXPERIENCE_BLOCK_RE.sub("", text)


def load_full_trajectories(folder: str) -> List[Dict[str, Any]]:
    """Load per-task JSON files from a full_trajectories/ folder.

    Sorted by task_id (numeric when possible) for deterministic output.
    Skips the sibling ``images/`` subdirectory.
    """
    folder_path = Path(folder)
    files = [p for p in folder_path.iterdir() if p.is_file() and p.suffix == ".json"]

    def _sort_key(p: Path):
        stem = p.stem
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    files.sort(key=_sort_key)
    records = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


# ---------------------------------------------------------------------------
# Success criteria
# ---------------------------------------------------------------------------


def _success_exact_match(prediction: Any, ground_truth: Any) -> bool:
    if prediction is None or ground_truth is None:
        return False
    return str(prediction).strip() == str(ground_truth).strip()


def _success_llm_judge(prediction: Any, ground_truth: Any) -> bool:
    raise NotImplementedError("llm_judge success criterion not implemented")


def _success_mean_iou_point_5(prediction: Any, ground_truth: Any) -> bool:
    raise NotImplementedError("mean_iou_point_5 success criterion not implemented")


_SUCCESS_FNS = {
    "exact_match": _success_exact_match,
    "llm_judge": _success_llm_judge,
    "mean_iou_point_5": _success_mean_iou_point_5,
}


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _is_env_update(msg: Dict[str, Any]) -> bool:
    content = msg.get("content", "")
    if isinstance(content, str):
        return "Environment Update" in content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and "Environment Update" in str(part.get("text", "")):
                return True
    return False


def _is_rag_instruction(msg: Dict[str, Any]) -> bool:
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
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _resolve_image_url(url: str, input_dir: str) -> Optional[str]:
    """Resolve a relative image_url path against ``input_dir``.

    Returns the resolved file path (absolute when ``input_dir`` is absolute)
    if the file exists, else None. Inline ``data:`` URIs and unresolvable
    paths are skipped.
    """
    if not isinstance(url, str) or not url or url.startswith("data:"):
        return None
    candidate = url if os.path.isabs(url) else os.path.normpath(os.path.join(input_dir, url))
    return candidate if os.path.isfile(candidate) else None


def _extract_user_text_and_images(
    content: Any, input_dir: str
) -> Tuple[str, List[str]]:
    """Return (text, image_paths) for a user message's content list.

    Walks the parts in order: text parts are joined with newlines, image_url
    parts are resolved against ``input_dir``. The returned image list is in
    encounter order — same order they should be referenced in the prefixed
    ``<image>`` tokens.
    """
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    text_parts: List[str] = []
    image_paths: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "text":
            text_parts.append(item.get("text", ""))
        elif t == "image_url":
            url = (item.get("image_url") or {}).get("url", "")
            resolved = _resolve_image_url(url, input_dir)
            if resolved is not None:
                image_paths.append(resolved)
    return "\n".join(text_parts), image_paths


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


# ---------------------------------------------------------------------------
# Record conversion
# ---------------------------------------------------------------------------


def convert_record(
    record: Dict[str, Any],
    input_dir: str,
    skip_env_updates: bool = True,
    skip_rag_instructions: bool = False,
    strip_experience_block: bool = True,
) -> Optional[Dict[str, Any]]:
    """Convert one full_trajectories record to LlamaFactory ShareGPT format.

    Inserts ``<image>`` placeholders inline (one per image, prepended to the
    user message that originally contained that image) and accumulates the
    resolved file paths into the ShareGPT ``images`` list in the same order.
    Returns None if the record should be skipped.
    """
    messages = record.get("messages", [])
    if not messages:
        return None

    conversations: List[Dict[str, str]] = []
    images: List[str] = []

    for msg in messages:
        role = msg.get("role", "")

        if skip_env_updates and role == "user" and _is_env_update(msg):
            continue
        if skip_rag_instructions and role == "user" and _is_rag_instruction(msg):
            continue

        if role == "system":
            sharegpt_role = "system"
            value = _extract_text_content(msg.get("content", ""))
            if strip_experience_block:
                value = _strip_experience_block(value)
        elif role == "user":
            sharegpt_role = "human"
            text, msg_images = _extract_user_text_and_images(
                msg.get("content", ""), input_dir
            )
            if msg_images:
                value = ("<image>\n" * len(msg_images)) + text
                images.extend(msg_images)
            else:
                value = text
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

    # Defensive: <image> token count must match images list length, otherwise
    # LlamaFactory will reject or mis-align the sample. Only human turns
    # carry placeholders, so other roles are excluded to avoid false
    # positives from incidental "<image>" substrings in tool output.
    token_count = sum(
        c["value"].count("<image>") for c in conversations if c["from"] == "human"
    )
    assert token_count == len(images), (
        f"<image> token / images-list mismatch: tokens={token_count} "
        f"images={len(images)} (task_id={record.get('task_id')})"
    )

    result: Dict[str, Any] = {"conversations": conversations}
    if images:
        result["images"] = images
    return result


def convert_trajectories(
    input_path: str,
    output_path: str,
    filter_success: bool = False,
    success_criteria: str = "exact_match",
    skip_env_updates: bool = True,
    skip_rag_instructions: bool = False,
    strip_experience_block: bool = True,
    max_samples: Optional[int] = None,
) -> int:
    """Convert a full_trajectories/ folder to LlamaFactory SFT format.

    Returns the number of converted samples.
    """
    if not os.path.isdir(input_path):
        raise NotADirectoryError(
            f"--input must be a full_trajectories/ folder, got: {input_path}"
        )

    if filter_success and success_criteria not in _SUCCESS_FNS:
        raise ValueError(
            f"Unknown success_criteria {success_criteria!r}; "
            f"expected one of {sorted(_SUCCESS_FNS)}"
        )
    success_fn = _SUCCESS_FNS[success_criteria] if filter_success else None

    records = load_full_trajectories(input_path)
    converted = []

    for record in records:
        if filter_success and not success_fn(
            record.get("prediction"), record.get("ground_truth")
        ):
            continue

        sample = convert_record(
            record,
            input_dir=input_path,
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
        description="Convert mini-vlm-toolkit full_trajectories/ to LlamaFactory SFT format."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a full_trajectories/ folder containing per-task JSON files",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the LlamaFactory ShareGPT JSON file",
    )
    parser.add_argument(
        "--filter-success", action="store_true",
        help="Only keep samples whose prediction matches ground_truth under --success-criteria.",
    )
    parser.add_argument(
        "--success-criteria",
        choices=sorted(_SUCCESS_FNS.keys()),
        default="exact_match",
        help="Comparator used by --filter-success (default: exact_match). "
             "'llm_judge' and 'mean_iou_point_5' are not implemented yet.",
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
        success_criteria=args.success_criteria,
        skip_env_updates=args.skip_env_updates,
        skip_rag_instructions=args.skip_rag_instructions,
        strip_experience_block=args.strip_experience_block,
        max_samples=args.max_samples,
    )
    print(f"Converted {count} trajectories → {args.output}")


if __name__ == "__main__":
    main()
