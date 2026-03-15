"""
Reflector module: analyzes agent trajectories to extract reusable
tool-usage principles from playground (Q&A) data.

Three generation strategies:
  - batch:              compress all trajectories → 1 reflector call → principles
  - instance:           per-instance reflection, save lesson files (no summary)
  - instance_summarize: per-instance reflection + summarize into principles
"""

import copy
import json
import os
import random
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import yaml

from ..smp import load as smp_load
from ..utils.api_utils import chat_completion_request
from .constant import *


# ====================================================================
# 1. Trajectory loading
# ====================================================================

def load_full_trajectories(trajectory_dir: str) -> List[Dict[str, Any]]:
    """Load per-instance JSONs from a full_trajectories/ directory.

    Each file is ``{task_id, dataset, model_name, agent_class, prediction, messages}``.
    """
    records = []
    if not os.path.isdir(trajectory_dir):
        raise FileNotFoundError(f"Trajectory directory not found: {trajectory_dir}")
    for fname in sorted(os.listdir(trajectory_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(trajectory_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            record = json.load(f)
        records.append(record)
    return records


def load_results_with_answers(result_file: str) -> Dict[Any, Dict[str, str]]:
    """Load a result file and return ``{index: {prediction, answer, question}}``."""
    data = smp_load(result_file)
    if hasattr(data, "iterrows"):
        mapping = {}
        for _, row in data.iterrows():
            entry: Dict[str, str] = {}
            if "prediction" in row:
                entry["prediction"] = str(row["prediction"])
            if "answer" in row:
                entry["answer"] = str(row["answer"])
                # map option to entry as well
                if row["answer"] in row:
                    entry[str(row["answer"])] = str(row[str(row["answer"])])
                    
            if "question" in row:
                entry["question"] = str(row["question"])
            mapping[row["index"]] = entry
        return mapping
    if isinstance(data, dict):
        return data
    return {}


# ====================================================================
# 2. Labeling & filtering
# ====================================================================

def _default_judge(prediction: str, answer: str) -> bool:
    pred = prediction.strip().lower()
    ans = answer.strip().lower()
    if not ans:
        return False
    return ans in pred or pred == ans


def label_trajectories(
    trajectories: List[Dict[str, Any]],
    results: Dict[Any, Dict[str, str]],
    judge_fn: Optional[Callable[[str, str], bool]] = None,
) -> List[Dict[str, Any]]:
    """Label each trajectory with ``is_correct`` and ``ground_truth``.

    Returns a single list of all labeled records.
    """
    judge = judge_fn or _default_judge
    labeled: List[Dict[str, Any]] = []
    for record in trajectories:
        task_id = record.get("task_id")
        result_entry = results.get(task_id, {})
        answer = result_entry.get("answer", "")
        prediction = record.get("prediction") or result_entry.get("prediction", "")
        question = result_entry.get("question", "")

        rec = copy.deepcopy(record)
        rec["ground_truth"] = answer
        rec["question"] = question or rec.get("question", "")
        rec["is_correct"] = judge(str(prediction), str(answer))
        labeled.append(rec)
    return labeled


def filter_for_reflection(
    labeled: List[Dict[str, Any]],
    reflect_on: str = "all",
) -> List[Dict[str, Any]]:
    """Filter labeled trajectories for reflection.

    Args:
        reflect_on: ``"all"`` or ``"failures_only"``
    """
    if reflect_on == "failures_only":
        return [r for r in labeled if not r.get("is_correct", True)]
    return labeled


# ====================================================================
# 2b. Trajectory pruning
# ====================================================================

def prune_trajectory_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prune raw trajectory messages and return multimodal content blocks.

    Removes:
    - System prompt messages (role=system)
    - User messages containing "Environment Update"
    - ``tool_call_id`` from tool role messages
    - ``id`` from tool_calls in assistant messages

    Returns:
        List of OpenAI content blocks (text and image_url dicts) representing
        the pruned trajectory with actual images preserved.
    """
    pruned: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")

        # Skip system messages
        if role == "system":
            continue

        # Skip user messages containing "Environment Update"
        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and "Environment Update" in content:
                continue
            # content can be a list of dicts (multimodal)
            if isinstance(content, list):
                text_parts = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
                if "Environment Update" in text_parts:
                    continue

        cleaned = copy.deepcopy(msg)

        # Remove tool_call_id from tool messages
        if role == "tool":
            cleaned.pop("tool_call_id", None)

        # Remove id from tool_calls in assistant messages
        if role == "assistant" and "tool_calls" in cleaned:
            for tc in cleaned["tool_calls"]:
                tc.pop("id", None)

        pruned.append(cleaned)

    # prune the last response since it's already in {prediction}
    if pruned and pruned[-1].get("role") == "assistant":
        last_msg = pruned.pop()

    # Build a list of content blocks (text + image_url)
    blocks: List[Dict[str, Any]] = []
    for msg in pruned:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "assistant" and "tool_calls" in msg:
            tool_calls = msg["tool_calls"]
            tc_strs = []
            for tc in tool_calls:
                fn = tc.get("function", tc)
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                # Scrub image data from tool call arguments
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        parsed = None
                    if isinstance(parsed, dict):
                        parsed = {k: _scrub_image_arg(k, v) for k, v in parsed.items()}
                        args = json.dumps(parsed, ensure_ascii=False)
                tc_strs.append(f"{name}({args})")
            blocks.append({"type": "text", "text": f"[assistant] Tool calls: {'; '.join(tc_strs)}"})
            if content:
                blocks.append({"type": "text", "text": f"[assistant] {content}"})
        elif role == "tool":
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)
            blocks.append({"type": "text", "text": f"[tool] {content}"})
        else:
            if isinstance(content, list):
                # Multimodal content: pass through text and image_url blocks
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        blocks.append({"type": "text", "text": f"[{role}] {part.get('text', '')}"})
                    elif part.get("type") == "image_url":
                        blocks.append(part)
            elif isinstance(content, str):
                blocks.append({"type": "text", "text": f"[{role}] {content}"})
            else:
                blocks.append({"type": "text", "text": f"[{role}] {json.dumps(content, ensure_ascii=False)}"})

    return blocks


# ====================================================================
# 3. Trajectory formatting
# ====================================================================

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _scrub_image_arg(key: str, value: Any) -> Any:
    """Replace long image data in tool args with a placeholder."""
    if isinstance(value, str) and len(value) > 200:
        if "base64" in key.lower() or "image" in key.lower():
            return "<image_data>"
    return value


def _extract_tool_calls_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract tool call + tool response pairs from the OpenAI messages list."""
    turns = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            turn = {"tool_calls": msg["tool_calls"], "tool_responses": []}
            # Collect subsequent tool responses
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                turn["tool_responses"].append(messages[j])
                j += 1
            turns.append(turn)
            i = j
        else:
            i += 1
    return turns


def compress_trajectory(record: Dict[str, Any], max_result_chars: int = 80) -> str:
    """Compress a labeled trajectory into a ~200-token one-line digest.

    Format: ``[CORRECT/INCORRECT] Q: ... | Tools: name(args)→result, ... | Ans: ... | GT: ...``
    """
    label = "CORRECT" if record.get("is_correct") else "INCORRECT"
    question = record.get("question", "")
    prediction = record.get("prediction", "")
    ground_truth = record.get("ground_truth", "")

    # Extract tool calls from messages
    messages = record.get("messages", [])
    tool_parts = []
    turns = _extract_tool_calls_from_messages(messages)
    for turn in turns:
        for tc in turn.get("tool_calls", []):
            fn = tc.get("function", tc)
            name = fn.get("name", "?")
            args = fn.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(args, dict):
                args = {k: _scrub_image_arg(k, v) for k, v in args.items()}
                # Keep only non-image args for brevity
                args = {k: v for k, v in args.items() if v != "<image_data>"}
            args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            args_str = _truncate(args_str, 100)

            # Find matching response
            tc_id = tc.get("id", "")
            result_text = ""
            for tr in turn.get("tool_responses", []):
                if tr.get("tool_call_id") == tc_id or not tc_id:
                    content = tr.get("content", "")
                    if isinstance(content, (dict, list)):
                        content = json.dumps(content, ensure_ascii=False)
                    result_text = _truncate(str(content), max_result_chars)
                    break
            tool_parts.append(f"{name}({args_str})→{result_text}")

    tools_str = ", ".join(tool_parts) if tool_parts else "no tools used"
    return f"[{label}] Q: {_truncate(question, 150)} | Tools: {tools_str} | Ans: {_truncate(str(prediction), 100)} | GT: {_truncate(str(ground_truth), 100)}"


def format_trajectory_full(
    record: Dict[str, Any],
    max_tool_response_chars: int = 500,
) -> str:
    """Render a full trajectory for per-instance reflection (map_reduce)."""
    parts: List[str] = []
    label = "CORRECT" if record.get("is_correct") else "INCORRECT"
    parts.append(f"[{label}]")

    question = record.get("question", "")
    if question:
        parts.append(f"Question: {question}")

    messages = record.get("messages", [])
    turns = _extract_tool_calls_from_messages(messages)
    for turn_num, turn in enumerate(turns, 1):
        for tc in turn.get("tool_calls", []):
            fn = tc.get("function", tc)
            name = fn.get("name", "?")
            args = fn.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(args, dict):
                args = {k: _scrub_image_arg(k, v) for k, v in args.items()}
            parts.append(f"  Turn {turn_num} → {name}({json.dumps(args, ensure_ascii=False)})")

        for tr in turn.get("tool_responses", []):
            content = tr.get("content", "")
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)
            content = _truncate(str(content), max_tool_response_chars)
            parts.append(f"    Result: {content}")

    # Assistant reasoning (from messages with role=assistant, no tool_calls)
    for msg in messages:
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            content = msg.get("content", "")
            if content:
                parts.append(f"  Reasoning: {_truncate(str(content), 500)}")

    parts.append(f"Agent Answer: {record.get('prediction', '')}")
    parts.append(f"Ground Truth: {record.get('ground_truth', '')}")
    return "\n".join(parts)

# ====================================================================
# 5. Strategy A: batch
# ====================================================================

def run_reflector_batch(
    labeled: List[Dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    previous_experience: Optional[str] = None,
    max_examples: int = 50,
    max_result_chars: int = 80,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Compress all trajectories into digests, then make 1 reflector call."""
    sampled = (
        random.sample(labeled, max_examples)
        if len(labeled) > max_examples
        else labeled
    )
    digests = [compress_trajectory(r, max_result_chars) for r in sampled]

    n_correct = sum(1 for r in labeled if r.get("is_correct"))
    n_incorrect = len(labeled) - n_correct

    user_parts = [
        f"I have {len(labeled)} trajectories ({n_correct} correct, {n_incorrect} incorrect) "
        f"from an AI agent using tools to answer visual questions.",
        f"\nShowing {len(digests)} representative examples:\n",
    ]
    for i, d in enumerate(digests, 1):
        user_parts.append(f"{i}. {d}")

    user_parts.append("\nPlease analyze these trajectories and extract reusable tool-usage principles.")

    if previous_experience:
        system_prompt = REFLECTOR_REFINE_SYSTEM_PROMPT.format(
            previous_experience=previous_experience
        )
    else:
        system_prompt = REFLECTOR_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    response = chat_completion_request(
        model=model,
        messages=messages,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return response["choices"][0]["message"]["content"]


# ====================================================================
# 6. Strategy B: map_reduce
# ====================================================================
def reflect_on_instance(
    record: Dict[str, Any],
    *,
    model: str,
    base_url: str,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    max_tool_response_chars: int = 500,
    instance_reflection_prompt: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Reflect on a single trajectory and return a short per-instance lesson.

    Uses ``prune_trajectory_messages`` to clean up the raw messages and
    fills the configurable prompt template with ``{pruned_trajectory}``,
    ``{last_response}``, and ``{ground_truth}`` placeholders.
    """
    raw_messages = record.get("messages", [])
    prediction = record.get("prediction", "")
    content_blocks = prune_trajectory_messages(raw_messages)
    ground_truth = record.get("ground_truth", "")

    prompt_template = resolve_instance_reflection_prompt(instance_reflection_prompt)

    # Build multimodal content: [before_text] + [trajectory blocks] + [after_text]
    user_content: List[Dict[str, Any]] = []
    user_content.append({"type": "text", "text": f"- Failed trajectory:"})
    user_content.extend(content_blocks)
    user_content.append({"type": "text", "text": f"- Agent's final answer: {prediction}"})
    user_content.append({"type": "text", "text": f"- Ground truth answer: {ground_truth}"})

    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": user_content},
    ]
    # create messages snapshot for debugging, but skip image data for brevity
    debug_messages = copy.deepcopy(messages)
    for msg in debug_messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    part["image_url"] = "<image_url>"
    # print(f"Reflection input messages for task_id={record.get('task_id', '')}:\n{json.dumps(debug_messages, ensure_ascii=False, indent=2)}\n{'='*50}\n")
    response = chat_completion_request(
        model=model,
        messages=messages,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    # print(f"Reflection response for task_id={record.get('task_id', '')}:\n{response}\n{'='*50}\n")
    return response["choices"][0]["message"]["content"]


def save_instance_lessons(
    lessons: Dict[str, str],
    output_dir: str,
) -> None:
    """Save per-instance lessons to a directory. Replaces existing for same task_id."""
    os.makedirs(output_dir, exist_ok=True)
    for task_id, lesson in lessons.items():
        fpath = os.path.join(output_dir, f"{task_id}.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(lesson)


def load_instance_lessons(output_dir: str) -> Dict[str, str]:
    """Load accumulated per-instance lessons from a directory."""
    lessons = {}
    if not os.path.isdir(output_dir):
        return lessons
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".txt"):
            continue
        task_id = fname[:-4]  # strip .txt
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            lessons[task_id] = f.read().strip()
    return lessons


def summarize_lessons(
    lessons: List[str],
    *,
    model: str,
    base_url: str,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    chunk_size: int = 0,
    previous_experience: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Final reduce: aggregate per-instance lessons into general principles.

    When *previous_experience* is provided, uses the refine prompt to update
    existing principles with only the new lessons, rather than re-summarizing
    everything from scratch.

    When *chunk_size* > 0 and ``len(lessons) > chunk_size``, delegates to
    :func:`summarize_lessons_hierarchical` for chunked summarization.
    Otherwise uses the original single-call approach.
    """
    if chunk_size > 0 and len(lessons) > chunk_size:
        return summarize_lessons_hierarchical(
            lessons,
            model=model,
            base_url=base_url,
            chunk_size=chunk_size,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            previous_experience=previous_experience,
            extra_body=extra_body,
        )

    user_parts = [
        f"I have {len(lessons)} per-instance lessons from the latest trial. "
        f"Please {'refine the principles based on' if previous_experience else 'synthesize them into principles from'} these lessons.\n",
    ]
    for i, lesson in enumerate(lessons, 1):
        user_parts.append(f"### Instance {i}\n{lesson}\n")

    if previous_experience:
        system_prompt = SUMMARIZER_REFINE_PROMPT.format(
            previous_experience=previous_experience
        )
    else:
        system_prompt = SUMMARIZER_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    response = chat_completion_request(
        model=model,
        messages=messages,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return response["choices"][0]["message"]["content"]


def summarize_lessons_hierarchical(
    lessons: List[str],
    *,
    model: str,
    base_url: str,
    chunk_size: int = 20,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    previous_experience: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Hierarchical summarization: chunk lessons, summarize each, then merge.

    1. Split *lessons* into chunks of *chunk_size*.
    2. Summarize each chunk with ``SUMMARIZE_CHUNK_PROMPT``.
    3. Merge all chunk summaries (with previous principles if available)
       into final structured principles.
    """
    import logging
    logger = logging.getLogger("ExperienceLearning")

    # Split into chunks
    chunks = [
        lessons[i : i + chunk_size]
        for i in range(0, len(lessons), chunk_size)
    ]
    num_chunks = len(chunks)
    logger.info(
        f"Hierarchical summarization: {len(lessons)} lessons -> "
        f"{num_chunks} chunks of ~{chunk_size}"
    )

    # Phase 1: summarize each chunk
    chunk_summaries = []
    for idx, chunk in enumerate(chunks, 1):
        logger.info(f"Summarizing chunk {idx}/{num_chunks}...")
        user_parts = [
            f"Here are {len(chunk)} per-instance lessons (chunk {idx}/{num_chunks}):\n",
        ]
        for i, lesson in enumerate(chunk, 1):
            user_parts.append(f"### Instance {i}\n{lesson}\n")

        messages = [
            {"role": "system", "content": SUMMARIZE_CHUNK_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ]
        response = chat_completion_request(
            model=model,
            messages=messages,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        chunk_summaries.append(response["choices"][0]["message"]["content"])

    # Phase 2: merge chunk summaries (with previous principles if refining)
    logger.info(f"Merging {num_chunks} chunk summaries into final principles...")
    merge_parts = [
        f"Below are {num_chunks} intermediate summaries from chunked lesson analysis.\n"
        f"Please merge them into a final set of principles.\n",
    ]
    for idx, summary in enumerate(chunk_summaries, 1):
        merge_parts.append(f"### Chunk {idx}/{num_chunks}\n{summary}\n")

    if previous_experience:
        merge_system_prompt = SUMMARIZER_REFINE_PROMPT.format(
            previous_experience=previous_experience
        )
    else:
        merge_system_prompt = SUMMARIZE_MERGE_PROMPT

    messages = [
        {"role": "system", "content": merge_system_prompt},
        {"role": "user", "content": "\n".join(merge_parts)},
    ]
    response = chat_completion_request(
        model=model,
        messages=messages,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return response["choices"][0]["message"]["content"]


# ====================================================================
# 7. Dispatcher
# ====================================================================

def run_reflector(
    labeled: List[Dict[str, Any]],
    *,
    generation_strategy: str,
    model: str,
    base_url: str,
    api_key: Optional[str] = None,
    previous_experience: Optional[str] = None,
    instance_lessons_dir: Optional[str] = None,
    instance_reflection_prompt: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """Dispatch to batch, instance, or instance_summarize.

    For ``batch``: returns principles string (called per trial).
    For ``instance``: updates instance_lessons_dir, returns None.
    For ``instance_summarize``: per-instance reflection + summarize, returns principles string.
    """
    if generation_strategy == "batch":
        return run_reflector_batch(
            labeled,
            model=model,
            base_url=base_url,
            api_key=api_key,
            previous_experience=previous_experience,
            **kwargs,
        )
    elif generation_strategy in ("instance", "instance_summarize"):
        # Load existing lessons
        existing = {}
        if instance_lessons_dir:
            existing = load_instance_lessons(instance_lessons_dir)

        # Reflect on each instance, collecting new lessons from this trial
        new_lessons: Dict[str, str] = {}
        for record in labeled:
            task_id = str(record.get("task_id", ""))
            if not task_id:
                continue
            lesson = reflect_on_instance(
                record,
                model=model,
                base_url=base_url,
                api_key=api_key,
                instance_reflection_prompt=instance_reflection_prompt,
                **{k: v for k, v in kwargs.items()
                   if k in ("temperature", "max_tokens", "max_tool_response_chars", "extra_body")},
            )
            existing[task_id] = lesson
            new_lessons[task_id] = lesson

        # Save updated lessons
        if instance_lessons_dir:
            save_instance_lessons(existing, instance_lessons_dir)

        if generation_strategy == "instance_summarize":
            # Summarize: if we have previous experience, refine with only
            # the new lessons; otherwise summarize all accumulated lessons.
            if previous_experience and new_lessons:
                lessons_list = list(new_lessons.values())
            else:
                lessons_list = list(existing.values())
            if not lessons_list:
                return ""
            return summarize_lessons(
                lessons_list,
                model=model,
                base_url=base_url,
                api_key=api_key,
                previous_experience=previous_experience,
                **{k: v for k, v in kwargs.items()
                   if k in ("temperature", "max_tokens", "extra_body", "chunk_size")},
            )
        return None  # instance strategy: no summarization
    else:
        raise ValueError(
            f"Unknown generation_strategy: {generation_strategy}. "
            f"Use 'batch', 'instance', or 'instance_summarize'."
        )


# ====================================================================
# 8. Experience file I/O
# ====================================================================

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)


def save_experience(
    principles: str,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save extracted principles to a markdown file with YAML front-matter."""
    meta = metadata or {}
    meta.setdefault("generated_by", "experience_learning")
    meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    front_matter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    content = f"---\n{front_matter}---\n\n{principles}\n"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.abspath(output_path)


def load_experience(experience_path: str) -> str:
    """Load an experience file and return the principles text (no front-matter)."""
    with open(experience_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = _FRONT_MATTER_RE.match(content)
    if match:
        content = content[match.end():]
    return content.strip()
