"""Convert benchmark trajectory records into verl multi-turn SFT parquet."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import pandas as pd

from rl.prompts.system_prompts import build_hermes_system_prompt

_TOOL_CALL_BLOCK = "<tool_call>{payload}</tool_call>"


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_prediction_table(path: str) -> dict[Any, dict[str, Any]]:
    suffix = path.rsplit(".", 1)[-1].lower()
    if suffix == "xlsx":
        df = pd.read_excel(path)
    elif suffix == "csv":
        df = pd.read_csv(path)
    elif suffix == "tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix in {"json", "jsonl"}:
        if suffix == "jsonl":
            df = pd.DataFrame(_read_jsonl(path))
        else:
            raw = json.load(open(path, "r", encoding="utf-8"))
            df = pd.DataFrame(raw if isinstance(raw, list) else raw.get("data", []))
    elif suffix == "pkl":
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported prediction file format: {suffix}")

    mapping = {}
    if "index" not in df.columns:
        return mapping
    for _, row in df.iterrows():
        mapping[row["index"]] = row.to_dict()
    return mapping


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _canonical_letter(value: Any) -> str:
    text = _normalize_text(value).upper()
    match = re.search(r"\b([A-Z])\b", text)
    return match.group(1) if match else ""


def _is_correct(prediction: Any, ground_truth: Any) -> bool:
    pred = _normalize_text(prediction)
    gt = _normalize_text(ground_truth)
    if not pred or not gt:
        return False

    pred_letter = _canonical_letter(pred)
    gt_letter = _canonical_letter(gt)
    if pred_letter and gt_letter:
        return pred_letter == gt_letter
    return pred.lower() == gt.lower()


def _extract_input_user_and_images(trajectory: dict[str, Any]) -> tuple[str, list[Any]]:
    input_messages = trajectory.get("input_messages")
    chunks = []
    images = []

    if isinstance(input_messages, list):
        for item in input_messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            item_type = item.get("type")
            value = item.get("value")
            if role == "system":
                continue
            if item_type == "image":
                chunks.append("<image>")
                if value is not None:
                    images.append(value)
            elif item_type == "text":
                chunks.append(str(value or ""))
            elif isinstance(value, str):
                chunks.append(value)

    user_content = "\n".join(part for part in chunks if part and str(part).strip())
    return user_content, images


def _render_hermes_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    rendered = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        args = call.get("args")
        if args is None:
            raw = call.get("raw_arguments")
            if isinstance(raw, str):
                try:
                    args = json.loads(raw)
                except Exception:
                    args = {}
        if not isinstance(args, dict):
            args = {}
        payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        rendered.append(_TOOL_CALL_BLOCK.format(payload=payload))
    return "\n".join(rendered)


def _trajectory_to_messages(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any], list[dict[str, Any]]]:
    trajectory = record.get("trajectory") if isinstance(record, dict) else None
    trajectory = trajectory if isinstance(trajectory, dict) else {}

    snapshot = trajectory.get("input_snapshot") if isinstance(trajectory.get("input_snapshot"), dict) else {}
    tool_defs = snapshot.get("tool_definitions") if isinstance(snapshot.get("tool_definitions"), list) else []

    user_content, images = _extract_input_user_and_images(trajectory)
    if not user_content:
        user_content = _normalize_text(record.get("prediction")) or "Solve the task."

    system_prompt = snapshot.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        system_prompt = build_hermes_system_prompt(tool_schemas=tool_defs)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    chat_history = trajectory.get("chat_history")
    if isinstance(chat_history, list):
        for turn in chat_history:
            if not isinstance(turn, dict):
                continue
            assistant = turn.get("assistant") if isinstance(turn.get("assistant"), dict) else {}
            assistant_content = _normalize_text(assistant.get("content"))
            tool_calls = assistant.get("tool_calls") if isinstance(assistant.get("tool_calls"), list) else []
            tool_text = _render_hermes_tool_calls(tool_calls)

            merged = "\n".join(part for part in [assistant_content, tool_text] if part)
            if merged:
                messages.append({"role": "assistant", "content": merged})

            tool_responses = turn.get("tool_responses") if isinstance(turn.get("tool_responses"), list) else []
            for payload in tool_responses:
                compact = {
                    "ok": payload.get("ok") if isinstance(payload, dict) else None,
                    "result": payload.get("result") if isinstance(payload, dict) else None,
                    "error": payload.get("error") if isinstance(payload, dict) else None,
                }
                messages.append({"role": "tool", "content": json.dumps(compact, ensure_ascii=False)})

    final_answer = _normalize_text(trajectory.get("final_answer") or record.get("prediction"))
    if final_answer:
        if not messages or messages[-1].get("role") != "assistant" or _normalize_text(messages[-1].get("content")) != final_answer:
            messages.append({"role": "assistant", "content": final_answer})

    return messages, images, tool_defs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and convert trajectories for SFT warmup.")
    parser.add_argument("--trajectory-jsonl", required=True, help="Path to trajectories.jsonl")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--pred-file", default=None, help="Prediction table (xlsx/csv/tsv/json/jsonl/pkl) for correctness filtering")
    parser.add_argument("--require-correct", action="store_true", help="Keep only rows where prediction matches ground truth")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit output sample count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _read_jsonl(args.trajectory_jsonl)

    pred_table = {}
    if args.pred_file:
        pred_table = _load_prediction_table(args.pred_file)

    rows = []
    for record in records:
        task_id = record.get("task_id")
        if args.require_correct and pred_table:
            pred_row = pred_table.get(task_id, {})
            prediction = pred_row.get("prediction", record.get("prediction"))
            ground_truth = pred_row.get("answer", pred_row.get("gt", pred_row.get("label", "")))
            if not _is_correct(prediction, ground_truth):
                continue
        elif args.require_correct and not pred_table:
            # No supervision source to validate correctness.
            continue

        messages, images, tools = _trajectory_to_messages(record)
        rows.append(
            {
                "messages": messages,
                "images": images,
                "tools": tools,
                "task_id": task_id,
                "dataset": record.get("dataset"),
            }
        )

        if args.max_samples > 0 and len(rows) >= args.max_samples:
            break

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)

    print(
        json.dumps(
            {
                "input_records": len(records),
                "output_records": len(rows),
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
