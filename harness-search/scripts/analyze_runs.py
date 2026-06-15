#!/usr/bin/env python3
"""Analyze harness-search experiment artifacts.

The analyzer is intentionally best-effort. It understands the common artifacts
from `run_experience_learning.py` and `run_benchmark.py`, but it should still
produce useful summaries when only a subset of files exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_TRACKER = Path("harness-search/performance_tracker.csv")


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_text(path: Path, limit: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > limit:
        return text[:limit] + "\n[truncated]"
    return text


def coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0.5
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "correct", "hit"}:
            return True
        if low in {"0", "false", "no", "n", "incorrect", "miss"}:
            return False
        try:
            return float(low) > 0.5
        except ValueError:
            return None
    return None


def first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def shallow_find(mapping: Any, keys: Iterable[str], max_depth: int = 3) -> Any:
    key_set = set(keys)
    queue: List[Tuple[Any, int]] = [(mapping, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > max_depth:
            continue
        if isinstance(node, dict):
            for key, value in node.items():
                if key in key_set:
                    return value
                if isinstance(value, (dict, list)):
                    queue.append((value, depth + 1))
        elif isinstance(node, list):
            for value in node:
                if isinstance(value, (dict, list)):
                    queue.append((value, depth + 1))
    return None


def infer_dataset_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "full_trajectories" in parts:
        idx = parts.index("full_trajectories")
        if idx >= 2:
            maybe_trial = parts[idx - 1]
            maybe_dataset = parts[idx - 2]
            if not maybe_trial.startswith("trial_"):
                return maybe_trial
            return maybe_dataset
    if path.parent.name == "instance_lessons" and len(parts) >= 3:
        return Path(*parts[:-2]).name
    return path.parent.name


def load_critiques(roots: List[Path]) -> Dict[Tuple[str, str], Dict[str, str]]:
    critiques: Dict[Tuple[str, str], Dict[str, str]] = {}
    for root in roots:
        for path in root.glob("**/instance_lessons/*.txt"):
            dataset = infer_dataset_from_path(path)
            task_id = path.stem
            critiques[(dataset, task_id)] = {
                "path": str(path),
                "text": safe_text(path, limit=3000),
            }
    return critiques


def extract_tool_records(chat_history: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    calls: List[Dict[str, Any]] = []
    responses: List[Dict[str, Any]] = []
    if not isinstance(chat_history, list):
        return calls, responses
    for turn in chat_history:
        if not isinstance(turn, dict):
            continue
        assistant = turn.get("assistant")
        if isinstance(assistant, dict):
            for call in assistant.get("tool_calls") or []:
                if isinstance(call, dict):
                    calls.append({
                        "turn": turn.get("turn"),
                        "name": call.get("name"),
                        "args": call.get("args"),
                        "parse_error": call.get("parse_error"),
                    })
        for payload in turn.get("tool_responses") or []:
            if isinstance(payload, dict):
                error = payload.get("error")
                code = error.get("code") if isinstance(error, dict) else None
                responses.append({
                    "turn": turn.get("turn"),
                    "tool_name": payload.get("tool_name"),
                    "ok": payload.get("ok"),
                    "error_code": code,
                    "error": error,
                    "result_type": type(payload.get("result")).__name__,
                })
    return calls, responses


def extract_message_previews(messages: Any, limit: int = 6) -> List[str]:
    previews: List[str] = []
    if not isinstance(messages, list):
        return previews
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            pieces = [
                str(part.get("text"))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            text = "\n".join(pieces)
        if "Tool `" in text or text.startswith("Environment Update"):
            previews.append(text[:500].replace("\n", "\\n"))
            if len(previews) >= limit:
                break
    return previews


def analyze_trajectory(path: Path, critiques: Dict[Tuple[str, str], Dict[str, str]]) -> Optional[Dict[str, Any]]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        return None

    trajectory = raw.get("trajectory") if isinstance(raw.get("trajectory"), dict) else raw
    task_id = str(first_present(raw, ["task_id", "index"]) or path.stem)
    dataset = str(first_present(raw, ["dataset", "dataset_name"]) or infer_dataset_from_path(path))
    prediction = first_present(raw, ["prediction", "final_answer", "response"])
    if prediction is None and isinstance(trajectory, dict):
        prediction = first_present(trajectory, ["prediction", "final_answer", "response"])
    correct = coerce_bool(first_present(raw, ["is_correct", "correct", "hit", "score", "match"]))
    if correct is None:
        correct = coerce_bool(shallow_find(raw, ["is_correct", "correct", "hit", "score", "match"]))

    chat_history = trajectory.get("chat_history") if isinstance(trajectory, dict) else None
    messages = trajectory.get("messages") if isinstance(trajectory, dict) else raw.get("messages")
    tool_calls, tool_responses = extract_tool_records(chat_history)
    error_codes = [
        str(item.get("error_code"))
        for item in tool_responses
        if item.get("error_code")
    ]
    parse_error_codes = [
        str(call["parse_error"].get("code"))
        for call in tool_calls
        if isinstance(call.get("parse_error"), dict) and call["parse_error"].get("code")
    ]
    critique = critiques.get((dataset, task_id)) or critiques.get(("", task_id))

    return {
        "path": str(path),
        "dataset": dataset,
        "task_id": task_id,
        "is_correct": correct,
        "prediction_preview": str(prediction or "")[:300],
        "tool_call_count": len(tool_calls),
        "tool_names": [str(call.get("name")) for call in tool_calls if call.get("name")],
        "tool_error_codes": error_codes,
        "parse_error_codes": parse_error_codes,
        "tool_result_message_previews": extract_message_previews(messages),
        "critique_path": critique.get("path") if critique else None,
        "critique_preview": (critique.get("text", "")[:600] if critique else ""),
    }


def discover_accuracy_trends(roots: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for root in roots:
        for path in root.glob("**/accuracy_trends.json"):
            if path in seen:
                continue
            seen.add(path)
            payload = read_json(path)
            if not isinstance(payload, list):
                continue
            dataset = path.parent.name
            for item in payload:
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "dataset": dataset,
                    "trial": item.get("trial"),
                    "accuracy": item.get("accuracy"),
                    "correct": item.get("correct"),
                    "total": item.get("total"),
                    "path": str(path),
                })
    return rows


def discover_experience_files(roots: List[Path]) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    seen = set()
    for root in roots:
        for path in root.glob("**/experience*.md"):
            if path in seen:
                continue
            seen.add(path)
            files.append({
                "path": str(path),
                "dataset": path.parent.name,
                "preview": safe_text(path, limit=2000),
            })
    return files


def collect_trajectories(roots: List[Path], max_trajectories: int) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for root in roots:
        for path in root.glob("**/full_trajectories/*.json"):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
            if len(paths) >= max_trajectories:
                return paths
    return paths


def build_outcome_map(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Optional[bool]]:
    out = {}
    for record in records:
        key = (str(record.get("dataset") or ""), str(record.get("task_id") or ""))
        out[key] = record.get("is_correct")
    return out


def compare_records(
    tool_records: List[Dict[str, Any]],
    reference_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tool_map = build_outcome_map(tool_records)
    ref_map = build_outcome_map(reference_records)
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    examples: List[Dict[str, Any]] = []
    for key, tool_correct in tool_map.items():
        ref_correct = ref_map.get(key)
        if tool_correct is None or ref_correct is None:
            continue
        dataset, task_id = key
        if tool_correct and not ref_correct:
            bucket = "improved"
        elif ref_correct and not tool_correct:
            bucket = "regressed"
        else:
            bucket = "same_correct" if tool_correct else "same_wrong"
        summary[dataset][bucket] += 1
        if bucket in {"improved", "regressed"} and len(examples) < 30:
            examples.append({
                "dataset": dataset,
                "task_id": task_id,
                "bucket": bucket,
            })
    return {
        "by_dataset": {dataset: dict(counts) for dataset, counts in summary.items()},
        "examples": examples,
    }


def tokenize_for_patterns(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_\\-]{2,}", text.lower())


def make_recommendations(
    records: List[Dict[str, Any]],
    experience_files: List[Dict[str, Any]],
    comparison: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    failed = [r for r in records if r.get("is_correct") is False]
    no_tool_failures = [r for r in failed if int(r.get("tool_call_count") or 0) == 0]
    error_counter = Counter(
        code
        for r in records
        for code in (r.get("tool_error_codes") or []) + (r.get("parse_error_codes") or [])
    )
    tool_counter = Counter(
        name for r in records for name in (r.get("tool_names") or []) if name
    )

    if no_tool_failures:
        recs.append({
            "target_surface": "system_prompt",
            "priority": "medium",
            "reason": (
                f"{len(no_tool_failures)} inspected failures used no tools. "
                "Review tool-use trigger policy before changing execution logic."
            ),
            "evidence": [r["path"] for r in no_tool_failures[:8]],
        })

    if error_counter:
        top_errors = error_counter.most_common(8)
        target = "system_prompt"
        if any(code in {"invalid_tool_arguments_json", "invalid_tool_arguments_type"} for code, _ in top_errors):
            target = "tool_description"
        if any(code == "unresolved_image_reference" for code, _ in top_errors):
            target = "system_prompt"
        recs.append({
            "target_surface": target,
            "priority": "high",
            "reason": f"Tool or parse errors appeared in inspected trajectories: {top_errors}.",
            "evidence": [
                r["path"]
                for r in records
                if (r.get("tool_error_codes") or []) or (r.get("parse_error_codes") or [])
            ][:10],
        })

    critique_text = "\n".join(r.get("critique_preview") or "" for r in records)
    critique_terms = Counter(tokenize_for_patterns(critique_text))
    informative_terms = [
        item for item in critique_terms.most_common(20)
        if item[0] not in {"the", "and", "for", "with", "that", "this", "tool", "should"}
    ]
    if informative_terms:
        recs.append({
            "target_surface": "tool_result_message",
            "priority": "medium",
            "reason": (
                "Instance critiques contain repeated action/result interpretation terms: "
                f"{informative_terms[:10]}."
            ),
            "evidence": [
                r.get("critique_path")
                for r in records
                if r.get("critique_path")
            ][:10],
        })

    if comparison:
        for dataset, counts in comparison.get("by_dataset", {}).items():
            regressions = int(counts.get("regressed", 0))
            improvements = int(counts.get("improved", 0))
            if regressions > improvements:
                recs.append({
                    "target_surface": "system_prompt",
                    "priority": "high",
                    "reason": (
                        f"{dataset}: tool-enabled run regressed on {regressions} "
                        f"instances and improved on {improvements}."
                    ),
                    "evidence": [
                        ex for ex in comparison.get("examples", [])
                        if ex.get("dataset") == dataset
                    ][:10],
                })

    if not recs and (records or experience_files):
        recs.append({
            "target_surface": "no_action",
            "priority": "low",
            "reason": "No strong repeated harness-text pattern was detected in inspected artifacts.",
            "evidence": [],
        })

    if tool_counter:
        for rec in recs:
            rec.setdefault("context", {})["most_used_tools"] = tool_counter.most_common(10)
    return recs


def render_summary(
    *,
    roots: List[Path],
    accuracy_rows: List[Dict[str, Any]],
    experience_files: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    comparison: Optional[Dict[str, Any]],
) -> str:
    lines: List[str] = ["# Harness Search Analysis", ""]
    lines.append("## Artifact Roots")
    for root in roots:
        lines.append(f"- `{root}`")
    lines.append("")

    lines.append("## Accuracy Trends")
    if accuracy_rows:
        for row in accuracy_rows:
            lines.append(
                "- {dataset} trial {trial}: {accuracy}% ({correct}/{total})".format(
                    dataset=row.get("dataset"),
                    trial=row.get("trial"),
                    accuracy=row.get("accuracy"),
                    correct=row.get("correct"),
                    total=row.get("total"),
                )
            )
    else:
        lines.append("- No `accuracy_trends.json` files found.")
    lines.append("")

    lines.append("## Trajectory Signals")
    lines.append(f"- Inspected trajectories: {len(records)}")
    incorrect = sum(1 for r in records if r.get("is_correct") is False)
    unknown = sum(1 for r in records if r.get("is_correct") is None)
    lines.append(f"- Incorrect: {incorrect}; unknown correctness: {unknown}")
    tool_counter = Counter(name for r in records for name in (r.get("tool_names") or []))
    if tool_counter:
        lines.append(f"- Most used tools: {tool_counter.most_common(10)}")
    error_counter = Counter(
        code
        for r in records
        for code in (r.get("tool_error_codes") or []) + (r.get("parse_error_codes") or [])
    )
    if error_counter:
        lines.append(f"- Tool/parse errors: {error_counter.most_common(10)}")
    lines.append("")

    lines.append("## Experience Files")
    if experience_files:
        for item in experience_files[:20]:
            lines.append(f"- `{item['path']}`")
    else:
        lines.append("- No experience markdown files found.")
    lines.append("")

    if comparison:
        lines.append("## Tool Vs No-Tool Comparison")
        if comparison.get("by_dataset"):
            for dataset, counts in comparison["by_dataset"].items():
                lines.append(f"- {dataset}: {counts}")
        else:
            lines.append("- No comparable per-instance correctness records found.")
        lines.append("")

    lines.append("## Recommendations")
    for rec in recommendations:
        lines.append(
            f"- `{rec.get('target_surface')}` / {rec.get('priority')}: {rec.get('reason')}"
        )
    lines.append("")
    return "\n".join(lines)


def append_tracker(
    tracker_path: Path,
    *,
    round_id: str,
    mode: str,
    agent_config: str,
    reference_run: str,
    num_examples: str,
    seed: str,
    max_tries: str,
    accuracy_rows: List[Dict[str, Any]],
) -> None:
    if not accuracy_rows:
        return
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    exists = tracker_path.exists()
    fieldnames = [
        "round_id", "dataset", "category", "mode", "agent_config", "reference_run",
        "num_examples", "seed", "max_tries", "trial", "accuracy", "total",
        "correct", "delta_vs_baseline", "run_dir", "notes",
    ]
    with tracker_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in accuracy_rows:
            writer.writerow({
                "round_id": round_id,
                "dataset": row.get("dataset", ""),
                "category": "",
                "mode": mode,
                "agent_config": agent_config,
                "reference_run": reference_run,
                "num_examples": num_examples,
                "seed": seed,
                "max_tries": max_tries,
                "trial": row.get("trial", ""),
                "accuracy": row.get("accuracy", ""),
                "total": row.get("total", ""),
                "correct": row.get("correct", ""),
                "delta_vs_baseline": "",
                "run_dir": str(Path(str(row.get("path", ""))).parent),
                "notes": "analyze_runs.py",
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze harness search artifacts.")
    parser.add_argument("--round-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, action="append", default=[])
    parser.add_argument("--reference-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tracker-csv", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--append-tracker", action="store_true")
    parser.add_argument("--round-id", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--agent-config", default="")
    parser.add_argument("--num-examples", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--max-tries", default="")
    parser.add_argument("--max-trajectories", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots: List[Path] = []
    if args.round_dir:
        roots.append(args.round_dir)
    roots.extend(args.artifact_dir)
    roots = [path.resolve() for path in roots if path]
    if not roots:
        raise SystemExit("Provide --round-dir or at least one --artifact-dir.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (args.round_dir / "analysis") if args.round_dir else (roots[0] / "analysis")
    output_dir = output_dir.resolve()

    critiques = load_critiques(roots)
    trajectory_paths = collect_trajectories(roots, args.max_trajectories)
    records = [
        record for path in trajectory_paths
        if (record := analyze_trajectory(path, critiques)) is not None
    ]

    reference_records: List[Dict[str, Any]] = []
    if args.reference_run_dir:
        reference_root = args.reference_run_dir.resolve()
        reference_critiques = load_critiques([reference_root])
        reference_records = [
            record for path in collect_trajectories([reference_root], args.max_trajectories)
            if (record := analyze_trajectory(path, reference_critiques)) is not None
        ]

    comparison = compare_records(records, reference_records) if reference_records else None
    accuracy_rows = discover_accuracy_trends(roots)
    experience_files = discover_experience_files(roots)
    recommendations = make_recommendations(records, experience_files, comparison)

    write_jsonl(output_dir / "instance_analysis.jsonl", records)
    write_json(output_dir / "recommendations.json", {
        "roots": [str(root) for root in roots],
        "reference_run_dir": str(args.reference_run_dir.resolve()) if args.reference_run_dir else None,
        "recommendations": recommendations,
        "comparison": comparison,
    })
    summary = render_summary(
        roots=roots,
        accuracy_rows=accuracy_rows,
        experience_files=experience_files,
        records=records,
        recommendations=recommendations,
        comparison=comparison,
    )
    (output_dir / "analysis_summary.md").write_text(summary, encoding="utf-8")

    if args.append_tracker:
        append_tracker(
            args.tracker_csv,
            round_id=args.round_id,
            mode=args.mode,
            agent_config=args.agent_config,
            reference_run=str(args.reference_run_dir or ""),
            num_examples=args.num_examples,
            seed=args.seed,
            max_tries=args.max_tries,
            accuracy_rows=accuracy_rows,
        )

    print(f"Wrote analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
