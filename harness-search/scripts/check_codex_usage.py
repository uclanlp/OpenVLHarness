#!/usr/bin/env python3
"""Check whether `codex exec --json` emits usage and reset metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def local_time(epoch_seconds: Any) -> str:
    try:
        return datetime.fromtimestamp(int(epoch_seconds)).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        return "unknown"


def parse_token_count_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    if event.get("type") != "event_msg":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict) or payload.get("type") != "token_count":
        return None
    return payload


def summarize_usage(payload: Dict[str, Any]) -> str:
    rate_limits = payload.get("rate_limits") or {}
    info = payload.get("info") or {}
    total_usage = info.get("total_token_usage") or {}
    last_usage = info.get("last_token_usage") or {}

    lines = ["Codex usage event found:"]
    for name in ("primary", "secondary"):
        limit = rate_limits.get(name)
        if not isinstance(limit, dict):
            continue
        used = limit.get("used_percent")
        remaining = None
        try:
            remaining = max(0.0, 100.0 - float(used))
        except (TypeError, ValueError):
            pass
        lines.append(f"- {name}:")
        lines.append(f"  used_percent: {used}")
        if remaining is not None:
            lines.append(f"  remaining_percent: {remaining:.1f}")
        lines.append(f"  window_minutes: {limit.get('window_minutes')}")
        lines.append(f"  resets_at: {limit.get('resets_at')} ({local_time(limit.get('resets_at'))})")

    if total_usage:
        lines.append(f"- total_token_usage: {json.dumps(total_usage, sort_keys=True)}")
    if last_usage:
        lines.append(f"- last_token_usage: {json.dumps(last_usage, sort_keys=True)}")
    return "\n".join(lines)


def run_codex(args: argparse.Namespace) -> int:
    command = [
        args.codex_bin,
        "exec",
        "--json",
        "--skip-git-repo-check",
    ]
    if args.cd:
        command.extend(["--cd", str(args.cd)])
    if args.model:
        command.extend(["--model", args.model])
    if args.reasoning_effort:
        command.extend(["--config", f'model_reasoning_effort="{args.reasoning_effort}"'])
    command.append(args.prompt)

    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    latest_usage: Optional[Dict[str, Any]] = None
    for line in process.stdout:
        if args.print_json:
            print(line, end="")
        usage = parse_token_count_line(line)
        if usage is not None:
            latest_usage = usage

    returncode = process.wait()
    print(f"\ncodex exec returncode: {returncode}")
    if latest_usage is None:
        print("No token_count event found.")
        return 1 if returncode == 0 else returncode

    print(summarize_usage(latest_usage))
    return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Codex exec JSONL usage metadata.")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--cd", type=Path, default=Path.cwd(), help="Working directory passed to codex exec --cd.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--prompt", default="Reply with exactly: ok")
    parser.add_argument("--print-json", action="store_true", help="Also print raw JSONL events.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_codex(parse_args()))
