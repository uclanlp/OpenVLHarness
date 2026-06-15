#!/usr/bin/env python3
"""Simple tree-search controller for harness-text optimization."""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from state_store import (  # noqa: E402
    ProposalConflict,
    create_base_snapshot,
    create_child_state_from_proposals,
    create_initial_state,
    delete_scratch_repo,
    materialize_state,
    rank_states,
    read_json,
    update_state,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOT = REPO_ROOT / "harness-search"
DEFAULT_WORK_ROOT = Path("/local/<user>/tool_vlm/harness_search_logs")
DEFAULT_AGENT_CONFIG = "harness-search/configs/run_benchmark_agent.yaml"
DEFAULT_JUDGE_CONFIG = "harness-search/configs/judge.yaml"
DEFAULT_CODEX_BIN = "/home/<user>/software/codex-latest-binary"
CODEX_STATUS_PROBE = SEARCH_ROOT / "scripts" / "check_codex_status_pty.py"


def utcish_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utcish_now()}] {message}", flush=True)


def build_campaign_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in prefix)
    return f"{safe}_{stamp}" if safe else f"search_{stamp}"


def create_campaign_root(work_root: Path, campaign_id: str, *, explicit: bool) -> Tuple[str, Path]:
    if explicit:
        campaign_root = work_root / campaign_id
        campaign_root.mkdir(parents=True, exist_ok=False)
        return campaign_id, campaign_root

    suffix = 1
    while True:
        candidate_id = campaign_id if suffix == 1 else f"{campaign_id}_{suffix}"
        campaign_root = work_root / candidate_id
        try:
            campaign_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        return candidate_id, campaign_root


def command_to_display(command: List[str]) -> str:
    pieces = []
    for part in command:
        text = str(part)
        pieces.append(json.dumps(text) if " " in text else text)
    return " ".join(pieces)


def run_command(
    *,
    command: List[str],
    cwd: Path,
    state: Dict[str, Any],
    label: str,
    dry_run: bool,
    input_text: Optional[str] = None,
    log_path: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    state_path: Optional[Path] = None,
    timeout_seconds: Optional[float] = None,
) -> int:
    entry = {
        "label": label,
        "command": command,
        "display": command_to_display(command),
        "cwd": str(cwd),
        "status": "planned" if dry_run else "running",
        "started_at": utcish_now(),
    }
    if log_path is not None:
        entry["log_path"] = str(log_path)
    state.setdefault("commands", []).append(entry)
    if state_path is not None:
        write_json(state_path, state)
    log_suffix = f" (log: {log_path})" if log_path is not None else ""
    log(f"START {label}: {entry['display']}{log_suffix}")
    if dry_run:
        log(f"PLAN {label}")
        return 0

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_handle:
            header = f"$ {entry['display']}\n".encode("utf-8", errors="replace")
            log_handle.write(header)
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=bool(timeout_seconds and timeout_seconds > 0),
            )
            if input_text is not None and process.stdin is not None:
                process.stdin.write(input_text.encode("utf-8"))
                process.stdin.close()
            assert process.stdout is not None
            deadline = time.monotonic() + float(timeout_seconds) if timeout_seconds and timeout_seconds > 0 else None
            timed_out = False
            while True:
                if deadline is not None and time.monotonic() >= deadline and process.poll() is None:
                    timed_out = True
                    entry["timeout_seconds"] = timeout_seconds
                    entry["status"] = "timeout"
                    log(f"TIMEOUT {label}: exceeded {timeout_seconds}s; terminating process")
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except Exception:
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except Exception:
                            process.kill()
                    break

                readable, _, _ = select.select([process.stdout], [], [], 0.5)
                if readable:
                    chunk = process.stdout.read1(4096) if hasattr(process.stdout, "read1") else process.stdout.read(4096)
                    if not chunk:
                        if process.poll() is not None:
                            break
                        continue
                    log_handle.write(chunk)
                    log_handle.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                elif process.poll() is not None:
                    remaining = process.stdout.read()
                    if remaining:
                        log_handle.write(remaining)
                        log_handle.flush()
                        sys.stdout.buffer.write(remaining)
                        sys.stdout.buffer.flush()
                    break
            returncode = process.wait()
            if timed_out:
                returncode = 124
    else:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            input=input_text,
            text=True,
            env=env,
            check=False,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        returncode = result.returncode

    entry["returncode"] = returncode
    entry["finished_at"] = utcish_now()
    if entry.get("status") != "timeout":
        entry["status"] = "complete" if returncode == 0 else "failed"
    if state_path is not None:
        write_json(state_path, state)
    log(f"DONE {label}: returncode={returncode}")
    return int(returncode)


def coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0.5
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "correct", "hit"}:
            return True
        if text in {"0", "false", "no", "n", "incorrect", "miss"}:
            return False
        try:
            return float(text) > 0.5
        except ValueError:
            return None
    return None


def coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def shallow_find(payload: Any, keys: set[str], max_depth: int = 4) -> Any:
    queue: List[Tuple[Any, int]] = [(payload, 0)]
    while queue:
        node, depth = queue.pop(0)
        if depth > max_depth:
            continue
        if isinstance(node, dict):
            for key, value in node.items():
                if key in keys:
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
        if idx >= 1:
            return parts[idx - 1]
    return path.parent.name


def extract_eval(payload: Any) -> Tuple[Optional[bool], Optional[float]]:
    if not isinstance(payload, dict):
        return None, None
    eval_payload = payload.get("eval")
    correct = None
    score = None
    if isinstance(eval_payload, dict):
        correct = coerce_bool(eval_payload.get("correct"))
        score = coerce_float(eval_payload.get("score"))
    if correct is None:
        correct = coerce_bool(shallow_find(payload, {"is_correct", "correct", "hit", "match"}))
    if score is None:
        score = coerce_float(shallow_find(payload, {"score"}))
    if correct is None and score is not None:
        correct = score > 0.5
    if score is None and correct is not None:
        score = float(correct)
    return correct, score


def aggregate_state_metrics(artifact_root: Path, datasets: List[str]) -> Dict[str, Any]:
    grouped: Dict[str, List[Tuple[Optional[bool], Optional[float]]]] = {dataset: [] for dataset in datasets}
    for path in artifact_root.glob("**/full_trajectories/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dataset = str(payload.get("dataset") or infer_dataset_from_path(path))
        if dataset not in grouped:
            grouped.setdefault(dataset, [])
        grouped[dataset].append(extract_eval(payload))

    rows: List[Dict[str, Any]] = []
    dataset_scores: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        values = grouped.get(dataset, [])
        correct_values = [correct for correct, _ in values if correct is not None]
        if not correct_values:
            raise ValueError(f"no evaluable full_trajectories records for dataset {dataset}")
        correct_count = sum(1 for correct in correct_values if correct)
        total = len(correct_values)
        accuracy = correct_count / total
        row = {
            "dataset": dataset,
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total,
        }
        rows.append(row)
        dataset_scores[dataset] = row

    score = sum(row["accuracy"] for row in rows) / len(rows)
    return {
        "score": score,
        "dataset_scores": dataset_scores,
        "rows": rows,
        "artifact_root": str(artifact_root),
    }


def parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def write_router_env_with_port(source_path: Path, target_path: Path, port: int) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"router env file does not exist: {source_path}")
    lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    output_lines = []
    wrote_port = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped.split("=", 1)[0].strip() == "PORT":
            if not wrote_port:
                output_lines.append(f"PORT={port}")
                wrote_port = True
            continue
        output_lines.append(line)
    if not wrote_port:
        output_lines.append(f"PORT={port}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def write_agent_config_with_tool_base(source_path: Path, target_path: Path, tool_base: str) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"agent config does not exist: {source_path}")
    config = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"agent config must be a YAML mapping: {source_path}")
    agent_kwargs = config.setdefault("agent_kwargs", {})
    if not isinstance(agent_kwargs, dict):
        raise ValueError(f"agent_kwargs must be a YAML mapping: {source_path}")
    agent_kwargs["tool_base"] = tool_base
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def materialize_campaign_runtime_config(args: argparse.Namespace) -> None:
    source_router_env = resolve_repo_path(args.router_env_file)
    source_agent_config = resolve_repo_path(args.agent_config)
    source_env_values = parse_env_file(source_router_env)
    router_port = args.router_port or int(source_env_values.get("PORT") or os.environ.get("PORT") or "8103")
    tool_base = f"http://localhost:{router_port}"

    config_dir = args.campaign_root / "configs"
    router_env_path = config_dir / "router.env"
    agent_config_path = config_dir / "agent.yaml"
    write_router_env_with_port(source_router_env, router_env_path, router_port)
    write_agent_config_with_tool_base(source_agent_config, agent_config_path, tool_base)

    args.source_router_env_file = str(source_router_env)
    args.source_agent_config = str(source_agent_config)
    args.router_port = router_port
    args.tool_base = tool_base
    args.router_env_file = str(router_env_path)
    args.agent_config = str(agent_config_path)


def build_benchmark_env(repo_root: Path, tool_base: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(parse_env_file(repo_root / ".env"))
    env["PYTHONUNBUFFERED"] = "1"
    env["TOOL_SERVER_URL"] = tool_base
    return env


def make_eval_runtime(args: argparse.Namespace, eval_dir: Path, router_port: Optional[int]) -> Dict[str, Any]:
    if router_port is None:
        return {
            "router_port": args.router_port,
            "tool_base": args.tool_base,
            "router_env_file": args.router_env_file,
            "router_state_dir": str(args.router_state_dir),
            "agent_config": args.agent_config,
            "benchmark_env": args.benchmark_env,
        }

    tool_base = f"http://localhost:{router_port}"
    runtime_dir = eval_dir / "runtime"
    router_env_path = runtime_dir / "router.env"
    agent_config_path = runtime_dir / "agent.yaml"
    write_router_env_with_port(Path(args.source_router_env_file), router_env_path, router_port)
    write_agent_config_with_tool_base(Path(args.source_agent_config), agent_config_path, tool_base)
    return {
        "router_port": router_port,
        "tool_base": tool_base,
        "router_env_file": str(router_env_path),
        "router_state_dir": str(runtime_dir / "router_state"),
        "agent_config": str(agent_config_path),
        "benchmark_env": build_benchmark_env(REPO_ROOT, tool_base),
    }


def router_host(args: argparse.Namespace) -> str:
    values = parse_env_file(Path(args.router_env_file))
    return values.get("HOST") or os.environ.get("HOST") or "127.0.0.1"


def port_is_available(host: str, port: int) -> Tuple[bool, str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
    except OSError as exc:
        return False, str(exc)
    return True, ""


def require_router_ports_available(args: argparse.Namespace, ports: Iterable[int], context: str) -> None:
    host = router_host(args)
    busy = []
    for port in ports:
        ok, reason = port_is_available(host, int(port))
        if not ok:
            busy.append(f"{host}:{port} ({reason})")
    if busy:
        raise RuntimeError(f"{context}: router port(s) unavailable: {', '.join(busy)}")


def init_runbook(path: Path, *, args: argparse.Namespace, campaign_id: str, datasets: List[str]) -> None:
    text = f"""# Harness Search Runbook

## Campaign Setup

| field | value |
| --- | --- |
| campaign_id | `{campaign_id}` |
| datasets | `{", ".join(datasets)}` |
| num_examples_per_dataset | `{args.num_examples}` |
| num_proc | `{args.num_proc}` |
| parallel_states | `{args.parallel_states}` |
| crossover_max_size | `{args.crossover_max_size}` |
| pool_size | `{args.pool_size}` |
| max_proposals | `{args.max_proposals}` |
| router_port | `{args.router_port}` |
| tool_base | `{args.tool_base}` |
| playground_max_tasks | `{args.playground_max_tasks}` |
| playground_trials_per_task | `{args.playground_trials_per_task}` |
| objective | unweighted mean dataset accuracy |
| allowed_edit_surface | `constant.py`, tool `tool.py`, `process_tool_output.py` |

## Current Pool

| state_id | parent | round | avg_accuracy | dataset_scores | patch_summary | status |
| --- | --- | --- | --- | --- | --- | --- |

## Tried Ideas

| round | parent_state | proposal_id | child_state | idea | files | avg_accuracy | dataset_scores | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Lessons
"""
    path.write_text(text, encoding="utf-8")


def build_codex_args(args: argparse.Namespace, cwd: Path, output_last_message: Path) -> List[str]:
    command = [
        args.codex_bin,
        "exec",
        "--cd",
        str(cwd),
        "--skip-git-repo-check",
    ]
    added_dirs = []
    for path in [cwd, args.campaign_root]:
        resolved = Path(path).resolve()
        if resolved not in added_dirs:
            added_dirs.append(resolved)
            command.extend(["--add-dir", str(resolved)])
    command.extend([
        "--sandbox",
        "danger-full-access",
        "--output-last-message",
        str(output_last_message),
    ])
    if args.codex_exec_json_usage:
        command.append("--json")
    if args.codex_model:
        command.extend(["--model", args.codex_model])
    if args.codex_reasoning_effort:
        command.extend(["--config", f'model_reasoning_effort="{args.codex_reasoning_effort}"'])
    command.append("-")
    return command


def codex_phase_log_path(logs_dir: Path, label: str, attempt: int) -> Path:
    if attempt == 1:
        return logs_dir / f"{label}.log"
    return logs_dir / f"{label}.attempt_{attempt:02d}.log"


def max_codex_attempts(args: argparse.Namespace) -> int:
    return 1 + max(0, int(args.codex_exec_retries))


def extract_latest_codex_usage(log_path: Path) -> Optional[Dict[str, Any]]:
    if not log_path.exists():
        return None

    latest: Optional[Dict[str, Any]] = None
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "event_msg":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        latest = payload
    return latest


def record_codex_usage(
    *,
    args: argparse.Namespace,
    log_path: Path,
    state: Dict[str, Any],
    state_path: Optional[Path],
) -> None:
    threshold = float(args.codex_usage_threshold_percent)
    if threshold <= 0:
        return

    usage = extract_latest_codex_usage(log_path)
    if not usage:
        log(f"No Codex token_count usage event found in {log_path}; skipping usage guard update")
        return

    rate_limits = usage.get("rate_limits")
    if not isinstance(rate_limits, dict):
        return
    primary = rate_limits.get("primary")
    if not isinstance(primary, dict):
        return

    used_percent = primary.get("used_percent")
    resets_at = primary.get("resets_at")
    try:
        used = float(used_percent)
        reset_epoch = int(resets_at)
    except (TypeError, ValueError):
        return

    window_minutes = primary.get("window_minutes")
    remaining = max(0.0, 100.0 - used)
    usage_record = {
        "checked_at": utcish_now(),
        "log_path": str(log_path),
        "primary": primary,
        "secondary": rate_limits.get("secondary"),
        "total_token_usage": (usage.get("info") or {}).get("total_token_usage"),
        "last_token_usage": (usage.get("info") or {}).get("last_token_usage"),
    }
    state.setdefault("codex_usage_checks", []).append(usage_record)
    args.last_codex_usage_record = usage_record
    if state_path is not None:
        write_json(state_path, state)

    log(
        "Codex primary usage "
        f"{used:.1f}% used ({remaining:.1f}% remaining"
        + (f", window={window_minutes}m" if window_minutes is not None else "")
        + ")"
    )
    return


def sleep_before_codex_if_needed(args: argparse.Namespace) -> None:
    threshold = float(args.codex_usage_threshold_percent)
    if threshold <= 0:
        return

    if args.codex_status_probe != "off":
        status = probe_codex_status(args)
        if status:
            five_hour = status.get("five_hour")
            if isinstance(five_hour, dict):
                remaining_percent = five_hour.get("remaining_percent")
                reset_epoch = five_hour.get("reset_epoch")
                try:
                    remaining = float(remaining_percent)
                    used = max(0.0, 100.0 - remaining)
                    reset_at_epoch = int(reset_epoch)
                except (TypeError, ValueError):
                    remaining = None
                    used = None
                    reset_at_epoch = None
                if remaining is not None and used is not None and reset_at_epoch is not None:
                    log(f"Codex /status 5h limit: {remaining:.1f}% remaining ({used:.1f}% used)")
                    if used >= threshold:
                        sleep_until_codex_reset(
                            args=args,
                            used=used,
                            threshold=threshold,
                            reset_epoch=reset_at_epoch,
                        )
                    return

    usage_record = getattr(args, "last_codex_usage_record", None)
    if not isinstance(usage_record, dict):
        return
    primary = usage_record.get("primary")
    if not isinstance(primary, dict):
        return
    used_percent = primary.get("used_percent")
    resets_at = primary.get("resets_at")
    try:
        used = float(used_percent)
        reset_epoch = int(resets_at)
    except (TypeError, ValueError):
        return

    if used < threshold:
        return

    sleep_until_codex_reset(
        args=args,
        used=used,
        threshold=threshold,
        reset_epoch=reset_epoch,
    )


def sleep_until_codex_reset(
    *,
    args: argparse.Namespace,
    used: float,
    threshold: float,
    reset_epoch: int,
) -> None:
    sleep_for = reset_epoch - int(time.time()) + int(args.codex_usage_sleep_cushion_seconds)
    if sleep_for <= 0:
        log("Codex primary usage is over threshold, but reset time has already passed")
        args.last_codex_usage_record = None
        return

    reset_time = datetime.fromtimestamp(reset_epoch).isoformat(timespec="seconds")
    log(
        f"Codex primary usage {used:.1f}% reached threshold {threshold:.1f}%; "
        f"sleeping {sleep_for}s until reset at {reset_time}"
    )
    time.sleep(sleep_for)
    args.last_codex_usage_record = None


def probe_codex_status(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    command = [
        sys.executable,
        str(CODEX_STATUS_PROBE),
        "--codex-bin",
        str(args.codex_bin),
        "--cd",
        str(args.base_repo if getattr(args, "base_repo", None) and args.base_repo.exists() else REPO_ROOT),
        "--timeout",
        str(args.codex_status_timeout),
        "--startup-delay",
        str(args.codex_status_startup_delay),
    ]
    if args.codex_model:
        command.extend(["--model", args.codex_model])
    if args.codex_reasoning_effort:
        command.extend(["--reasoning-effort", args.codex_reasoning_effort])

    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = (
            "Codex /status PTY probe failed; "
            f"returncode={result.returncode}; stderr={result.stderr.strip() or '<empty>'}"
        )
        if args.codex_status_failure_mode == "fail":
            raise RuntimeError(message)
        log(f"WARNING: {message}")
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        message = f"Codex /status PTY probe returned non-JSON output: {exc}"
        if args.codex_status_failure_mode == "fail":
            raise RuntimeError(message)
        log(f"WARNING: {message}")
        return None
    if not payload.get("ok") or not isinstance(payload.get("status"), dict):
        message = f"Codex /status PTY probe returned unsuccessful payload: {payload}"
        if args.codex_status_failure_mode == "fail":
            raise RuntimeError(message)
        log(f"WARNING: {message}")
        return None
    args.last_codex_status = payload["status"]
    return payload["status"]


def read_template(name: str) -> str:
    return (SEARCH_ROOT / name).read_text(encoding="utf-8")


def render_analysis_prompt(
    *,
    args: argparse.Namespace,
    parent_state: Dict[str, Any],
    round_dir: Path,
    analysis_dir: Path,
) -> str:
    return read_template("CODEX_ANALYSIS_PROMPT.md").format(
        campaign_root=args.campaign_root,
        runbook=args.campaign_root / "RUNBOOK.md",
        state_id=parent_state["state_id"],
        state_json=args.states_root / parent_state["state_id"] / "state.json",
        eval_dir=args.states_root / parent_state["state_id"] / "eval",
        performance_summary=args.states_root / parent_state["state_id"] / "eval" / "performance_summary.json",
        analysis_dir=analysis_dir,
        analysis_summary=analysis_dir / "analysis_summary.md",
        recommendations_json=analysis_dir / "recommendations.json",
        instance_analysis_jsonl=analysis_dir / "instance_analysis.jsonl",
        datasets=", ".join(args.datasets),
        num_examples=args.num_examples,
        round_dir=round_dir,
    )


def render_proposer_prompt(
    *,
    args: argparse.Namespace,
    parent_state: Dict[str, Any],
    round_dir: Path,
    analysis_dir: Path,
    proposals_dir: Path,
) -> str:
    return read_template("CODEX_PROPOSER_PROMPT.md").format(
        campaign_root=args.campaign_root,
        runbook=args.campaign_root / "RUNBOOK.md",
        state_id=parent_state["state_id"],
        state_json=args.states_root / parent_state["state_id"] / "state.json",
        eval_dir=args.states_root / parent_state["state_id"] / "eval",
        analysis_summary=analysis_dir / "analysis_summary.md",
        proposals_dir=proposals_dir,
        max_proposals=args.max_proposals,
        playground_max_tasks=args.playground_max_tasks,
        playground_trials_per_task=args.playground_trials_per_task,
        playground_replay_script="harness-search/scripts/image_q_a.py",
        router_env_file=args.router_env_file,
        router_state_dir=args.router_state_dir,
        router_health_timeout=args.router_health_timeout,
        agent_config=args.agent_config,
        datasets=", ".join(args.datasets),
        num_examples=args.num_examples,
        round_dir=round_dir,
    )


def render_bookkeeper_prompt(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    round_state_path: Path,
) -> str:
    template_name = (
        "CODEX_BOOKKEEPER_PROMPT_WITH_CROSSOVERS.md"
        if args.crossover_max_size > 1
        else "CODEX_BOOKKEEPER_PROMPT.md"
    )
    return read_template(template_name).format(
        campaign_root=args.campaign_root,
        runbook=args.campaign_root / "RUNBOOK.md",
        round_dir=round_dir,
        round_state=round_state_path,
        campaign_state=args.campaign_root / "campaign_state.json",
    )


def router_command(args: argparse.Namespace, action: str, runtime: Optional[Dict[str, Any]] = None) -> List[str]:
    router_env_file = runtime["router_env_file"] if runtime else args.router_env_file
    router_state_dir = runtime["router_state_dir"] if runtime else str(args.router_state_dir)
    return [
        sys.executable,
        "harness-search/scripts/manage_router.py",
        action,
        "--env-file",
        router_env_file,
        "--state-dir",
        router_state_dir,
        "--health-timeout",
        str(args.router_health_timeout),
    ]


def stop_router(
    args: argparse.Namespace,
    scratch_repo: Path,
    state: Dict[str, Any],
    *,
    dry_run: bool,
    runtime: Optional[Dict[str, Any]] = None,
) -> None:
    run_command(
        command=router_command(args, "stop", runtime),
        cwd=scratch_repo,
        state=state,
        label="router_stop",
        dry_run=dry_run,
        log_path=Path(state.get("log_dir", args.campaign_root / "logs")) / "router_stop.log",
        state_path=Path(state["state_path"]) if state.get("state_path") else None,
    )


def evaluate_state(
    *,
    args: argparse.Namespace,
    state_id: str,
    dry_run: bool,
    router_port: Optional[int] = None,
) -> Dict[str, Any]:
    state_dir = args.states_root / state_id
    eval_dir = state_dir / "eval"
    artifacts_dir = eval_dir / "artifacts"
    logs_dir = eval_dir / "logs"
    scratch_repo = args.campaign_root / "scratch" / f"eval_{state_id}"
    runtime = make_eval_runtime(args, eval_dir, None if dry_run else router_port)
    eval_state: Dict[str, Any] = {
        "state_id": state_id,
        "status": "dry_run" if dry_run else "running",
        "created_at": utcish_now(),
        "log_dir": str(logs_dir),
        "state_path": str(eval_dir / "eval_state.json"),
        "router_port": runtime["router_port"],
        "tool_base": runtime["tool_base"],
        "router_env_file": runtime["router_env_file"],
        "agent_config": runtime["agent_config"],
        "commands": [],
    }
    eval_state_path = eval_dir / "eval_state.json"
    write_json(eval_state_path, eval_state)

    if not dry_run:
        log(f"Materializing state {state_id} for evaluation into {scratch_repo}")
        materialize_state(args.base_repo, state_dir, scratch_repo)
        log(f"Materialized state {state_id}")

    try:
        code = run_command(
            command=router_command(args, "restart", runtime),
            cwd=scratch_repo if not dry_run else REPO_ROOT,
            state=eval_state,
            label="router_restart",
            dry_run=dry_run,
            log_path=logs_dir / "router_restart.log",
            state_path=eval_state_path,
        )
        if code != 0:
            raise RuntimeError("router restart failed")

        command = [
            sys.executable,
            "run_benchmark.py",
            "--config",
            runtime["agent_config"],
            "--judge-config",
            args.judge_config,
            "--data",
            *args.datasets,
            "--num-proc",
            str(args.num_proc),
            "--num-examples",
            str(args.num_examples),
            "--work-dir",
            str(artifacts_dir),
            "--run-tag",
            state_id,
            "--save-html",
            "0",
        ]
        code = run_command(
            command=command,
            cwd=scratch_repo if not dry_run else REPO_ROOT,
            state=eval_state,
            label="benchmark",
            dry_run=dry_run,
            log_path=logs_dir / "benchmark.log",
            env=runtime["benchmark_env"],
            state_path=eval_state_path,
        )
        if code != 0:
            raise RuntimeError("benchmark failed")

        if dry_run:
            metrics = {"score": None, "dataset_scores": {}, "rows": []}
            update_state(state_dir, {"status": "dry_run", "eval_dir": str(eval_dir)})
        else:
            metrics = aggregate_state_metrics(artifacts_dir, args.datasets)
            write_json(eval_dir / "performance_summary.json", metrics)
            write_json(eval_dir / "score.json", metrics)
            update_state(
                state_dir,
                {
                    "status": "evaluated",
                    "score": metrics["score"],
                    "dataset_scores": metrics["dataset_scores"],
                    "eval_dir": str(eval_dir),
                    "performance_summary": str(eval_dir / "performance_summary.json"),
                    "evaluated_at": utcish_now(),
                },
            )

        eval_state["status"] = "dry_run" if dry_run else "complete"
        eval_state["metrics"] = metrics
    except Exception as exc:
        eval_state["status"] = "failed"
        eval_state["error"] = f"{type(exc).__name__}: {exc}"
        update_state(
            state_dir,
            {
                "status": "failed",
                "failure_reason": eval_state["error"],
                "eval_dir": str(eval_dir),
            },
        )
    finally:
        if not dry_run and scratch_repo.exists():
            stop_router(args, scratch_repo, eval_state, dry_run=False, runtime=runtime)
            log(f"Deleting evaluation scratch repo {scratch_repo}")
            delete_scratch_repo(scratch_repo)
        eval_state["finished_at"] = utcish_now()
        write_json(eval_state_path, eval_state)

    return read_json(state_dir / "state.json")


def run_codex_analysis(
    *,
    args: argparse.Namespace,
    parent_state: Dict[str, Any],
    round_dir: Path,
    dry_run: bool,
) -> Tuple[bool, Path]:
    parent_id = parent_state["state_id"]
    parent_round_dir = round_dir / "parents" / parent_id
    analysis_dir = parent_round_dir / "analysis"
    logs_dir = parent_round_dir / "logs"
    scratch_repo = args.campaign_root / "scratch" / f"analysis_{round_dir.name}_{parent_id}"
    analysis_state: Dict[str, Any] = {
        "label": "codex_analysis",
        "parent_state": parent_id,
        "state_path": str(analysis_dir / "analysis_state.json"),
        "commands": [],
    }
    analysis_state_path = analysis_dir / "analysis_state.json"
    write_json(analysis_state_path, analysis_state)

    try:
        if not dry_run and getattr(args, "resume_mode", False) and (analysis_dir / "analysis_summary.md").exists():
            log(f"Resuming: accepting existing analysis for parent {parent_id}")
            analysis_state["status"] = "complete"
            analysis_state["resumed_from_artifacts"] = True
            write_json(analysis_state_path, analysis_state)
            return True, analysis_dir

        prompt = render_analysis_prompt(
            args=args,
            parent_state=parent_state,
            round_dir=round_dir,
            analysis_dir=analysis_dir,
        )
        prompt_path = analysis_dir / "prompt.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

        last_error = None
        for attempt in range(1, max_codex_attempts(args) + 1):
            attempt_scratch = scratch_repo if attempt == 1 else scratch_repo.with_name(f"{scratch_repo.name}_attempt_{attempt:02d}")
            try:
                if not dry_run:
                    log(f"Materializing parent {parent_id} for analysis attempt {attempt} into {attempt_scratch}")
                    materialize_state(args.base_repo, args.states_root / parent_id, attempt_scratch)
                    log(f"Materialized parent {parent_id} for analysis attempt {attempt}")
                    sleep_before_codex_if_needed(args)
                command = build_codex_args(args, attempt_scratch if not dry_run else REPO_ROOT, analysis_dir / "last_message.md")
                codex_log_path = codex_phase_log_path(logs_dir, "codex_analysis", attempt)
                code = run_command(
                    command=command,
                    cwd=attempt_scratch if not dry_run else REPO_ROOT,
                    state=analysis_state,
                    label="codex_analysis",
                    dry_run=dry_run,
                    input_text=prompt,
                    log_path=codex_log_path,
                    state_path=analysis_state_path,
                    timeout_seconds=args.codex_exec_timeout_seconds,
                )
                if not dry_run and args.codex_exec_json_usage:
                    record_codex_usage(
                        args=args,
                        log_path=codex_log_path,
                        state=analysis_state,
                        state_path=analysis_state_path,
                    )
                if code != 0:
                    raise RuntimeError(f"codex analysis failed with returncode={code}")
                if not dry_run and not (analysis_dir / "analysis_summary.md").exists():
                    raise FileNotFoundError(f"missing analysis summary: {analysis_dir / 'analysis_summary.md'}")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                analysis_state.setdefault("attempt_failures", []).append({
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at": utcish_now(),
                })
                write_json(analysis_state_path, analysis_state)
                if attempt >= max_codex_attempts(args):
                    raise
                log(f"Retrying codex_analysis for parent {parent_id} after attempt {attempt} failed: {exc}")
            finally:
                if not dry_run and attempt_scratch.exists():
                    log(f"Deleting analysis scratch repo {attempt_scratch}")
                    delete_scratch_repo(attempt_scratch)
        if last_error is not None:
            raise last_error
        analysis_state["status"] = "dry_run" if dry_run else "complete"
        ok = True
    except Exception as exc:
        analysis_state["status"] = "failed"
        analysis_state["error"] = f"{type(exc).__name__}: {exc}"
        ok = False
    finally:
        write_json(analysis_state_path, analysis_state)
    return ok, analysis_dir


def validate_proposals(proposals_dir: Path, expected_count: int) -> List[Path]:
    if not (proposals_dir / "proposals.json").exists():
        raise FileNotFoundError(f"missing proposals.json under {proposals_dir}")
    if not (proposals_dir / "playground" / "HYPOTHESES.md").exists():
        raise FileNotFoundError("missing playground/HYPOTHESES.md")
    if not (proposals_dir / "playground" / "PLAYGROUND_SUMMARY.md").exists():
        raise FileNotFoundError("missing playground/PLAYGROUND_SUMMARY.md")

    expected_names = [f"p{i:03d}" for i in range(1, expected_count + 1)]
    proposal_dirs = [proposals_dir / name for name in expected_names]
    missing = [path.name for path in proposal_dirs if not path.is_dir()]
    extra = sorted(
        path.name
        for path in proposals_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("p")
        and path.name[1:].isdigit()
        and path.name not in expected_names
    )
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            f"expected proposal dirs {expected_names}; " + ", ".join(details)
        )
    for proposal_dir in proposal_dirs:
        if not (proposal_dir / "metadata.json").exists():
            raise FileNotFoundError(f"missing metadata.json: {proposal_dir}")
        if not (proposal_dir / "PROPOSAL.md").exists():
            raise FileNotFoundError(f"missing PROPOSAL.md: {proposal_dir}")
        if not (proposal_dir / "files").exists():
            raise FileNotFoundError(f"missing files/: {proposal_dir}")
    return proposal_dirs


def run_codex_proposer(
    *,
    args: argparse.Namespace,
    parent_state: Dict[str, Any],
    round_dir: Path,
    analysis_dir: Path,
    dry_run: bool,
) -> Tuple[bool, Path, List[Path]]:
    parent_id = parent_state["state_id"]
    parent_round_dir = round_dir / "parents" / parent_id
    proposals_dir = parent_round_dir / "proposals"
    logs_dir = parent_round_dir / "logs"
    scratch_repo = args.campaign_root / "scratch" / f"proposer_{round_dir.name}_{parent_id}"
    proposer_state: Dict[str, Any] = {
        "label": "codex_proposer",
        "parent_state": parent_id,
        "commands": [],
        "log_dir": str(logs_dir),
        "state_path": str(proposals_dir / "proposer_state.json"),
    }
    proposer_state_path = proposals_dir / "proposer_state.json"
    write_json(proposer_state_path, proposer_state)

    try:
        if not dry_run and getattr(args, "resume_mode", False):
            try:
                proposal_dirs = validate_proposals(proposals_dir, args.max_proposals)
            except Exception:
                proposal_dirs = []
            else:
                log(f"Resuming: accepting existing proposals for parent {parent_id}")
                proposer_state["status"] = "complete"
                proposer_state["resumed_from_artifacts"] = True
                write_json(proposer_state_path, proposer_state)
                return True, proposals_dir, proposal_dirs

        prompt = render_proposer_prompt(
            args=args,
            parent_state=parent_state,
            round_dir=round_dir,
            analysis_dir=analysis_dir,
            proposals_dir=proposals_dir,
        )
        prompt_path = proposals_dir / "prompt.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

        last_error = None
        for attempt in range(1, max_codex_attempts(args) + 1):
            attempt_scratch = scratch_repo if attempt == 1 else scratch_repo.with_name(f"{scratch_repo.name}_attempt_{attempt:02d}")
            try:
                if not dry_run:
                    log(f"Materializing parent {parent_id} for proposer attempt {attempt} into {attempt_scratch}")
                    materialize_state(args.base_repo, args.states_root / parent_id, attempt_scratch)
                    log(f"Materialized parent {parent_id} for proposer attempt {attempt}")
                    sleep_before_codex_if_needed(args)
                command = build_codex_args(args, attempt_scratch if not dry_run else REPO_ROOT, proposals_dir / "last_message.md")
                codex_log_path = codex_phase_log_path(logs_dir, "codex_proposer", attempt)
                code = run_command(
                    command=command,
                    cwd=attempt_scratch if not dry_run else REPO_ROOT,
                    state=proposer_state,
                    label="codex_proposer",
                    dry_run=dry_run,
                    input_text=prompt,
                    log_path=codex_log_path,
                    state_path=proposer_state_path,
                    timeout_seconds=args.codex_exec_timeout_seconds,
                )
                if not dry_run and args.codex_exec_json_usage:
                    record_codex_usage(
                        args=args,
                        log_path=codex_log_path,
                        state=proposer_state,
                        state_path=proposer_state_path,
                    )
                if code != 0:
                    raise RuntimeError(f"codex proposer failed with returncode={code}")
                proposal_dirs = [] if dry_run else validate_proposals(proposals_dir, args.max_proposals)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                proposer_state.setdefault("attempt_failures", []).append({
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at": utcish_now(),
                })
                write_json(proposer_state_path, proposer_state)
                if attempt >= max_codex_attempts(args):
                    raise
                log(f"Retrying codex_proposer for parent {parent_id} after attempt {attempt} failed: {exc}")
            finally:
                if not dry_run and attempt_scratch.exists():
                    stop_router(args, attempt_scratch, proposer_state, dry_run=False)
                    log(f"Deleting proposer scratch repo {attempt_scratch}")
                    delete_scratch_repo(attempt_scratch)
        if last_error is not None:
            raise last_error
        proposer_state["status"] = "dry_run" if dry_run else "complete"
        ok = True
    except Exception as exc:
        proposer_state["status"] = "failed"
        proposer_state["error"] = f"{type(exc).__name__}: {exc}"
        proposal_dirs = []
        ok = False
    finally:
        write_json(proposer_state_path, proposer_state)
    return ok, proposals_dir, proposal_dirs


def run_bookkeeper(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    round_state_path: Path,
    dry_run: bool,
) -> bool:
    bookkeeping_dir = round_dir / "bookkeeping"
    bookkeeping_state_path = bookkeeping_dir / "bookkeeping_state.json"
    bookkeeping_state: Dict[str, Any] = {
        "label": "codex_bookkeeper",
        "state_path": str(bookkeeping_state_path),
        "commands": [],
    }
    if not dry_run and getattr(args, "resume_mode", False) and (bookkeeping_dir / "round_summary.json").exists():
        log(f"Resuming: accepting existing bookkeeper output for {round_dir.name}")
        bookkeeping_state["status"] = "complete"
        bookkeeping_state["resumed_from_artifacts"] = True
        write_json(bookkeeping_state_path, bookkeeping_state)
        return True

    prompt = render_bookkeeper_prompt(args=args, round_dir=round_dir, round_state_path=round_state_path)
    bookkeeping_dir.mkdir(parents=True, exist_ok=True)
    (bookkeeping_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    code = 1
    for attempt in range(1, max_codex_attempts(args) + 1):
        try:
            if not dry_run:
                sleep_before_codex_if_needed(args)
            command = build_codex_args(args, args.base_repo if args.base_repo.exists() else REPO_ROOT, bookkeeping_dir / "last_message.md")
            codex_log_path = codex_phase_log_path(bookkeeping_dir, "codex_bookkeeper", attempt)
            code = run_command(
                command=command,
                cwd=args.base_repo if args.base_repo.exists() else REPO_ROOT,
                state=bookkeeping_state,
                label="codex_bookkeeper",
                dry_run=dry_run,
                input_text=prompt,
                log_path=codex_log_path,
                state_path=bookkeeping_state_path,
                timeout_seconds=args.codex_exec_timeout_seconds,
            )
            if not dry_run and args.codex_exec_json_usage:
                record_codex_usage(
                    args=args,
                    log_path=codex_log_path,
                    state=bookkeeping_state,
                    state_path=bookkeeping_state_path,
                )
            if code != 0:
                raise RuntimeError(f"codex bookkeeper failed with returncode={code}")
            break
        except Exception as exc:
            bookkeeping_state.setdefault("attempt_failures", []).append({
                "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at": utcish_now(),
            })
            write_json(bookkeeping_state_path, bookkeeping_state)
            if attempt >= max_codex_attempts(args):
                code = 1
                break
            log(f"Retrying codex_bookkeeper for {round_dir.name} after attempt {attempt} failed: {exc}")
    bookkeeping_state["status"] = "dry_run" if dry_run else ("complete" if code == 0 else "failed")
    write_json(bookkeeping_state_path, bookkeeping_state)
    return code == 0


def load_state_payloads(states_root: Path, state_ids: Iterable[str]) -> List[Dict[str, Any]]:
    payloads = []
    for state_id in state_ids:
        path = states_root / state_id / "state.json"
        if path.exists():
            payloads.append(read_json(path))
    return payloads


def next_state_id(next_index: int) -> str:
    return f"s{next_index:04d}"


def proposal_combo_id(proposal_dirs: Iterable[Path]) -> str:
    return "+".join(path.name for path in proposal_dirs)


def proposal_combinations(proposal_dirs: List[Path], max_size: int) -> List[Tuple[Path, ...]]:
    capped = min(max(1, max_size), len(proposal_dirs))
    combos: List[Tuple[Path, ...]] = []
    for size in range(1, capped + 1):
        combos.extend(combinations(proposal_dirs, size))
    return combos


def child_failure_payload(args: argparse.Namespace, job: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    failure = {
        "state_id": job["state_id"],
        "parent_state_id": job["parent_state_id"],
        "proposal_id": job.get("proposal_id"),
        "source_proposal_ids": job.get("source_proposal_ids", []),
        "combo_size": job.get("combo_size", 1),
        "is_crossover": bool(job.get("is_crossover")),
        "proposal_dir": str(job["proposal_dirs"][0]) if job.get("proposal_dirs") else None,
        "proposal_dirs": [str(path) for path in job.get("proposal_dirs", [])],
        "status": "failed",
        "failure_reason": f"{type(exc).__name__}: {exc}",
    }
    write_json(args.states_root / job["state_id"] / "state.json", failure)
    return failure


def evaluate_child_jobs(args: argparse.Namespace, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not jobs:
        return []

    ordered_results: Dict[str, Dict[str, Any]] = {}
    parallel_states = max(1, int(args.parallel_states))
    for batch_start in range(0, len(jobs), parallel_states):
        batch = jobs[batch_start : batch_start + parallel_states]
        ports = [int(args.router_port) + slot for slot in range(len(batch))]
        require_router_ports_available(
            args,
            ports,
            f"parallel evaluation batch starting at child {batch[0]['state_id']}",
        )
        log(
            "Evaluating child batch "
            f"{[job['state_id'] for job in batch]} on router ports {ports}"
        )
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_job = {
                executor.submit(
                    evaluate_state,
                    args=args,
                    state_id=job["state_id"],
                    dry_run=False,
                    router_port=port,
                ): job
                for job, port in zip(batch, ports)
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    payload = future.result()
                except Exception as exc:
                    payload = child_failure_payload(args, job, exc)
                ordered_results[job["state_id"]] = payload
                log(f"Finished child {job['state_id']} with status {payload.get('status')}")

    return [ordered_results[job["state_id"]] for job in jobs]


def existing_child_for_combo(round_state: Dict[str, Any], parent_id: str, combo_id: str) -> Optional[Dict[str, Any]]:
    for child in round_state.get("children", []):
        if child.get("parent_state_id") == parent_id and child.get("proposal_id") == combo_id:
            return child
    return None


def upsert_round_child(round_state: Dict[str, Any], child_payload: Dict[str, Any]) -> None:
    children = [
        child
        for child in round_state.get("children", [])
        if child.get("state_id") != child_payload.get("state_id")
    ]
    children.append(child_payload)
    round_state["children"] = children


def run_round(
    *,
    args: argparse.Namespace,
    round_index: int,
    pool: List[Dict[str, Any]],
    campaign_state: Dict[str, Any],
    dry_run: bool,
) -> List[Dict[str, Any]]:
    round_id = f"r{round_index:03d}"
    round_dir = args.campaign_root / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    round_state_path = round_dir / "round_state.json"
    if getattr(args, "resume_mode", False) and round_state_path.exists():
        round_state = read_json(round_state_path)
        round_state["status"] = "running"
        round_state.setdefault("parents", [])
        round_state.setdefault("children", [])
        round_state.setdefault("skipped_crossovers", [])
        log(f"Resuming incomplete round {round_id}")
    else:
        round_state: Dict[str, Any] = {
            "round": round_id,
            "status": "dry_run" if dry_run else "running",
            "parents": [],
            "children": [],
            "skipped_crossovers": [],
            "crossover_max_size": args.crossover_max_size,
            "previous_pool": [state["state_id"] for state in pool],
        }

    parents = pool
    if args.max_states_per_round is not None:
        parents = parents[: args.max_states_per_round]

    for parent_state in parents:
        parent_id = parent_state["state_id"]
        parent_entry = next(
            (entry for entry in round_state["parents"] if entry.get("state_id") == parent_id),
            None,
        )
        if parent_entry is not None and parent_entry.get("status") == "complete":
            log(f"Resuming: skipping completed parent {parent_id} in {round_id}")
            continue
        log(f"Expanding parent {parent_id} in {round_id}")
        if parent_entry is None:
            parent_entry = {"state_id": parent_id, "status": "running"}
            round_state["parents"].append(parent_entry)
        else:
            parent_entry["status"] = "running"
        ok, analysis_dir = run_codex_analysis(
            args=args,
            parent_state=parent_state,
            round_dir=round_dir,
            dry_run=dry_run or args.skip_codex,
        )
        parent_entry["analysis_dir"] = str(analysis_dir)
        if not ok and not dry_run:
            parent_entry["status"] = "analysis_failed"
            log(f"Analysis failed for parent {parent_id}")
            raise RuntimeError(f"{round_id}: analysis failed for parent {parent_id}")

        ok, proposals_dir, proposal_dirs = run_codex_proposer(
            args=args,
            parent_state=parent_state,
            round_dir=round_dir,
            analysis_dir=analysis_dir,
            dry_run=dry_run or args.skip_codex,
        )
        parent_entry["proposals_dir"] = str(proposals_dir)
        if not ok and not dry_run:
            parent_entry["status"] = "proposer_failed"
            log(f"Proposer failed for parent {parent_id}")
            raise RuntimeError(f"{round_id}: proposer failed for parent {parent_id}")
        if dry_run or args.skip_codex:
            parent_entry["status"] = "dry_run"
            continue

        child_jobs: List[Dict[str, Any]] = []
        candidate_combos = proposal_combinations(proposal_dirs, args.crossover_max_size)
        parent_entry["candidate_count"] = len(candidate_combos)
        parent_entry["single_proposal_count"] = len(proposal_dirs)
        if args.crossover_max_size > 1:
            log(
                f"Generated {len(candidate_combos)} single/crossover candidates "
                f"from {len(proposal_dirs)} proposals for parent {parent_id}"
            )
        for combo_dirs in candidate_combos:
            combo_id = proposal_combo_id(combo_dirs)
            existing_child = existing_child_for_combo(round_state, parent_id, combo_id)
            if existing_child is not None:
                campaign_state.setdefault("states", {})[existing_child["state_id"]] = existing_child
                if existing_child.get("status") == "evaluated":
                    log(f"Resuming: skipping evaluated child {existing_child['state_id']} from {combo_id}")
                    continue
                log(f"Resuming: re-evaluating existing child {existing_child['state_id']} from {combo_id}")
                child_jobs.append({
                    "state_id": existing_child["state_id"],
                    "parent_state_id": parent_id,
                    "proposal_id": combo_id,
                    "source_proposal_ids": existing_child.get("source_proposal_ids", [path.name for path in combo_dirs]),
                    "combo_size": existing_child.get("combo_size", len(combo_dirs)),
                    "is_crossover": bool(existing_child.get("is_crossover")),
                    "proposal_dirs": list(combo_dirs),
                })
                continue

            state_id = next_state_id(campaign_state["next_state_index"])
            creation_index = campaign_state["next_state_index"]
            source_proposal_ids = [path.name for path in combo_dirs]
            combo_size = len(combo_dirs)
            job = {
                "state_id": state_id,
                "parent_state_id": parent_id,
                "proposal_id": combo_id,
                "source_proposal_ids": source_proposal_ids,
                "combo_size": combo_size,
                "is_crossover": combo_size > 1,
                "proposal_dirs": list(combo_dirs),
            }
            try:
                log(f"Creating child {state_id} from {combo_id}")
                create_child_state_from_proposals(
                    base_repo=args.base_repo,
                    states_root=args.states_root,
                    parent_state_dir=args.states_root / parent_id,
                    proposal_dirs=list(combo_dirs),
                    state_id=state_id,
                    created_round=round_index,
                    creation_index=creation_index,
                )
                campaign_state["next_state_index"] += 1
                child_jobs.append(job)
            except ProposalConflict as exc:
                if combo_size > 1:
                    skipped = {
                        "round": round_id,
                        "parent_state_id": parent_id,
                        "proposal_id": combo_id,
                        "source_proposal_ids": source_proposal_ids,
                        "combo_size": combo_size,
                        "proposal_dirs": [str(path) for path in combo_dirs],
                        "status": "skipped_conflict",
                        "reason": str(exc),
                    }
                    round_state["skipped_crossovers"].append(skipped)
                    parent_entry.setdefault("skipped_crossovers", []).append(skipped)
                    log(f"Skipped crossover {combo_id}: {exc}")
                    continue
                campaign_state["next_state_index"] += 1
                failure = child_failure_payload(args, job, exc)
                round_state["children"].append(failure)
                campaign_state.setdefault("states", {})[state_id] = failure
                log(f"Child {state_id} failed before/during evaluation: {failure['failure_reason']}")
            except Exception as exc:
                campaign_state["next_state_index"] += 1
                failure = child_failure_payload(args, job, exc)
                round_state["children"].append(failure)
                campaign_state.setdefault("states", {})[state_id] = failure
                log(f"Child {state_id} failed before/during evaluation: {failure['failure_reason']}")
        parent_entry["skipped_crossover_count"] = len(parent_entry.get("skipped_crossovers", []))
        parent_entry["evaluation_count"] = len(child_jobs)
        for child_payload in evaluate_child_jobs(args, child_jobs):
            upsert_round_child(round_state, child_payload)
            campaign_state.setdefault("states", {})[child_payload["state_id"]] = child_payload
        parent_entry["status"] = "complete"

    joint_payloads = load_state_payloads(
        args.states_root,
        [state["state_id"] for state in pool]
        + [child.get("state_id") for child in round_state["children"] if child.get("state_id")],
    )
    next_pool = rank_states(joint_payloads, args.pool_size) if not dry_run else pool
    round_state["next_pool"] = [state["state_id"] for state in next_pool]
    round_state["status"] = "dry_run" if dry_run else "complete"
    round_state_path = round_dir / "round_state.json"
    write_json(round_state_path, round_state)

    if not args.skip_codex:
        log(f"Running bookkeeper for {round_id}")
        if not run_bookkeeper(args=args, round_dir=round_dir, round_state_path=round_state_path, dry_run=dry_run):
            raise RuntimeError(f"{round_id}: bookkeeper failed")
    return next_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run harness tree search.")
    parser.add_argument("--datasets", nargs="+", default=["ERQA", "MMSIBench_wo_circular"])
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--pool-size", type=int, default=3)
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--num-proc", type=int, default=50)
    parser.add_argument("--parallel-states", type=int, default=4)
    parser.add_argument("--crossover-max-size", type=int, default=1)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--resume-campaign-root", type=Path, default=None)
    parser.add_argument("--round-prefix", default="harness_search")
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--agent-config", default=DEFAULT_AGENT_CONFIG)
    parser.add_argument("--judge-config", default=DEFAULT_JUDGE_CONFIG)
    parser.add_argument("--router-env-file", default="mini_vlm_toolkit/tool_infra/.env.latest")
    parser.add_argument("--router-port", type=int, default=8103)
    parser.add_argument("--router-health-timeout", type=float, default=90.0)
    parser.add_argument("--codex-bin", default=DEFAULT_CODEX_BIN)
    parser.add_argument("--codex-model", default="gpt-5.5")
    parser.add_argument("--codex-reasoning-effort", default="medium")
    parser.add_argument(
        "--codex-usage-threshold-percent",
        type=float,
        default=90.0,
        help="Before the next Codex exec call, sleep if the previous primary 5-hour usage window reached this percent; set 0 to disable.",
    )
    parser.add_argument(
        "--codex-usage-sleep-cushion-seconds",
        type=int,
        default=60,
        help="Extra seconds to sleep after the primary Codex usage window reset time.",
    )
    parser.add_argument(
        "--codex-status-probe",
        choices=["auto", "off"],
        default="auto",
        help="Use a short-lived PTY Codex /status probe before Codex exec calls.",
    )
    parser.add_argument(
        "--codex-status-timeout",
        type=float,
        default=25.0,
        help="Timeout in seconds for the PTY Codex /status probe.",
    )
    parser.add_argument(
        "--codex-status-startup-delay",
        type=float,
        default=8.0,
        help="Seconds to wait before sending /status in the PTY Codex status probe.",
    )
    parser.add_argument(
        "--codex-status-failure-mode",
        choices=["warn", "fail"],
        default="warn",
        help="Whether an unparsable/unavailable Codex /status probe warns and continues or fails the campaign.",
    )
    parser.add_argument(
        "--codex-exec-json-usage",
        action="store_true",
        help="Run codex exec with --json and parse token_count events as a fallback/debug usage source. This streams JSONL to logs/stdout.",
    )
    parser.add_argument(
        "--codex-exec-timeout-seconds",
        type=float,
        default=3600.0,
        help="Hard timeout for each codex exec phase; set 0 to disable.",
    )
    parser.add_argument(
        "--codex-exec-retries",
        type=int,
        default=1,
        help="Retries after the first failed/timed-out codex exec attempt.",
    )
    parser.add_argument("--playground-max-tasks", type=int, default=20)
    parser.add_argument("--playground-trials-per-task", type=int, default=5)
    parser.add_argument("--max-states-per-round", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-codex", action="store_true")
    return parser.parse_args()


def state_index_from_id(state_id: str) -> Optional[int]:
    if len(state_id) >= 2 and state_id.startswith("s") and state_id[1:].isdigit():
        return int(state_id[1:])
    return None


def next_state_index_from_disk(states_root: Path) -> int:
    indexes = []
    if states_root.exists():
        for path in states_root.iterdir():
            if path.is_dir():
                index = state_index_from_id(path.name)
                if index is not None:
                    indexes.append(index)
    return max(indexes, default=0) + 1


def setup_resume_campaign(args: argparse.Namespace) -> Tuple[Dict[str, Any], Path]:
    campaign_root = args.resume_campaign_root.resolve()
    campaign_state_path = campaign_root / "campaign_state.json"
    if not campaign_state_path.exists():
        raise SystemExit(f"--resume-campaign-root missing campaign_state.json: {campaign_state_path}")
    campaign_state = read_json(campaign_state_path)
    args.resume_mode = True
    args.campaign_root = campaign_root
    args.base_repo = campaign_root / "base_repo"
    args.states_root = campaign_root / "states"
    args.datasets = list(campaign_state.get("datasets") or args.datasets)
    args.num_examples = int(campaign_state.get("num_examples", args.num_examples))
    args.num_proc = int(campaign_state.get("num_proc", args.num_proc))
    args.parallel_states = int(campaign_state.get("parallel_states", args.parallel_states))
    args.crossover_max_size = int(campaign_state.get("crossover_max_size", args.crossover_max_size))
    args.num_rounds = int(campaign_state.get("num_rounds", args.num_rounds))
    args.pool_size = int(campaign_state.get("pool_size", args.pool_size))
    args.max_proposals = int(campaign_state.get("max_proposals", args.max_proposals))
    args.router_port = int(campaign_state.get("router_port", args.router_port))
    args.tool_base = str(campaign_state.get("tool_base") or f"http://localhost:{args.router_port}")
    args.router_env_file = str(campaign_state.get("router_env_file") or (campaign_root / "configs" / "router.env"))
    args.agent_config = str(campaign_state.get("agent_config") or (campaign_root / "configs" / "agent.yaml"))
    args.source_router_env_file = str(campaign_state.get("source_router_env_file") or args.router_env_file)
    args.source_agent_config = str(campaign_state.get("source_agent_config") or args.agent_config)
    args.router_state_dir = campaign_root / "logs" / "router"
    args.benchmark_env = build_benchmark_env(REPO_ROOT, args.tool_base)
    campaign_state["status"] = "running"
    campaign_state["resumed_at"] = utcish_now()
    campaign_state["next_state_index"] = max(
        int(campaign_state.get("next_state_index", 1)),
        next_state_index_from_disk(args.states_root),
    )
    write_json(campaign_state_path, campaign_state)
    return campaign_state, campaign_state_path


def setup_new_campaign(args: argparse.Namespace) -> Tuple[Dict[str, Any], Path]:
    campaign_id = args.campaign_id or build_campaign_id(args.round_prefix)
    campaign_id, campaign_root = create_campaign_root(
        args.work_root.resolve(),
        campaign_id,
        explicit=bool(args.campaign_id),
    )
    args.resume_mode = False
    args.campaign_root = campaign_root
    args.base_repo = campaign_root / "base_repo"
    args.states_root = campaign_root / "states"
    args.datasets = list(args.datasets)
    materialize_campaign_runtime_config(args)
    if args.router_port + args.parallel_states - 1 > 65535:
        raise SystemExit("--router-port + --parallel-states - 1 must be <= 65535")
    args.router_state_dir = campaign_root / "logs" / "router"
    args.benchmark_env = build_benchmark_env(REPO_ROOT, args.tool_base)

    campaign_state_path = campaign_root / "campaign_state.json"
    campaign_state: Dict[str, Any] = {
        "campaign_id": campaign_id,
        "status": "dry_run" if args.dry_run else "running",
        "created_at": utcish_now(),
        "campaign_root": str(campaign_root),
        "datasets": args.datasets,
        "num_examples": args.num_examples,
        "num_proc": args.num_proc,
        "parallel_states": args.parallel_states,
        "crossover_max_size": args.crossover_max_size,
        "num_rounds": args.num_rounds,
        "pool_size": args.pool_size,
        "max_proposals": args.max_proposals,
        "router_port": args.router_port,
        "tool_base": args.tool_base,
        "router_env_file": args.router_env_file,
        "agent_config": args.agent_config,
        "source_router_env_file": args.source_router_env_file,
        "source_agent_config": args.source_agent_config,
        "codex_usage_guard": {
            "threshold_percent": args.codex_usage_threshold_percent,
            "sleep_cushion_seconds": args.codex_usage_sleep_cushion_seconds,
            "window": "primary",
            "status_probe": args.codex_status_probe,
            "status_timeout": args.codex_status_timeout,
            "status_startup_delay": args.codex_status_startup_delay,
            "status_failure_mode": args.codex_status_failure_mode,
            "exec_json_usage": args.codex_exec_json_usage,
            "exec_timeout_seconds": args.codex_exec_timeout_seconds,
            "exec_retries": args.codex_exec_retries,
        },
        "playground": {
            "max_tasks": args.playground_max_tasks,
            "trials_per_task": args.playground_trials_per_task,
        },
        "next_state_index": 1,
        "states": {},
        "rounds": [],
    }
    write_json(campaign_state_path, campaign_state)
    init_runbook(campaign_root / "RUNBOOK.md", args=args, campaign_id=campaign_id, datasets=args.datasets)
    return campaign_state, campaign_state_path


def pool_from_state_ids(args: argparse.Namespace, state_ids: Iterable[str]) -> List[Dict[str, Any]]:
    return load_state_payloads(args.states_root, state_ids)


def completed_round_pool(args: argparse.Namespace, campaign_state: Dict[str, Any], round_index: int) -> Optional[List[Dict[str, Any]]]:
    round_id = f"r{round_index:03d}"
    round_state_path = args.campaign_root / "rounds" / round_id / "round_state.json"
    if not round_state_path.exists():
        return None
    try:
        round_state = read_json(round_state_path)
    except Exception:
        return None
    if round_state.get("status") != "complete":
        return None
    state_ids = round_state.get("next_pool") or []
    if not state_ids:
        for entry in campaign_state.get("rounds", []):
            if entry.get("round") == round_id:
                state_ids = entry.get("pool") or []
                break
    payloads = pool_from_state_ids(args, state_ids)
    return payloads or None


def record_round_completion(campaign_state: Dict[str, Any], round_id: str, pool: List[Dict[str, Any]]) -> None:
    rounds = [entry for entry in campaign_state.get("rounds", []) if entry.get("round") != round_id]
    rounds.append({"round": round_id, "pool": [state["state_id"] for state in pool]})
    campaign_state["rounds"] = sorted(rounds, key=lambda item: item.get("round", ""))


def main() -> int:
    args = parse_args()
    if args.num_rounds < 0:
        raise SystemExit("--num-rounds must be >= 0")
    if args.pool_size < 1:
        raise SystemExit("--pool-size must be >= 1")
    if args.max_proposals < 1:
        raise SystemExit("--max-proposals must be >= 1")
    if args.num_proc < 1:
        raise SystemExit("--num-proc must be >= 1")
    if args.parallel_states < 1:
        raise SystemExit("--parallel-states must be >= 1")
    if not (1 <= args.crossover_max_size <= 4):
        raise SystemExit("--crossover-max-size must be in [1, 4]")
    if args.router_port is not None and not (1 <= args.router_port <= 65535):
        raise SystemExit("--router-port must be in [1, 65535]")
    if not (0 <= args.codex_usage_threshold_percent <= 100):
        raise SystemExit("--codex-usage-threshold-percent must be in [0, 100]")
    if args.codex_usage_sleep_cushion_seconds < 0:
        raise SystemExit("--codex-usage-sleep-cushion-seconds must be >= 0")
    if args.codex_status_timeout <= 0:
        raise SystemExit("--codex-status-timeout must be > 0")
    if args.codex_status_startup_delay < 0:
        raise SystemExit("--codex-status-startup-delay must be >= 0")
    if args.codex_status_startup_delay >= args.codex_status_timeout:
        raise SystemExit("--codex-status-startup-delay must be less than --codex-status-timeout")
    if args.codex_exec_timeout_seconds < 0:
        raise SystemExit("--codex-exec-timeout-seconds must be >= 0")
    if args.codex_exec_retries < 0:
        raise SystemExit("--codex-exec-retries must be >= 0")

    if args.resume_campaign_root is not None:
        campaign_state, campaign_state_path = setup_resume_campaign(args)
        if args.router_port + args.parallel_states - 1 > 65535:
            raise SystemExit("--router-port + --parallel-states - 1 must be <= 65535")
        base_state = read_json(args.states_root / "s0000" / "state.json")
        if base_state.get("status") == "created" and (args.states_root / "s0000" / "eval" / "performance_summary.json").exists():
            base_state = read_json(args.states_root / "s0000" / "state.json")
        log(f"Resuming campaign at {args.campaign_root}")
    else:
        campaign_state, campaign_state_path = setup_new_campaign(args)

        if not args.dry_run:
            log(f"Creating base repo snapshot at {args.base_repo}")
            create_base_snapshot(REPO_ROOT, args.base_repo)
            log(f"Created base repo snapshot at {args.base_repo}")
        else:
            args.base_repo.mkdir(parents=True, exist_ok=True)

        create_initial_state(args.states_root)
        log("Evaluating baseline state s0000")
        if not args.dry_run:
            require_router_ports_available(args, [args.router_port], "baseline evaluation")
        base_state = evaluate_state(args=args, state_id="s0000", dry_run=args.dry_run)
        campaign_state["states"]["s0000"] = base_state
        write_json(campaign_state_path, campaign_state)

    if base_state.get("status") not in {"evaluated", "dry_run"}:
        campaign_state["status"] = "failed"
        campaign_state["error"] = "baseline evaluation failed"
        write_json(campaign_state_path, campaign_state)
        print(f"Campaign state: {campaign_state_path}")
        return 1

    if getattr(args, "resume_mode", False) and campaign_state.get("pool"):
        pool = pool_from_state_ids(args, campaign_state.get("pool", [])) or [base_state]
    else:
        pool = [base_state]
    if args.skip_codex and not args.dry_run:
        campaign_state["status"] = "complete"
        campaign_state["finished_at"] = utcish_now()
        campaign_state["pool"] = [state["state_id"] for state in pool]
        write_json(campaign_state_path, campaign_state)
        print(f"Campaign state: {campaign_state_path}")
        return 0

    for round_index in range(1, args.num_rounds + 1):
        if getattr(args, "resume_mode", False):
            completed_pool = completed_round_pool(args, campaign_state, round_index)
            if completed_pool is not None:
                pool = completed_pool
                campaign_state["pool"] = [state["state_id"] for state in pool]
                log(f"Resuming: skipping completed round r{round_index:03d}; pool {[state['state_id'] for state in pool]}")
                continue
        log(f"Starting search round r{round_index:03d} with pool {[state['state_id'] for state in pool]}")
        try:
            pool = run_round(
                args=args,
                round_index=round_index,
                pool=pool,
                campaign_state=campaign_state,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            campaign_state["status"] = "failed"
            campaign_state["error"] = f"{type(exc).__name__}: {exc}"
            campaign_state["failed_round"] = f"r{round_index:03d}"
            campaign_state["updated_at"] = utcish_now()
            write_json(campaign_state_path, campaign_state)
            log(f"FAILED search round r{round_index:03d}: {campaign_state['error']}")
            print(f"Campaign state: {campaign_state_path}")
            return 1
        record_round_completion(campaign_state, f"r{round_index:03d}", pool)
        campaign_state["pool"] = [state["state_id"] for state in pool]
        campaign_state["updated_at"] = utcish_now()
        write_json(campaign_state_path, campaign_state)
        log(f"Finished search round r{round_index:03d}; pool {[state['state_id'] for state in pool]}")

    campaign_state["status"] = "dry_run" if args.dry_run else "complete"
    campaign_state["finished_at"] = utcish_now()
    write_json(campaign_state_path, campaign_state)
    print(f"Campaign state: {campaign_state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
