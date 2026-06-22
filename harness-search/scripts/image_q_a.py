#!/usr/bin/env python3
"""Replay one saved image-QA trajectory with the current agent harness."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_ROOT = REPO_ROOT / "harness-search"
DEFAULT_AGENT_CONFIG = PLAYGROUND_ROOT / "configs" / "run_benchmark_agent.yaml"
DEFAULT_ROUTER_ENV_FILE = PLAYGROUND_ROOT / "scripts" / ".env.latest"
DEFAULT_ROUTER_MANAGER = PLAYGROUND_ROOT / "scripts" / "manage_router.py"
DEFAULT_ROUTER_LOG = PLAYGROUND_ROOT / "runs" / ".router" / "router.log"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def safe_name(value: Any) -> str:
    text = str(value or "task")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def resolve_local_reference(value: str, *, trajectory_path: Path) -> str:
    """Resolve saved trajectory-relative image paths for agent.generate."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.startswith("file://"):
        return text[len("file://"):]
    if text.startswith(("http://", "https://", "data:")):
        return text
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    base_dir = trajectory_path.parent
    candidates = [
        base_dir / candidate,
        base_dir.parent / candidate,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def convert_content_part(part: Any, *, trajectory_path: Path) -> Optional[Dict[str, Any]]:
    if isinstance(part, str):
        return {"type": "text", "value": part}
    if not isinstance(part, dict):
        return {"type": "text", "value": str(part)}

    part_type = part.get("type")
    if part_type in {"text", "input_text"}:
        return {"type": "text", "value": str(part.get("text", ""))}

    if part_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            value = image_url.get("url")
        else:
            value = image_url
        if not value:
            return None
        return {
            "type": "image",
            "value": resolve_local_reference(str(value), trajectory_path=trajectory_path),
        }

    if part_type in {"image", "input_image"}:
        value = part.get("value") or part.get("url") or part.get("image")
        if not value:
            return None
        return {
            "type": "image",
            "value": resolve_local_reference(str(value), trajectory_path=trajectory_path),
        }

    if "text" in part:
        return {"type": "text", "value": str(part.get("text", ""))}
    return {"type": "text", "value": json.dumps(part, ensure_ascii=False)}


def message_to_agent_input(message: Dict[str, Any], *, trajectory_path: Path) -> List[Dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "value": content}]
    if isinstance(content, list):
        converted = [
            convert_content_part(part, trajectory_path=trajectory_path)
            for part in content
        ]
        return [part for part in converted if part is not None]
    if isinstance(content, dict):
        converted = convert_content_part(content, trajectory_path=trajectory_path)
        return [converted] if converted is not None else []
    return [{"type": "text", "value": str(content)}]


def extract_replay_message(payload: Dict[str, Any], *, trajectory_path: Path) -> List[Dict[str, Any]]:
    """Extract the original user task and intentionally drop old harness turns."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("trajectory payload must contain a messages list")

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            replay_message = message_to_agent_input(message, trajectory_path=trajectory_path)
            if replay_message:
                return replay_message
            raise ValueError("first user message did not contain replayable content")
    raise ValueError("trajectory messages did not contain a user message")


def build_output_path(out_dir: Path, *, trial_name: Optional[str], payload: Dict[str, Any]) -> Path:
    if trial_name:
        name = safe_name(trial_name)
    else:
        dataset = safe_name(payload.get("dataset") or "dataset")
        task_id = safe_name(payload.get("task_id") or "task")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{dataset}_{task_id}_{stamp}"
    if not name.endswith(".json"):
        name += ".json"
    return out_dir / name


def strip_raw_sft(value: Any) -> Any:
    """Remove raw SFT payloads from replay artifacts; they are too large for review."""
    if isinstance(value, dict):
        return {
            key: strip_raw_sft(child)
            for key, child in value.items()
            if key != "raw_sft"
        }
    if isinstance(value, list):
        return [strip_raw_sft(child) for child in value]
    return value


def command_to_display(command: List[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def tail_text(path: Path, *, max_chars: int) -> Optional[str]:
    if max_chars <= 0 or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<failed to read {path}: {exc}>"
    return text[-max_chars:]


def run_router_restart(
    *,
    env_file: Path,
    state_dir: Path,
    health_timeout: float,
    stop_timeout: float,
    use_tmux: bool,
    tmux_session: str,
    log_tail_chars: int,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(DEFAULT_ROUTER_MANAGER),
        "restart",
        "--env-file",
        str(env_file),
        "--state-dir",
        str(state_dir),
        "--health-timeout",
        str(health_timeout),
        "--stop-timeout",
        str(stop_timeout),
    ]
    if use_tmux:
        command.extend(["--tmux", "--tmux-session", tmux_session])

    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    status_command = [
        sys.executable,
        str(DEFAULT_ROUTER_MANAGER),
        "status",
        "--env-file",
        str(env_file),
        "--state-dir",
        str(state_dir),
    ]
    status_result = subprocess.run(
        status_command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command_to_display(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status_command": command_to_display(status_command),
        "status_returncode": status_result.returncode,
        "status_stdout": status_result.stdout,
        "status_stderr": status_result.stderr,
        "router_log_tail": tail_text(state_dir / "router.log", max_chars=log_tail_chars),
    }


def write_router_failure_artifact(
    *,
    trajectory_path: Path,
    agent_config: Path,
    out_dir: Path,
    trial_name: Optional[str],
    router_restart: Dict[str, Any],
    error: Optional[str] = None,
) -> Path:
    payload: Dict[str, Any] = {}
    try:
        loaded = read_json(trajectory_path)
        if isinstance(loaded, dict):
            payload = loaded
    except Exception as exc:
        error = f"{error}; failed to read trajectory: {exc}" if error else f"failed to read trajectory: {exc}"

    artifact = {
        "status": "router_restart_failed",
        "source_trajectory": str(trajectory_path),
        "agent_config": str(agent_config),
        "dataset": payload.get("dataset"),
        "task_id": payload.get("task_id"),
        "old_prediction": payload.get("prediction"),
        "ground_truth": payload.get("ground_truth"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "error": error,
        "router_restart": router_restart,
    }
    output_path = build_output_path(out_dir, trial_name=trial_name, payload=payload)
    write_json(output_path, artifact)
    return output_path


def run_replay(
    *,
    trajectory_path: Path,
    agent_config: Path,
    out_dir: Path,
    trial_name: Optional[str],
) -> Path:
    from mini_vlm_toolkit.agent import build_agent

    payload = read_json(trajectory_path)
    if not isinstance(payload, dict):
        raise ValueError(f"trajectory payload must be a JSON object: {trajectory_path}")

    replay_message = extract_replay_message(payload, trajectory_path=trajectory_path)
    agent = build_agent(str(agent_config))

    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.monotonic()
    result = agent.generate(
        message=copy.deepcopy(replay_message),
        dataset=payload.get("dataset"),
        task_id=payload.get("task_id"),
    )
    elapsed = time.monotonic() - start

    if isinstance(result, dict):
        clean_result = strip_raw_sft(result)
        new_prediction = clean_result.get("prediction")
        new_trajectory = (clean_result.get("extra_records") or {}).get("trajectory")
    else:
        clean_result = result
        new_prediction = result
        new_trajectory = None

    artifact = {
        "source_trajectory": str(trajectory_path),
        "agent_config": str(agent_config),
        "dataset": payload.get("dataset"),
        "task_id": payload.get("task_id"),
        "old_prediction": payload.get("prediction"),
        "ground_truth": payload.get("ground_truth"),
        "started_at": started_at,
        "elapsed_seconds": round(elapsed, 3),
        "new_prediction": new_prediction,
        "replay_message": replay_message,
        "result": clean_result,
    }
    if new_trajectory is not None:
        artifact["trajectory"] = new_trajectory

    output_path = build_output_path(out_dir, trial_name=trial_name, payload=payload)
    write_json(output_path, artifact)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one saved full_trajectory task.")
    parser.add_argument("--trajectory", required=True, help="Path to full_trajectories/<task_id>.json")
    parser.add_argument("--agent-config", default=str(DEFAULT_AGENT_CONFIG), help="Current agent config to replay with")
    parser.add_argument("--out-dir", required=True, help="Directory for replay artifact JSON")
    parser.add_argument("--trial-name", default=None, help="Optional output artifact basename")
    parser.add_argument(
        "--no-restart-router",
        action="store_true",
        help="Debug opt-out: do not restart and health-check the router before replay.",
    )
    parser.add_argument("--router-env-file", default=str(DEFAULT_ROUTER_ENV_FILE))
    parser.add_argument("--router-state-dir", default=None)
    parser.add_argument("--router-health-timeout", type=float, default=30.0)
    parser.add_argument("--router-stop-timeout", type=float, default=10.0)
    parser.add_argument("--router-tmux", action="store_true")
    parser.add_argument("--router-tmux-session", default="harness_router")
    parser.add_argument("--router-log-tail-chars", type=int, default=12000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory_path = Path(args.trajectory).resolve()
    agent_config = Path(args.agent_config).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not args.no_restart_router:
        router_state_dir = (
            Path(args.router_state_dir).resolve()
            if args.router_state_dir
            else (PLAYGROUND_ROOT / "runs" / ".router").resolve()
        )
        router_restart = run_router_restart(
            env_file=Path(args.router_env_file).resolve(),
            state_dir=router_state_dir,
            health_timeout=args.router_health_timeout,
            stop_timeout=args.router_stop_timeout,
            use_tmux=bool(args.router_tmux),
            tmux_session=args.router_tmux_session,
            log_tail_chars=args.router_log_tail_chars,
        )
        if router_restart["returncode"] != 0:
            output_path = write_router_failure_artifact(
                trajectory_path=trajectory_path,
                agent_config=agent_config,
                out_dir=out_dir,
                trial_name=args.trial_name,
                router_restart=router_restart,
            )
            print(str(output_path))
            return router_restart["returncode"] or 1

        else:
            print(f"Router restart succeeded: {router_restart['stdout']}")

    output_path = run_replay(
        trajectory_path=trajectory_path,
        agent_config=agent_config,
        out_dir=out_dir,
        trial_name=args.trial_name,
    )
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
