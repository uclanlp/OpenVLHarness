#!/usr/bin/env python3
"""Manage the harness-search tool router process.

The router must be restarted after edits to tool descriptions or schemas in
`tool.py`, because the FastAPI server loads tool metadata at startup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_ROOT = REPO_ROOT / "harness-search"
DEFAULT_STATE_DIR = PLAYGROUND_ROOT / "runs" / ".router"
DEFAULT_LAUNCH_SCRIPT = PLAYGROUND_ROOT / "scripts" / "launch_router.sh"
DEFAULT_ENV_FILE = PLAYGROUND_ROOT / "scripts" / ".env.latest"


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def router_host_port_from_env(env_file: Path) -> tuple[str, str]:
    values = parse_env_file(env_file)
    host = values.get("HOST") or os.environ.get("HOST") or "127.0.0.1"
    port = values.get("PORT") or os.environ.get("PORT") or "8101"
    return host, port


def health_url_from_env(env_file: Path) -> str:
    host, port = router_host_port_from_env(env_file)
    return f"http://{host}:{port}/health"


def router_base_url_from_env(env_file: Path) -> str:
    host, port = router_host_port_from_env(env_file)
    return f"http://{host}:{port}"


def fetch_json_endpoint(url: str, timeout: float = 2.0) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                return None
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError):
        return None


def router_ready(base_url: str, timeout: float = 2.0) -> bool:
    tools = fetch_json_endpoint(f"{base_url}/tools", timeout=timeout)
    schemas = fetch_json_endpoint(f"{base_url}/llm-schema", timeout=timeout)
    return isinstance(tools, dict) and isinstance(schemas, list)


def wait_for_router_ready(base_url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if router_ready(base_url, timeout=2.0):
            return True
        time.sleep(1)
    return False


def process_cmdline(pid: int) -> str:
    proc_cmdline = Path("/proc") / str(pid) / "cmdline"
    if proc_cmdline.exists():
        try:
            return proc_cmdline.read_text(encoding="utf-8", errors="replace").replace("\x00", " ").strip()
        except OSError:
            pass
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip()


def managed_router_process(pid: int, state: Dict[str, Any]) -> bool:
    cmdline = process_cmdline(pid)
    if not cmdline:
        return False
    launch_script = str(state.get("launch_script") or DEFAULT_LAUNCH_SCRIPT)
    indicators = [
        "server.app:app",
        launch_script,
        Path(launch_script).name,
    ]
    return any(indicator and indicator in cmdline for indicator in indicators)


def _parse_lsof_pids(output: str) -> List[int]:
    pids: List[int] = []
    for line in output.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _parse_ss_pids(output: str, port: str) -> List[int]:
    pids: List[int] = []
    port_pattern = re.compile(rf"(?<!\d):{re.escape(str(port))}\b")
    pid_pattern = re.compile(r"pid=(\d+)")
    for line in output.splitlines():
        if not port_pattern.search(line):
            continue
        for match in pid_pattern.finditer(line):
            pids.append(int(match.group(1)))
    return pids


def _parse_fuser_pids(output: str) -> List[int]:
    pids: List[int] = []
    for match in re.finditer(r"\b(\d+)\b", output):
        pids.append(int(match.group(1)))
    return pids


def unique_pids(pids: List[int]) -> List[int]:
    seen = set()
    unique: List[int] = []
    for pid in pids:
        if pid > 0 and pid not in seen:
            seen.add(pid)
            unique.append(pid)
    return unique


def listening_pids_for_port(port: str) -> List[int]:
    commands = [
        (["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"], _parse_lsof_pids),
        (["ss", "-ltnp"], lambda output: _parse_ss_pids(output, port)),
        (["fuser", "-n", "tcp", str(port)], _parse_fuser_pids),
        (["pgrep", "-f", rf"(uvicorn|server\.app:app).*--port[ =]{re.escape(str(port))}"], _parse_lsof_pids),
    ]
    for command, parser in commands:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        except OSError:
            continue
        pids = unique_pids(parser(result.stdout))
        if pids:
            return pids
    return []


def wait_for_health(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    return False


def terminate_pids(pids: List[int], timeout: float, *, use_process_group: bool = False) -> Dict[str, Any]:
    live_pids = [pid for pid in unique_pids(pids) if pid_alive(pid)]
    if not live_pids:
        return {"status": "not_running", "pids": unique_pids(pids)}

    permission_denied: List[int] = []
    for pid in live_pids:
        try:
            if use_process_group:
                os.killpg(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            permission_denied.append(pid)

    wait_pids = [pid for pid in live_pids if pid not in set(permission_denied)]
    if not wait_pids:
        return {
            "status": "permission_denied",
            "pids": live_pids,
            "still_running": live_pids,
            "permission_denied": unique_pids(permission_denied),
        }

    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(not pid_alive(pid) for pid in wait_pids):
            result: Dict[str, Any] = {"status": "stopped", "pids": live_pids}
            if permission_denied:
                result["permission_denied"] = unique_pids(permission_denied)
            return result
        time.sleep(0.5)

    for pid in wait_pids:
        try:
            if use_process_group:
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            permission_denied.append(pid)

    still_running = [pid for pid in live_pids if pid_alive(pid)]
    if still_running:
        status = "permission_denied" if permission_denied else "still_running"
        return {
            "status": status,
            "pids": live_pids,
            "still_running": still_running,
            "permission_denied": unique_pids(permission_denied),
        }
    result = {"status": "killed", "pids": live_pids}
    if permission_denied:
        result["permission_denied"] = unique_pids(permission_denied)
    return result


def stop_from_state(state_path: Path, env_file: Path, timeout: float) -> Dict[str, Any]:
    state = read_json(state_path)
    _, port = router_host_port_from_env(env_file)
    port_pids = listening_pids_for_port(port)
    stop_results: Dict[str, Any] = {
        "port": port,
        "port_pids": port_pids,
    }

    if port_pids:
        stop_results["port_stop"] = terminate_pids(port_pids, timeout)

    tmux_session = str(state.get("tmux_session") or "").strip()
    if tmux_session:
        result = subprocess.run(
            ["tmux", "kill-session", "-t", tmux_session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if state_path.exists():
            state_path.unlink()
        stop_results.update({
            "status": "stopped_tmux" if result.returncode == 0 else "tmux_not_running",
            "tmux_session": tmux_session,
        })
        return stop_results

    pid = int(state.get("pid") or 0)
    session = bool(state.get("start_new_session"))
    if not pid_alive(pid):
        if state_path.exists():
            state_path.unlink()
        stop_results.update({"status": "stopped" if port_pids else "not_running", "pid": pid})
        return stop_results

    if managed_router_process(pid, state):
        stop_results["state_pid_stop"] = terminate_pids([pid], timeout, use_process_group=session)
    else:
        stop_results["stale_state_pid"] = pid
    if state_path.exists():
        state_path.unlink()
    stop_results["status"] = "stopped" if port_pids or stop_results.get("state_pid_stop") else "stale_state"
    return stop_results


def start_router(
    *,
    launch_script: Path,
    env_file: Path,
    state_path: Path,
    log_path: Path,
    health_timeout: float,
    use_tmux: bool,
    tmux_session: str,
) -> Dict[str, Any]:
    health_url = health_url_from_env(env_file)
    base_url = router_base_url_from_env(env_file)
    _, port = router_host_port_from_env(env_file)
    port_pids = listening_pids_for_port(port)
    if wait_for_health(health_url, timeout=1.5) and router_ready(base_url, timeout=2.0):
        return {
            "status": "already_running",
            "pids": port_pids,
            "port": port,
            "health_ok": True,
            "router_ready": True,
            "health_url": health_url,
            "state_path": str(state_path),
        }
    if port_pids:
        stop_from_state(state_path, env_file, timeout=health_timeout)
        port_pids = listening_pids_for_port(port)
        if port_pids:
            return {
                "status": "port_busy",
                "pids": port_pids,
                "port": port,
                "health_ok": wait_for_health(health_url, timeout=1.5),
                "router_ready": router_ready(base_url, timeout=2.0),
                "health_url": health_url,
                "state_path": str(state_path),
            }

    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if use_tmux:
        command = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            tmux_session,
            "bash",
            str(launch_script),
            str(env_file),
        ]
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)
        state = {
            "status": "running_tmux",
            "pid": 0,
            "tmux_session": tmux_session,
            "launch_script": str(launch_script),
            "env_file": str(env_file),
            "health_url": health_url,
            "base_url": base_url,
            "log_path": str(log_path),
            "start_new_session": False,
        }
    else:
        log_handle = log_path.open("ab")
        process = subprocess.Popen(
            ["bash", str(launch_script), str(env_file)],
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state = {
            "status": "running",
            "pid": process.pid,
            "port": port,
            "launch_script": str(launch_script),
            "env_file": str(env_file),
            "health_url": health_url,
            "base_url": base_url,
            "log_path": str(log_path),
            "start_new_session": True,
        }
    write_json(state_path, state)

    healthy = wait_for_health(health_url, health_timeout)
    ready = wait_for_router_ready(base_url, health_timeout) if healthy else False
    state["health_ok"] = healthy
    state["router_ready"] = ready
    write_json(state_path, state)
    if not ready:
        return {**state, "status": "started_unhealthy"}
    return state


def status_router(state_path: Path, env_file: Path) -> Dict[str, Any]:
    state = read_json(state_path)
    pid = int(state.get("pid") or 0)
    health_url = str(state.get("health_url") or health_url_from_env(env_file))
    base_url = str(state.get("base_url") or router_base_url_from_env(env_file))
    _, port = router_host_port_from_env(env_file)
    port_pids = listening_pids_for_port(port)
    health_ok = wait_for_health(health_url, timeout=1.5)
    ready = router_ready(base_url, timeout=2.0) if health_ok else False
    state_pid_alive = pid_alive(pid)
    if ready:
        status = "running"
    elif port_pids:
        status = "unhealthy"
    elif state_pid_alive:
        status = "stale_state"
    else:
        status = "not_running"
    return {
        "status": status,
        "pid": pid,
        "port": port,
        "port_pids": port_pids,
        "health_ok": health_ok,
        "router_ready": ready,
        "health_url": health_url,
        "base_url": base_url,
        "state_path": str(state_path),
        "log_path": state.get("log_path"),
        "tmux_session": state.get("tmux_session"),
        "state_pid_alive": state_pid_alive,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start, stop, restart, or inspect the tool router.")
    parser.add_argument("action", choices=["start", "stop", "restart", "status"])
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--launch-script", type=Path, default=DEFAULT_LAUNCH_SCRIPT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--health-timeout", type=float, default=30.0)
    parser.add_argument("--stop-timeout", type=float, default=10.0)
    parser.add_argument("--tmux", action="store_true")
    parser.add_argument("--tmux-session", default="harness_router")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_dir = args.state_dir.resolve()
    state_path = state_dir / "router_state.json"
    log_path = state_dir / "router.log"
    env_file = args.env_file.resolve()
    launch_script = args.launch_script.resolve()

    if args.action == "status":
        print(json.dumps(status_router(state_path, env_file), indent=2))
        return 0
    if args.action == "stop":
        print(json.dumps(stop_from_state(state_path, env_file, args.stop_timeout), indent=2))
        return 0
    if args.action == "restart":
        stop_from_state(state_path, env_file, args.stop_timeout)

    result = start_router(
        launch_script=launch_script,
        env_file=env_file,
        state_path=state_path,
        log_path=log_path,
        health_timeout=args.health_timeout,
        use_tmux=args.tmux,
        tmux_session=args.tmux_session,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("router_ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
