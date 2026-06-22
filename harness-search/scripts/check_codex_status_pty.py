#!/usr/bin/env python3
"""Headless `/status` probe for Codex CLI using a short-lived PTY session."""

from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import re
import struct
import select
import signal
import subprocess
import sys
import time
import termios
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\a]*(?:\a|\x1b\\)|\x1b[@-Z\\-_]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
PERCENT_RE = re.compile(r"(?P<remaining>\d+(?:\.\d+)?)\s*%\s*(?:left|remaining)", re.IGNORECASE)
RESET_RE = re.compile(r"resets?\s+(?P<reset>[^\n\r,;|]+)", re.IGNORECASE)


def clean_terminal_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_reset_time(reset_text: str, now: datetime) -> Optional[datetime]:
    text = reset_text.strip().strip("()[]")
    text = text.split(")", 1)[0].strip()
    text = re.sub(r"[^\w:\s]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    candidates = [
        "%H:%M",
        "%I:%M %p",
        "%I:%M%p",
        "%H:%M on %d %b",
        "%H:%M on %b %d",
        "%I:%M %p on %d %b",
        "%I:%M %p on %b %d",
    ]
    for fmt in candidates:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if "%d" in fmt or "%b" in fmt:
            candidate = parsed.replace(year=now.year, tzinfo=now.tzinfo)
            if candidate < now - timedelta(days=1):
                candidate = candidate.replace(year=now.year + 1)
            return candidate
        candidate = now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
        if candidate < now:
            candidate += timedelta(days=1)
        return candidate

    isoish = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(isoish)
    except ValueError:
        return None


def parse_limit_line(line: str, now: datetime) -> Optional[Dict[str, Any]]:
    percent_match = PERCENT_RE.search(line)
    if not percent_match:
        return None
    remaining = float(percent_match.group("remaining"))

    reset_at = None
    reset_match = RESET_RE.search(line)
    if reset_match:
        reset_at = parse_reset_time(reset_match.group("reset"), now)

    return {
        "remaining_percent": remaining,
        "used_percent": max(0.0, 100.0 - remaining),
        "reset_at": reset_at.isoformat(timespec="seconds") if reset_at else None,
        "reset_epoch": int(reset_at.timestamp()) if reset_at else None,
        "raw": line,
    }


def parse_status_output(text: str, now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now().astimezone()
    cleaned = clean_terminal_text(text)
    status: Dict[str, Any] = {
        "sampled_at": now.isoformat(timespec="seconds"),
        "five_hour": None,
        "weekly": None,
    }
    for line in cleaned.splitlines():
        lowered = line.lower()
        if ("5h" in lowered or "5 h" in lowered or "five-hour" in lowered or "5-hour" in lowered) and "limit" in lowered:
            status["five_hour"] = parse_limit_line(line, now)
        elif "weekly" in lowered and "limit" in lowered:
            status["weekly"] = parse_limit_line(line, now)
    return status


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=2)
        return
    except Exception:
        pass
    try:
        process.kill()
    except Exception:
        pass


def run_status_probe(args: argparse.Namespace) -> Dict[str, Any]:
    master_fd, slave_fd = pty.openpty()
    try:
        winsize = struct.pack("HHHH", int(args.rows), int(args.cols), 0, 0)
        termios.tcsetwinsize(slave_fd, (int(args.rows), int(args.cols)))
    except AttributeError:
        try:
            import fcntl

            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
        except Exception:
            pass
    except Exception:
        pass
    env = os.environ.copy()
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("NO_COLOR", "1")

    command = [
        args.codex_bin,
        "--no-alt-screen",
        "--cd",
        str(args.cd),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.reasoning_effort:
        command.extend(["--config", f'model_reasoning_effort="{args.reasoning_effort}"'])

    output = bytearray()
    process = subprocess.Popen(
        command,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=str(args.cd),
        env=env,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        close_fds=True,
    )
    os.close(slave_fd)

    try:
        deadline = time.monotonic() + args.timeout
        sent_status = False
        sent_exit = False
        startup_at = time.monotonic() + args.startup_delay
        while time.monotonic() < deadline:
            if not sent_status and time.monotonic() >= startup_at:
                os.write(master_fd, b"/status\r")
                sent_status = True
            if sent_status and not sent_exit and time.monotonic() > deadline - max(1.0, args.exit_delay):
                os.write(master_fd, b"/exit\r")
                sent_exit = True

            readable, _, _ = select.select([master_fd], [], [], 0.1)
            if readable:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                output.extend(chunk)
                cleaned = clean_terminal_text(output.decode("utf-8", errors="replace"))
                if "weekly" in cleaned.lower() and ("5h" in cleaned.lower() or "5-hour" in cleaned.lower()):
                    time.sleep(args.exit_delay)
                    if not sent_exit:
                        os.write(master_fd, b"/exit\r")
                        sent_exit = True
            if sent_exit and process.poll() is not None:
                break
        if not sent_exit:
            try:
                os.write(master_fd, b"/exit\r")
            except OSError:
                pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except Exception:
                    pass
            terminate_process(process)
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    raw_text = output.decode("utf-8", errors="replace")
    status = parse_status_output(raw_text)
    status["command"] = command
    if args.debug_raw:
        status["raw_output"] = clean_terminal_text(raw_text)
    if not status.get("five_hour"):
        cleaned = clean_terminal_text(raw_text)
        raise RuntimeError(
            "could not parse 5-hour Codex limit from PTY status output"
            + (f"; cleaned_output={cleaned!r}" if cleaned else "; cleaned_output=<empty>")
        )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read Codex /status in headless mode via PTY.")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--cd", type=Path, default=Path.cwd())
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--exit-delay", type=float, default=1.0)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--debug-raw", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        status = run_status_probe(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "status": status}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
