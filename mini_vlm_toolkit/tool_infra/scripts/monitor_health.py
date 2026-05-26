#!/usr/bin/env python3
"""Live health monitor for the tool router.

Polls /health, /backend-status, and /threadpool-info on an interval and prints
a compact, scrollable status frame. Highlights saturated backends and a tight
threadpool so a hung benchmark is easy to diagnose at a glance.

Usage:
    python scripts/monitor_health.py
    python scripts/monitor_health.py --url http://localhost:8101 --interval 1
    python scripts/monitor_health.py --once
    python scripts/monitor_health.py --csv /tmp/router-health.csv
"""

import argparse
import csv
import datetime as dt
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request


HEALTH_LATENCY_WARN_MS = 100.0  # /health should be sub-10ms; flag at 100ms.
THREADPOOL_TIGHT_AVAILABLE = 4   # Fewer than this is "tight".


def _get_json(base_url, path, timeout):
    url = base_url.rstrip("/") + path
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
        latency_ms = (time.time() - t0) * 1000.0
        return json.loads(payload.decode("utf-8")), latency_ms, None
    except urllib.error.HTTPError as exc:
        latency_ms = (time.time() - t0) * 1000.0
        return None, latency_ms, f"HTTP {exc.code}"
    except Exception as exc:
        latency_ms = (time.time() - t0) * 1000.0
        return None, latency_ms, f"{type(exc).__name__}: {exc}"


def _fmt_health(payload, latency_ms, err):
    if err is not None:
        return f"/health           : ERROR ({err}) after {latency_ms:.1f}ms"
    tag = "SLOW " if latency_ms >= HEALTH_LATENCY_WARN_MS else ""
    tool_count = payload.get("tool_count", "?") if isinstance(payload, dict) else "?"
    return f"/health           : 200 in {latency_ms:6.1f}ms  {tag}tools={tool_count}"


def _fmt_threadpool(payload, err):
    if err is not None or not isinstance(payload, dict):
        return f"/threadpool-info  : ERROR ({err or 'no payload'})"
    total = payload.get("total_tokens")
    borrowed = payload.get("borrowed_tokens")
    available = payload.get("available_tokens")
    tag = ""
    if isinstance(available, int) and available <= THREADPOOL_TIGHT_AVAILABLE:
        tag = "  [TIGHT]"
    return f"/threadpool       : borrowed={borrowed}/{total}  available={available}{tag}"


def _fmt_backend_status(payload, err, saturated_streak):
    lines = ["/backend-status:"]
    if err is not None or not isinstance(payload, dict):
        lines.append(f"  ERROR ({err or 'no payload'})")
        return lines
    # Stable column widths regardless of which backends appear.
    name_w = max((len(name) for name in payload), default=4)
    for name in sorted(payload):
        entry = payload[name] if isinstance(payload[name], dict) else {}
        limit = entry.get("limit", "?")
        in_flight = entry.get("in_flight", "?")
        full = isinstance(limit, int) and isinstance(in_flight, int) and in_flight >= limit and limit > 0
        if full:
            saturated_streak[name] = saturated_streak.get(name, 0) + 1
            streak = saturated_streak[name]
            tag = f"  [FULL]  saturated {streak} poll{'s' if streak != 1 else ''}"
        else:
            saturated_streak[name] = 0
            tag = ""
        lines.append(
            f"  {name:<{name_w}}  in_flight={in_flight:>3} / {str(limit):<3}{tag}"
        )
    return lines


def _fmt_tool_status(payload, err, top_n):
    lines = ["/tool-status:"]
    if err is not None or not isinstance(payload, dict):
        lines.append(f"  (unavailable: {err or 'no payload'})")
        return lines
    tools = payload.get("tools") or []
    total = payload.get("total_in_flight", sum(t.get("in_flight", 0) for t in tools))
    lines[0] = f"/tool-status      : total_in_flight={total}"
    if not tools:
        lines.append("  (no tools currently in flight)")
        return lines
    name_w = max(len(t.get("tool_name", "")) for t in tools[:top_n])
    for entry in tools[:top_n]:
        name = entry.get("tool_name", "?")
        in_flight = entry.get("in_flight", "?")
        gated = entry.get("gated")
        backend = entry.get("backend")
        tag = f"  [gated:{backend}]" if gated else "  [ungated]"
        lines.append(f"  {name:<{name_w}}  in_flight={in_flight:>3}{tag}")
    if len(tools) > top_n:
        lines.append(f"  ... +{len(tools) - top_n} more tools in flight")
    return lines


def _render(base_url, when, frames):
    print(f"\n=== {when.isoformat(timespec='seconds')}  {base_url}")
    for line in frames:
        print(line)
    sys.stdout.flush()


def _csv_row(base_url, when, health, health_lat, threadpool, backend_status):
    row = {
        "timestamp": when.isoformat(timespec="seconds"),
        "url": base_url,
        "health_latency_ms": f"{health_lat:.2f}" if health_lat is not None else "",
        "health_ok": "1" if isinstance(health, dict) and health.get("status") == "ok" else "0",
    }
    if isinstance(threadpool, dict):
        row["threadpool_total"] = threadpool.get("total_tokens", "")
        row["threadpool_borrowed"] = threadpool.get("borrowed_tokens", "")
        row["threadpool_available"] = threadpool.get("available_tokens", "")
    if isinstance(backend_status, dict):
        for backend, entry in backend_status.items():
            if not isinstance(entry, dict):
                continue
            row[f"{backend}_limit"] = entry.get("limit", "")
            row[f"{backend}_in_flight"] = entry.get("in_flight", "")
    return row


def _open_csv(path, fieldnames_seed):
    new_file = not os.path.exists(path)
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames_seed, extrasaction="ignore")
    if new_file:
        writer.writeheader()
    return fh, writer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=os.environ.get("TOOL_SERVER_URL", "http://localhost:8101"),
        help="Router base URL (default: $TOOL_SERVER_URL or http://localhost:8101)",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0,
        help="HTTP timeout per request in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Print one frame and exit (useful for ad-hoc checks)",
    )
    parser.add_argument(
        "--csv", default=None, metavar="PATH",
        help="Append each poll as a CSV row to PATH (in addition to console output)",
    )
    parser.add_argument(
        "--top-tools", type=int, default=10, metavar="N",
        help="Show top-N in-flight tools from /tool-status (default: 10)",
    )
    args = parser.parse_args()

    saturated_streak = {}
    csv_fh = None
    csv_writer = None
    csv_fields = None

    def _cleanup(*_):
        if csv_fh is not None:
            csv_fh.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    while True:
        when = dt.datetime.now()
        health, health_lat, health_err = _get_json(args.url, "/health", args.timeout)
        threadpool, _, threadpool_err = _get_json(args.url, "/threadpool-info", args.timeout)
        backend_status, _, backend_err = _get_json(args.url, "/backend-status", args.timeout)
        tool_status, _, tool_err = _get_json(args.url, "/tool-status", args.timeout)

        frames = [_fmt_health(health, health_lat, health_err),
                  _fmt_threadpool(threadpool, threadpool_err)]
        frames.extend(_fmt_backend_status(backend_status, backend_err, saturated_streak))
        frames.extend(_fmt_tool_status(tool_status, tool_err, args.top_tools))
        _render(args.url, when, frames)

        if args.csv:
            row = _csv_row(args.url, when, health, health_lat, threadpool, backend_status)
            if csv_writer is None:
                csv_fields = list(row.keys())
                csv_fh, csv_writer = _open_csv(args.csv, csv_fields)
            # Grow header on the fly if a new backend appears mid-run.
            for key in row:
                if key not in csv_fields:
                    csv_fields.append(key)
                    csv_fh.close()
                    csv_fh, csv_writer = _open_csv(args.csv, csv_fields)
            csv_writer.writerow(row)
            csv_fh.flush()

        if args.once:
            break
        time.sleep(args.interval)

    if csv_fh is not None:
        csv_fh.close()


if __name__ == "__main__":
    main()
