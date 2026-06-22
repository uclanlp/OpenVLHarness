#!/usr/bin/env python3
"""Render an HTML report for a harness-search campaign."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote


State = Dict[str, Any]


COLORS = {
    "best": "#16784b",
    "pool": "#2f67b1",
    "round": "#c46a18",
    "baseline": "#7a8491",
    "single": "#2f67b1",
    "crossover": "#c46a18",
    "selected": "#111827",
    "grid": "#d8dee8",
    "text": "#1f2937",
    "muted": "#6b7280",
    "pos": "#16784b",
    "neg": "#b42318",
    "tie": "#8a6d1d",
}


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def parse_round(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"(\d+)", value)
        if match:
            return int(match.group(1))
    return None


def state_id_num(state_id: str) -> int:
    match = re.search(r"(\d+)$", state_id)
    return int(match.group(1)) if match else 10**9


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def fmt_score(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def fmt_delta(value: Optional[float]) -> str:
    if value is None:
        return "-"
    cls = "pos" if value > 1e-12 else "neg" if value < -1e-12 else "tie"
    return f'<span class="{cls}">{value * 100:+.2f} pts</span>'


def short_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def load_campaign(campaign_root: Path) -> Dict[str, Any]:
    payload = read_json(campaign_root / "campaign_state.json")
    if not isinstance(payload, dict):
        raise SystemExit(f"Could not read campaign_state.json under {campaign_root}")
    return payload


def load_states(campaign_root: Path, campaign: Dict[str, Any]) -> Dict[str, State]:
    states: Dict[str, State] = {}
    campaign_states = campaign.get("states")
    if isinstance(campaign_states, dict):
        for state_id, payload in campaign_states.items():
            if isinstance(payload, dict):
                states[str(state_id)] = dict(payload)

    for path in sorted((campaign_root / "states").glob("s*/state.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        state_id = str(payload.get("state_id") or path.parent.name)
        merged = dict(payload)
        merged.update(states.get(state_id, {}))
        states[state_id] = merged

    return states


def score_of(state: State) -> Optional[float]:
    return as_float(state.get("score"))


def round_of(state: State) -> int:
    return parse_round(state.get("created_round")) or 0


def is_evaluated(state: State) -> bool:
    return score_of(state) is not None and state.get("status") == "evaluated"


def creation_index(state: State) -> int:
    value = as_float(state.get("creation_index"))
    if value is not None:
        return int(value)
    return state_id_num(str(state.get("state_id") or ""))


def dataset_accuracy(state: State, dataset: str) -> Optional[float]:
    scores = state.get("dataset_scores")
    if isinstance(scores, dict):
        value = scores.get(dataset)
        if isinstance(value, dict):
            return as_float(value.get("accuracy"))
        return as_float(value)
    return None


def val_score_of(state: State) -> Optional[float]:
    return as_float(state.get("val_score"))


def val_dataset_accuracy(state: State, dataset: str) -> Optional[float]:
    scores = state.get("val_dataset_scores")
    if isinstance(scores, dict):
        value = scores.get(dataset)
        if isinstance(value, dict):
            return as_float(value.get("accuracy"))
        return as_float(value)
    return None


def dataset_count_text(state: State, dataset: str) -> str:
    scores = state.get("dataset_scores")
    if not isinstance(scores, dict):
        return ""
    payload = scores.get(dataset)
    if not isinstance(payload, dict):
        return ""
    correct = payload.get("correct")
    total = payload.get("total")
    if correct is None or total is None:
        return ""
    return f" ({correct}/{total})"


def collect_datasets(campaign: Dict[str, Any], states: Dict[str, State]) -> List[str]:
    datasets: List[str] = []
    for item in campaign.get("datasets") or []:
        text = str(item)
        if text not in datasets:
            datasets.append(text)
    for state in states.values():
        scores = state.get("dataset_scores")
        if isinstance(scores, dict):
            for dataset in scores:
                if str(dataset) not in datasets:
                    datasets.append(str(dataset))
    return datasets


def state_sort_key(state: State) -> Tuple[float, int, str]:
    score = score_of(state)
    return (-(score if score is not None else -1e9), creation_index(state), str(state.get("state_id") or ""))


def rank_states(states: Iterable[State]) -> List[State]:
    return sorted((state for state in states if score_of(state) is not None), key=state_sort_key)


def load_round_states(campaign_root: Path) -> Dict[int, Dict[str, Any]]:
    round_states: Dict[int, Dict[str, Any]] = {}
    for path in sorted((campaign_root / "rounds").glob("r*/round_state.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        round_num = parse_round(payload.get("round")) or parse_round(path.parts[-2])
        if round_num is not None:
            round_states[round_num] = payload
    return round_states


def pool_by_round(campaign: Dict[str, Any], round_states: Dict[int, Dict[str, Any]]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {0: ["s0000"]}
    rounds = campaign.get("rounds")
    if isinstance(rounds, list):
        for item in rounds:
            if not isinstance(item, dict):
                continue
            round_num = parse_round(item.get("round"))
            pool = item.get("pool")
            if round_num is not None and isinstance(pool, list):
                out[round_num] = [str(state_id) for state_id in pool]
    for round_num, payload in sorted(round_states.items()):
        if payload.get("status") != "complete":
            continue
        pool = payload.get("next_pool")
        if isinstance(pool, list):
            out[round_num] = [str(state_id) for state_id in pool]
    return out


def load_round_summaries(campaign_root: Path) -> Dict[int, Dict[str, Any]]:
    summaries: Dict[int, Dict[str, Any]] = {}
    for path in sorted((campaign_root / "rounds").glob("r*/bookkeeping/round_summary.json")):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        round_num = parse_round(payload.get("round")) or parse_round(path.parts[-3])
        if round_num is not None:
            summaries[round_num] = payload
    return summaries


def round_dirs(campaign_root: Path) -> List[int]:
    values: List[int] = []
    for path in sorted((campaign_root / "rounds").glob("r*")):
        if path.is_dir():
            parsed = parse_round(path.name)
            if parsed is not None:
                values.append(parsed)
    return values


def evaluated_states(states: Dict[str, State]) -> List[State]:
    return sorted(
        [state for state in states.values() if is_evaluated(state)],
        key=lambda state: (round_of(state), creation_index(state), str(state.get("state_id") or "")),
    )


def best_state_before_round(states: Sequence[State], round_num: int) -> Optional[State]:
    candidates = [state for state in states if round_of(state) <= round_num]
    return rank_states(candidates)[0] if candidates else None


def best_child_in_round(states: Sequence[State], round_num: int) -> Optional[State]:
    candidates = [
        state for state in states
        if round_of(state) == round_num and state.get("parent_state_id")
    ]
    ranked = rank_states(candidates)
    return ranked[0] if ranked else None


def state_dataset_summary(state: State, datasets: Sequence[str]) -> str:
    parts = []
    for dataset in datasets:
        value = dataset_accuracy(state, dataset)
        if value is not None:
            parts.append(f"{dataset}={fmt_pct(value)}{dataset_count_text(state, dataset)}")
    return ", ".join(parts)


def link_for(path: Path, report_path: Path, label: str) -> str:
    try:
        rel = os.path.relpath(path, report_path.parent)
    except ValueError:
        rel = str(path)
    return f'<a href="{quote(rel)}">{esc(label)}</a>'


def proposal_label(state: State) -> str:
    proposal_id = str(state.get("proposal_id") or "")
    combo_size = state.get("combo_size")
    if combo_size:
        return f"{proposal_id} ({combo_size})"
    return proposal_id


def gate_of(state: State) -> Optional[Dict[str, Any]]:
    gate = state.get("promotion_gate")
    return gate if isinstance(gate, dict) else None


def gate_num(value: Any) -> str:
    """Compact gate count: ints (k=1) print as-is; fractional k>1 values round."""
    if isinstance(value, bool) or not isinstance(value, float):
        return str(value)
    return f"{value:.2f}".rstrip("0").rstrip(".")


def fmt_gate(gate: Optional[Dict[str, Any]]) -> str:
    if not gate:
        return '<span class="muted">-</span>'
    if gate.get("reason") == "insufficient_paired_data":
        return '<span class="tie" title="no shared paired instances">n/a</span>'
    lcb = as_float(gate.get("lcb"))
    promoted = bool(gate.get("promoted"))
    cls = "pos" if promoted else "neg"
    verdict = "promoted" if promoted else "gated"
    lcb_text = f"{lcb:+.1f}" if lcb is not None else "-"
    label = (
        f"fixed {gate_num(gate.get('fixed'))} / broke {gate_num(gate.get('broke'))} "
        f"/ net {gate_num(gate.get('net'))} / LCB {lcb_text} (z={gate.get('z')})"
    )
    return f'<span class="{cls}" title="{esc(label)}">{verdict} (LCB {lcb_text})</span>'


def child_delta(state: State, states: Dict[str, State]) -> Optional[float]:
    parent_id = state.get("parent_state_id")
    if not parent_id:
        return None
    parent = states.get(str(parent_id))
    if not parent:
        return None
    score = score_of(state)
    parent_score = score_of(parent)
    if score is None or parent_score is None:
        return None
    return score - parent_score


def compute_round_rows(
    *,
    states: Dict[str, State],
    evaluated: Sequence[State],
    pools: Dict[int, List[str]],
    summaries: Dict[int, Dict[str, Any]],
    round_states: Dict[int, Dict[str, Any]],
    last_complete_round: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_num in range(1, last_complete_round + 1):
        children = [
            state for state in evaluated
            if round_of(state) == round_num and state.get("parent_state_id")
        ]
        deltas = [delta for state in children if (delta := child_delta(state, states)) is not None]
        best_child = best_child_in_round(evaluated, round_num)
        prev_best = best_state_before_round(evaluated, round_num - 1)
        best_so_far = best_state_before_round(evaluated, round_num)
        selected = pools.get(round_num, [])
        selected_scores = [
            score for state_id in selected
            if (score := score_of(states.get(state_id, {}))) is not None
        ]
        summary = summaries.get(round_num, {})
        round_state = round_states.get(round_num, {})
        counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
        skipped = (
            counts.get("skipped_crossovers")
            if isinstance(counts, dict)
            else None
        )
        if skipped is None:
            skipped = summary.get("skipped_crossover_count")
        if skipped is None and isinstance(summary.get("skipped_crossovers"), list):
            skipped = len(summary.get("skipped_crossovers") or [])
        if skipped is None and isinstance(round_state.get("skipped_crossovers"), list):
            skipped = len(round_state.get("skipped_crossovers") or [])
        if skipped is None:
            skipped = 0

        rows.append({
            "round": round_num,
            "evaluated_children": len(children),
            "improved": sum(1 for delta in deltas if delta > 1e-12),
            "tied": sum(1 for delta in deltas if abs(delta) <= 1e-12),
            "worse": sum(1 for delta in deltas if delta < -1e-12),
            "avg_delta": mean(deltas) if deltas else None,
            "best_delta": max(deltas) if deltas else None,
            "best_child": best_child,
            "best_so_far": best_so_far,
            "best_gain": (
                score_of(best_so_far) - score_of(prev_best)
                if best_so_far and prev_best and score_of(best_so_far) is not None and score_of(prev_best) is not None
                else None
            ),
            "selected_pool": selected,
            "pool_mean": mean(selected_scores) if selected_scores else None,
            "skipped_crossovers": int(skipped or 0),
        })
    return rows


def overall_correct(state: State, datasets: Sequence[str]) -> Optional[Tuple[int, int]]:
    correct = 0
    total = 0
    for dataset in datasets:
        scores = state.get("dataset_scores")
        if not isinstance(scores, dict):
            return None
        item = scores.get(dataset)
        if not isinstance(item, dict):
            return None
        if not isinstance(item.get("correct"), int) or not isinstance(item.get("total"), int):
            return None
        correct += int(item["correct"])
        total += int(item["total"])
    return correct, total


def y_domain(series: Sequence[Sequence[Tuple[int, float]]]) -> Tuple[float, float]:
    values = [value for points in series for _, value in points]
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return max(0.0, lo - 0.01), min(1.0, hi + 0.01)
    pad = (hi - lo) * 0.12
    return max(0.0, lo - pad), min(1.0, hi + pad)


def svg_line_chart(
    *,
    series: Sequence[Dict[str, Any]],
    title: str,
    width: int = 980,
    height: int = 330,
) -> str:
    non_empty = [item for item in series if item.get("points")]
    if not non_empty:
        return '<div class="empty">No chart data available.</div>'
    left, right, top, bottom = 58, 18, 24, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_values = [x for item in non_empty for x, _ in item["points"]]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_max += 1
    y_min, y_max = y_domain([item["points"] for item in non_empty])

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
        '<rect class="chart-bg" x="0" y="0" width="100%" height="100%" rx="8"></rect>',
    ]
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = sy(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"></line>')
        parts.append(f'<text class="axis" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{fmt_pct(value, 1)}</text>')
    for x in range(int(x_min), int(x_max) + 1):
        xpos = sx(x)
        label = "base" if x == 0 else f"r{x:03d}"
        parts.append(f'<line class="grid xgrid" x1="{xpos:.1f}" y1="{top}" x2="{xpos:.1f}" y2="{height-bottom}"></line>')
        parts.append(f'<text class="axis" x="{xpos:.1f}" y="{height-20}" text-anchor="middle">{label}</text>')

    for item in non_empty:
        points = item["points"]
        color = item.get("color") or COLORS["pool"]
        dash = ' stroke-dasharray="5 5"' if item.get("dash") else ""
        point_text = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline class="line" points="{point_text}" stroke="{color}"{dash}></polyline>')
        for x, y in points:
            label = f"{item.get('name')}: {'base' if x == 0 else f'r{x:03d}'} {fmt_pct(y)}"
            parts.append(
                f'<circle class="point" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" '
                f'fill="{color}"><title>{esc(label)}</title></circle>'
            )

    legend_x = left
    legend_y = 18
    for item in non_empty:
        color = item.get("color") or COLORS["pool"]
        name = str(item.get("name") or "")
        parts.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4" fill="{color}"></circle>')
        parts.append(f'<text class="legend" x="{legend_x + 9}" y="{legend_y + 4}">{esc(name)}</text>')
        legend_x += 9 + max(76, len(name) * 7)
    parts.append("</svg>")
    return "\n".join(parts)


def svg_scatter(
    *,
    states: Sequence[State],
    selected_ids: Sequence[str],
    title: str,
    width: int = 980,
    height: int = 340,
) -> str:
    points = [
        (round_of(state), score_of(state), state)
        for state in states
        if score_of(state) is not None
    ]
    points = [(x, y, state) for x, y, state in points if y is not None]
    if not points:
        return '<div class="empty">No evaluated states available.</div>'
    selected = set(selected_ids)
    left, right, top, bottom = 58, 18, 24, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = min(x for x, _, _ in points), max(x for x, _, _ in points)
    if x_min == x_max:
        x_max += 1
    y_min, y_max = y_domain([[(x, y) for x, y, _ in points]])

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
        '<rect class="chart-bg" x="0" y="0" width="100%" height="100%" rx="8"></rect>',
    ]
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = sy(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"></line>')
        parts.append(f'<text class="axis" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{fmt_pct(value, 1)}</text>')
    for x in range(int(x_min), int(x_max) + 1):
        xpos = sx(x)
        label = "base" if x == 0 else f"r{x:03d}"
        parts.append(f'<line class="grid xgrid" x1="{xpos:.1f}" y1="{top}" x2="{xpos:.1f}" y2="{height-bottom}"></line>')
        parts.append(f'<text class="axis" x="{xpos:.1f}" y="{height-20}" text-anchor="middle">{label}</text>')

    for x, y, state in points:
        state_id = str(state.get("state_id") or "")
        is_selected = state_id in selected
        is_crossover = bool(state.get("is_crossover"))
        color = COLORS["crossover"] if is_crossover else COLORS["single"]
        if x == 0:
            color = COLORS["baseline"]
        radius = 5 if is_selected else 3
        stroke = COLORS["selected"] if is_selected else "none"
        stroke_width = 1.8 if is_selected else 0
        title_text = (
            f"{state_id} r{x:03d} score {fmt_pct(y)}"
            f" parent={state.get('parent_state_id') or '-'}"
            f" proposal={state.get('proposal_id') or '-'}"
        )
        parts.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius}" '
            f'fill="{color}" fill-opacity="0.68" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"><title>{esc(title_text)}</title></circle>'
        )

    legend = [
        ("single", COLORS["single"]),
        ("crossover", COLORS["crossover"]),
        ("current pool outline", COLORS["selected"]),
    ]
    legend_x = left
    legend_y = 18
    for name, color in legend:
        parts.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4" fill="{color}"></circle>')
        parts.append(f'<text class="legend" x="{legend_x + 9}" y="{legend_y + 4}">{esc(name)}</text>')
        legend_x += 9 + max(90, len(name) * 7)
    parts.append("</svg>")
    return "\n".join(parts)


def mini_stacked_bar(improved: int, tied: int, worse: int) -> str:
    total = improved + tied + worse
    if total <= 0:
        return '<div class="stack emptybar"></div>'
    parts = ['<div class="stack">']
    for cls, value in [("good", improved), ("tiebar", tied), ("bad", worse)]:
        if value:
            width = value / total * 100
            parts.append(f'<span class="{cls}" style="width:{width:.2f}%"></span>')
    parts.append("</div>")
    return "".join(parts)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, cls: str = "") -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f'<table class="{esc(cls)}"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def extract_lessons(summaries: Dict[int, Dict[str, Any]], limit: int = 8) -> List[Tuple[int, str]]:
    lessons: List[Tuple[int, str]] = []
    for round_num in sorted(summaries, reverse=True):
        raw = summaries[round_num].get("lessons")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    lessons.append((round_num, item.strip()))
                elif isinstance(item, dict):
                    text = item.get("lesson") or item.get("summary") or item.get("text")
                    if text:
                        lessons.append((round_num, str(text).strip()))
        if len(lessons) >= limit:
            break
    return lessons[:limit]


def combo_stats(evaluated: Sequence[State], states: Dict[str, State]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[State]] = defaultdict(list)
    for state in evaluated:
        if not state.get("parent_state_id"):
            continue
        combo_size = int(state.get("combo_size") or 1)
        grouped[combo_size].append(state)
    rows = []
    for combo_size, items in sorted(grouped.items()):
        scores = [score_of(item) for item in items if score_of(item) is not None]
        deltas = [delta for item in items if (delta := child_delta(item, states)) is not None]
        best = rank_states(items)[0] if items else None
        rows.append({
            "combo_size": combo_size,
            "count": len(items),
            "mean_score": mean(scores) if scores else None,
            "mean_delta": mean(deltas) if deltas else None,
            "improved": sum(1 for delta in deltas if delta > 1e-12),
            "best": best,
        })
    return rows


def edited_file_stats(evaluated: Sequence[State], states: Dict[str, State], limit: int = 12) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[State]] = defaultdict(list)
    for state in evaluated:
        if not state.get("parent_state_id"):
            continue
        edited_files = state.get("edited_files")
        if not isinstance(edited_files, list):
            edited_files = state.get("overlay_files")
        if not isinstance(edited_files, list):
            continue
        for item in edited_files:
            grouped[str(item)].append(state)
    rows = []
    for relpath, items in grouped.items():
        deltas = [delta for item in items if (delta := child_delta(item, states)) is not None]
        scores = [score_of(item) for item in items if score_of(item) is not None]
        best = rank_states(items)[0] if items else None
        rows.append({
            "file": relpath,
            "count": len(items),
            "mean_score": mean(scores) if scores else None,
            "mean_delta": mean(deltas) if deltas else None,
            "improved": sum(1 for delta in deltas if delta > 1e-12),
            "best": best,
        })
    rows.sort(key=lambda row: (-(row["mean_delta"] if row["mean_delta"] is not None else -1e9), -row["count"], row["file"]))
    return rows[:limit]


def render_report(
    *,
    campaign_root: Path,
    output_path: Path,
    campaign: Dict[str, Any],
    states: Dict[str, State],
    datasets: Sequence[str],
    val_datasets: Sequence[str],
    top_k: int,
) -> str:
    evaluated = evaluated_states(states)
    if not evaluated:
        raise SystemExit("No evaluated states found.")

    summaries = load_round_summaries(campaign_root)
    round_states = load_round_states(campaign_root)
    pools = pool_by_round(campaign, round_states)
    baseline = states.get("s0000") or best_state_before_round(evaluated, 0) or evaluated[0]
    best = rank_states(evaluated)[0]
    baseline_score = score_of(baseline)
    best_score = score_of(best)
    last_complete_round = max([0] + [round_num for round_num in pools if round_num > 0])
    current_pool = [str(item) for item in pools.get(last_complete_round, campaign.get("pool") or [])]
    max_eval_round = max(round_of(state) for state in evaluated)
    unfinished_rounds = [
        round_num for round_num in round_dirs(campaign_root)
        if round_num > last_complete_round
        and round_states.get(round_num, {}).get("status") != "complete"
    ]
    campaign_round_nums = {
        parse_round(item.get("round"))
        for item in campaign.get("rounds", [])
        if isinstance(item, dict)
    }
    round_state_only = sorted(
        round_num for round_num in pools
        if round_num and round_num not in campaign_round_nums
    )
    status_counts = Counter(str(state.get("status") or "unknown") for state in states.values())

    round_rows = compute_round_rows(
        states=states,
        evaluated=evaluated,
        pools=pools,
        summaries=summaries,
        round_states=round_states,
        last_complete_round=last_complete_round,
    )

    chart_rounds = list(range(0, max(last_complete_round, max_eval_round) + 1))
    best_points: List[Tuple[int, float]] = []
    pool_mean_points: List[Tuple[int, float]] = []
    round_best_points: List[Tuple[int, float]] = []
    for round_num in chart_rounds:
        best_state = best_state_before_round(evaluated, round_num)
        if best_state and score_of(best_state) is not None:
            best_points.append((round_num, float(score_of(best_state))))
        pool_ids = pools.get(round_num)
        if pool_ids:
            scores = [
                score for state_id in pool_ids
                if (score := score_of(states.get(state_id, {}))) is not None
            ]
            if scores:
                pool_mean_points.append((round_num, mean(scores)))
        round_best = best_child_in_round(evaluated, round_num) if round_num else baseline
        if round_best and score_of(round_best) is not None:
            round_best_points.append((round_num, float(score_of(round_best))))

    dataset_series = []
    for index, dataset in enumerate(datasets):
        points: List[Tuple[int, float]] = []
        for round_num in chart_rounds:
            best_state = best_state_before_round(evaluated, round_num)
            if best_state:
                value = dataset_accuracy(best_state, dataset)
                if value is not None:
                    points.append((round_num, value))
        palette = ["#16784b", "#2f67b1", "#c46a18", "#7b4da2", "#b42318"]
        dataset_series.append({
            "name": dataset,
            "points": points,
            "color": palette[index % len(palette)],
        })

    # Held-out validation of the best parent, recorded per completed round plus the baseline.
    val_baseline_score = val_score_of(baseline)
    val_points: List[Tuple[int, float]] = []
    if val_baseline_score is not None:
        val_points.append((0, float(val_baseline_score)))
    validation_round_entries: List[Tuple[int, Dict[str, Any]]] = []
    for round_num in sorted(round_states):
        round_state = round_states[round_num]
        if round_state.get("status") != "complete":
            continue
        validation = round_state.get("validation")
        if not isinstance(validation, dict):
            continue
        round_val_score = as_float(validation.get("val_score"))
        validation_round_entries.append((round_num, validation))
        if round_val_score is not None:
            val_points.append((round_num, float(round_val_score)))

    best_series = [
        {"name": "best found", "points": best_points, "color": COLORS["best"]},
        {"name": "selected pool mean", "points": pool_mean_points, "color": COLORS["pool"]},
        {"name": "round best child", "points": round_best_points, "color": COLORS["round"], "dash": True},
    ]
    if val_datasets:
        best_series.append({
            "name": "best-parent val (held-out)",
            "points": val_points,
            "color": "#7b4da2",
            "dash": True,
        })
    best_line = svg_line_chart(title="Overall score trend", series=best_series)
    dataset_line = svg_line_chart(title="Dataset accuracy of best-found state", series=dataset_series)
    scatter = svg_scatter(
        title="Evaluated state score distribution",
        states=evaluated,
        selected_ids=current_pool,
    )

    best_correct = overall_correct(best, datasets)
    base_correct = overall_correct(baseline, datasets)
    correct_delta_text = ""
    if best_correct and base_correct and best_correct[1] == base_correct[1]:
        correct_delta_text = f"{best_correct[0] - base_correct[0]:+d} / {best_correct[1]} examples"

    current_pool_scores = [
        score for state_id in current_pool
        if (score := score_of(states.get(state_id, {}))) is not None
    ]
    current_pool_mean = mean(current_pool_scores) if current_pool_scores else None
    current_pool_best = max(current_pool_scores) if current_pool_scores else None

    dataset_rows = []
    for dataset in datasets:
        base_value = dataset_accuracy(baseline, dataset)
        best_any: Optional[State] = None
        best_value: Optional[float] = None
        current_best_value: Optional[float] = None
        current_best_state: Optional[State] = None
        for state in evaluated:
            value = dataset_accuracy(state, dataset)
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
                best_any = state
            if str(state.get("state_id")) in set(current_pool):
                if current_best_value is None or value > current_best_value:
                    current_best_value = value
                    current_best_state = state
        dataset_rows.append([
            esc(dataset),
            fmt_pct(base_value) + esc(dataset_count_text(baseline, dataset)),
            (
                f"{fmt_pct(best_value)} "
                f"{link_for(campaign_root / 'states' / str(best_any.get('state_id')) / 'state.json', output_path, str(best_any.get('state_id')))}"
                if best_any else "-"
            ),
            (
                f"{fmt_pct(current_best_value)} "
                f"{link_for(campaign_root / 'states' / str(current_best_state.get('state_id')) / 'state.json', output_path, str(current_best_state.get('state_id')))}"
                if current_best_state else "-"
            ),
            fmt_delta((best_value - base_value) if best_value is not None and base_value is not None else None),
        ])

    # Held-out validation table: baseline row + one row per completed round's best parent.
    def validation_row(label: str, state: Optional[State]) -> List[str]:
        state = state or {}
        state_id = str(state.get("state_id") or "")
        held_out = val_score_of(state)
        per_dataset = ", ".join(
            f"{esc(dataset)}={fmt_pct(value)}"
            for dataset in val_datasets
            if (value := val_dataset_accuracy(state, dataset)) is not None
        )
        return [
            esc(label),
            link_for(campaign_root / "states" / state_id / "state.json", output_path, state_id) if state_id else "-",
            fmt_pct(score_of(state)),
            fmt_pct(held_out),
            per_dataset or "-",
            fmt_delta(
                (held_out - val_baseline_score)
                if held_out is not None and val_baseline_score is not None
                else None
            ),
        ]

    validation_rows: List[List[str]] = []
    if val_datasets:
        validation_rows.append(validation_row("base", baseline))
        for round_num, validation in validation_round_entries:
            best_state = states.get(str(validation.get("state_id")), {})
            validation_rows.append(validation_row(f"r{round_num:03d}", best_state))

    latest_val_score = (
        as_float(validation_round_entries[-1][1].get("val_score"))
        if validation_round_entries
        else val_baseline_score
    )
    val_gain = (
        latest_val_score - val_baseline_score
        if latest_val_score is not None and val_baseline_score is not None
        else None
    )

    round_table_rows = []
    for item in round_rows:
        best_child = item["best_child"]
        best_child_cell = "-"
        if best_child:
            sid = str(best_child.get("state_id"))
            best_child_cell = (
                f"{link_for(campaign_root / 'states' / sid / 'state.json', output_path, sid)} "
                f"{fmt_pct(score_of(best_child))} {fmt_delta(item['best_delta'])}"
            )
        selected_pool = ", ".join(
            link_for(campaign_root / "states" / state_id / "state.json", output_path, state_id)
            for state_id in item["selected_pool"]
        )
        round_table_rows.append([
            f"r{item['round']:03d}",
            str(item["evaluated_children"]),
            mini_stacked_bar(item["improved"], item["tied"], item["worse"]),
            f"{item['improved']} / {item['tied']} / {item['worse']}",
            fmt_delta(item["avg_delta"]),
            best_child_cell,
            fmt_delta(item["best_gain"]),
            fmt_pct(item["pool_mean"]),
            str(item["skipped_crossovers"]),
            selected_pool,
        ])

    current_pool_rows = []
    for state_id in current_pool:
        state = states.get(state_id)
        if not state:
            continue
        current_pool_rows.append([
            link_for(campaign_root / "states" / state_id / "state.json", output_path, state_id),
            f"r{round_of(state):03d}",
            fmt_pct(score_of(state)),
            fmt_delta((score_of(state) - baseline_score) if score_of(state) is not None and baseline_score is not None else None),
            esc(state_dataset_summary(state, datasets)),
            esc(proposal_label(state)),
            esc(short_text(state.get("patch_summary"), 220)),
        ])

    top_state_rows = []
    for state in rank_states(evaluated)[:top_k]:
        state_id = str(state.get("state_id"))
        top_state_rows.append([
            link_for(campaign_root / "states" / state_id / "state.json", output_path, state_id),
            f"r{round_of(state):03d}",
            esc(str(state.get("parent_state_id") or "-")),
            fmt_pct(score_of(state)),
            fmt_delta((score_of(state) - baseline_score) if score_of(state) is not None and baseline_score is not None else None),
            fmt_gate(gate_of(state)),
            esc(state_dataset_summary(state, datasets)),
            esc(proposal_label(state)),
            esc(short_text(state.get("patch_summary"), 240)),
        ])

    combo_rows = []
    for row in combo_stats(evaluated, states):
        best_state = row["best"]
        combo_rows.append([
            str(row["combo_size"]),
            str(row["count"]),
            fmt_pct(row["mean_score"]),
            fmt_delta(row["mean_delta"]),
            str(row["improved"]),
            (
                f"{link_for(campaign_root / 'states' / str(best_state.get('state_id')) / 'state.json', output_path, str(best_state.get('state_id')))} "
                f"{fmt_pct(score_of(best_state))}"
                if best_state else "-"
            ),
        ])

    file_rows = []
    for row in edited_file_stats(evaluated, states):
        best_state = row["best"]
        file_rows.append([
            f"<code>{esc(row['file'])}</code>",
            str(row["count"]),
            fmt_pct(row["mean_score"]),
            fmt_delta(row["mean_delta"]),
            str(row["improved"]),
            (
                f"{link_for(campaign_root / 'states' / str(best_state.get('state_id')) / 'state.json', output_path, str(best_state.get('state_id')))} "
                f"{fmt_pct(score_of(best_state))}"
                if best_state else "-"
            ),
        ])

    failed_states = [
        state for state in states.values()
        if state.get("status") != "evaluated"
    ]
    failed_rows = []
    for state in sorted(failed_states, key=lambda s: (round_of(s), creation_index(s)))[:25]:
        state_id = str(state.get("state_id") or "")
        failed_rows.append([
            link_for(campaign_root / "states" / state_id / "state.json", output_path, state_id) if state_id else "-",
            f"r{round_of(state):03d}",
            esc(str(state.get("parent_state_id") or "-")),
            esc(str(state.get("status") or "unknown")),
            esc(short_text(state.get("error") or state.get("reason") or state.get("patch_summary"), 220)),
        ])

    lesson_items = [
        f'<li><span class="pill">r{round_num:03d}</span> {esc(short_text(text, 360))}</li>'
        for round_num, text in extract_lessons(summaries)
    ]

    unfinished_text = ""
    if unfinished_rounds:
        unfinished_text = (
            '<p class="callout warn">Found artifact directories for unfinished round(s): '
            + ", ".join(f"r{round_num:03d}" for round_num in unfinished_rounds)
            + ". These proposals are not included in completed-round trend lines until evaluated.</p>"
        )
    round_state_only_text = ""
    if round_state_only:
        round_state_only_text = (
            '<p class="callout warn">Using completed round_state.json for round(s) not yet reflected in '
            'campaign_state.json: '
            + ", ".join(f"r{round_num:03d}" for round_num in round_state_only)
            + ".</p>"
        )

    val_card = ""
    validation_section = ""
    if val_datasets:
        val_card = (
            '<div class="card"><div class="label">Best Val (held-out)</div>'
            f'<div class="value">{fmt_pct(latest_val_score)}</div>'
            f'<div class="sub">vs baseline {fmt_delta(val_gain)} on {esc(", ".join(val_datasets))}</div></div>'
        )
        validation_section = f"""
  <h2>Validation (held-out test)</h2>
  <p class="muted">Best parent re-evaluated on <strong>{esc(", ".join(val_datasets))}</strong> after each round. These scores do not drive search ranking.</p>
  <div class="panel">
    {table(['round', 'best parent', 'search score (val_train)', 'held-out (val_test)', 'per-dataset held-out', 'delta vs baseline'], validation_rows) if validation_rows else '<div class="empty">No validation results recorded yet.</div>'}
  </div>
"""

    # Paired LCB promotion gate: which evaluated children cleared the bar vs their parent.
    gate_section = ""
    gate_children = [
        state for state in evaluated
        if state.get("parent_state_id") and gate_of(state) is not None
    ]
    if gate_children:
        gate_z_values = sorted({g.get("z") for state in gate_children if (g := gate_of(state)) is not None})
        gate_z_text = ", ".join(str(z) for z in gate_z_values if z is not None) or "-"
        promoted_count = sum(1 for state in gate_children if (gate_of(state) or {}).get("promoted"))
        gated_count = len(gate_children) - promoted_count
        gate_rows = []
        for state in sorted(
            gate_children,
            key=lambda s: (round_of(s), -((gate_of(s) or {}).get("lcb") or -1e9), creation_index(s)),
        ):
            gate = gate_of(state) or {}
            state_id = str(state.get("state_id") or "")
            gate_rows.append([
                link_for(campaign_root / "states" / state_id / "state.json", output_path, state_id),
                f"r{round_of(state):03d}",
                esc(str(state.get("parent_state_id") or "-")),
                gate_num(gate.get("fixed")),
                gate_num(gate.get("broke")),
                gate_num(gate.get("net")),
                gate_num(gate.get("discordant")),
                fmt_score(as_float(gate.get("lcb"))),
                fmt_gate(gate),
            ])
        gate_section = f"""
  <h2>Promotion Gate (paired LCB)</h2>
  <p class="muted">A child is promotable only when <code>(fixed - broke) - z*sqrt(fixed + broke) &gt; 0</code> vs its parent on the shared eval instances (z={esc(gate_z_text)}). Gated children stay out of the next pool regardless of raw accuracy. <strong>{promoted_count}</strong> promoted, <strong>{gated_count}</strong> gated.</p>
  <div class="panel">
    {table(['child', 'round', 'parent', 'fixed', 'broke', 'net', 'discordant', 'LCB', 'verdict'], gate_rows)}
  </div>
"""

    campaign_id = campaign.get("campaign_id") or campaign_root.name
    title = f"Harness Search Report - {campaign_id}"
    generated_note = f"Generated from {campaign_root}"
    best_gain = (best_score - baseline_score) if best_score is not None and baseline_score is not None else None
    rel_gain = (
        best_gain / baseline_score
        if best_gain is not None and baseline_score not in (None, 0)
        else None
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d8dee8;
      --good: {COLORS['pos']};
      --bad: {COLORS['neg']};
      --tie: {COLORS['tie']};
      --blue: {COLORS['pool']};
      --orange: {COLORS['round']};
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 22px 48px;
    }}
    header {{
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 18px 0 8px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    a {{
      color: #245a9c;
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: normal;
    }}
    .muted {{ color: var(--muted); }}
    .grid-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 16px 0 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      min-height: 74px;
    }}
    .card .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .card .value {{
      font-size: 22px;
      font-weight: 700;
      line-height: 1.15;
    }}
    .card .sub {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 5px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 12px 0;
      overflow: auto;
    }}
    .charts {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .chart-bg {{ fill: #fff; }}
    .grid {{
      stroke: var(--line);
      stroke-width: 1;
    }}
    .xgrid {{ stroke-dasharray: 2 5; }}
    .axis {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .legend {{
      fill: var(--ink);
      font-size: 12px;
    }}
    .line {{
      fill: none;
      stroke-width: 2.4;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .point {{ stroke: #fff; stroke-width: 1; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: var(--panel);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #374151;
      background: #eef2f7;
      font-size: 12px;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr:hover td {{ background: #fafcff; }}
    .pos {{ color: var(--good); font-weight: 650; }}
    .neg {{ color: var(--bad); font-weight: 650; }}
    .tie {{ color: var(--tie); font-weight: 650; }}
    .pill {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 1px 7px;
      background: #f9fafb;
      color: #374151;
      font-size: 12px;
      font-weight: 650;
    }}
    .callout {{
      border-left: 4px solid var(--blue);
      background: #f8fbff;
      padding: 10px 12px;
      margin: 12px 0;
    }}
    .callout.warn {{
      border-left-color: var(--orange);
      background: #fffaf3;
    }}
    .stack {{
      display: flex;
      width: 150px;
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: #eef2f7;
      border: 1px solid #d7dde8;
    }}
    .stack span {{ display: block; height: 100%; }}
    .stack .good {{ background: var(--good); }}
    .stack .tiebar {{ background: #d9b44a; }}
    .stack .bad {{ background: var(--bad); }}
    .empty {{
      color: var(--muted);
      padding: 18px;
    }}
    ul.lessons {{
      margin: 6px 0 0;
      padding-left: 20px;
    }}
    ul.lessons li {{ margin: 6px 0; }}
    @media (max-width: 760px) {{
      main {{ padding: 20px 12px 36px; }}
      h1 {{ font-size: 23px; }}
      .card .value {{ font-size: 18px; }}
      th, td {{ padding: 7px 6px; }}
      .stack {{ width: 90px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{esc(title)}</h1>
    <div class="muted">{esc(generated_note)}</div>
  </header>

  <section class="grid-cards">
    <div class="card"><div class="label">Campaign Status</div><div class="value">{esc(campaign.get('status') or '-')}</div><div class="sub">updated {esc(campaign.get('updated_at') or campaign.get('finished_at') or '-')}</div></div>
    <div class="card"><div class="label">Baseline Score</div><div class="value">{fmt_pct(baseline_score)}</div><div class="sub">{esc(state_dataset_summary(baseline, datasets))}</div></div>
    <div class="card"><div class="label">Best Score</div><div class="value">{fmt_pct(best_score)}</div><div class="sub">{link_for(campaign_root / 'states' / str(best.get('state_id')) / 'state.json', output_path, str(best.get('state_id')))} {fmt_delta(best_gain)}</div></div>
    <div class="card"><div class="label">Relative Gain</div><div class="value">{f'{rel_gain * 100:.1f}%' if rel_gain is not None else '-'}</div><div class="sub">{esc(correct_delta_text or 'correct-count delta unavailable')}</div></div>
    <div class="card"><div class="label">Current Pool Mean</div><div class="value">{fmt_pct(current_pool_mean)}</div><div class="sub">best in pool {fmt_pct(current_pool_best)}</div></div>
    <div class="card"><div class="label">Evaluated States</div><div class="value">{len(evaluated)}</div><div class="sub">{esc(dict(status_counts))}</div></div>
    {val_card}
  </section>

  <p class="callout">
    Best mean accuracy improved from <strong>{fmt_pct(baseline_score)}</strong> at baseline to
    <strong>{fmt_pct(best_score)}</strong> at {link_for(campaign_root / 'states' / str(best.get('state_id')) / 'state.json', output_path, str(best.get('state_id')))},
    a {fmt_delta(best_gain)} change over {len(datasets)} dataset(s). Last completed round: <strong>r{last_complete_round:03d}</strong>.
  </p>
  {unfinished_text}
  {round_state_only_text}

  <h2>Score Trends</h2>
  <div class="charts">
    <div class="panel">{best_line}</div>
    <div class="panel">{dataset_line}</div>
    <div class="panel">{scatter}</div>
  </div>

  <h2>Dataset View</h2>
  <div class="panel">
    {table(['dataset', 'baseline', 'best anywhere', 'best current-pool state', 'best gain'], dataset_rows)}
  </div>
{validation_section}

  <h2>Round Outcomes</h2>
  <div class="panel">
    {table(['round', 'evaluated children', 'delta mix', 'improve/tie/worse', 'avg child delta', 'best child', 'best-found gain', 'pool mean', 'skipped crossovers', 'selected pool'], round_table_rows)}
  </div>

  <h2>Current Pool</h2>
  <div class="panel">
    {table(['state', 'round', 'score', 'delta vs baseline', 'datasets', 'proposal', 'summary'], current_pool_rows)}
  </div>

  <h2>Top States</h2>
  <div class="panel">
    {table(['state', 'round', 'parent', 'score', 'delta vs baseline', 'promotion gate', 'datasets', 'proposal', 'summary'], top_state_rows)}
  </div>
{gate_section}

  <h2>Search Shape</h2>
  <div class="panel">
    <h3>By Combo Size</h3>
    {table(['combo size', 'count', 'mean score', 'mean delta vs parent', 'improved count', 'best state'], combo_rows)}
    <h3>By Edited File</h3>
    {table(['file', 'count', 'mean score', 'mean delta vs parent', 'improved count', 'best state'], file_rows)}
  </div>

  <h2>Recent Bookkeeping Lessons</h2>
  <div class="panel">
    <ul class="lessons">
      {''.join(lesson_items) if lesson_items else '<li>No lessons found in round summaries.</li>'}
    </ul>
  </div>

  <h2>Non-Evaluated States</h2>
  <div class="panel">
    {table(['state', 'round', 'parent', 'status', 'note'], failed_rows) if failed_rows else '<div class="empty">No non-evaluated states found.</div>'}
  </div>

  <h2>Artifacts</h2>
  <div class="panel">
    <p>{link_for(campaign_root / 'campaign_state.json', output_path, 'campaign_state.json')} &nbsp; {link_for(campaign_root / 'RUNBOOK.md', output_path, 'RUNBOOK.md')}</p>
  </div>
</main>
</body>
</html>
"""
    return html_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML visualization report for a harness-search campaign.")
    parser.add_argument("campaign_root", type=Path, help="Path to a harness-search campaign directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path. Defaults to <campaign_root>/search_report.html.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of top states to list.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    campaign = load_campaign(campaign_root)
    states = load_states(campaign_root, campaign)
    datasets = collect_datasets(campaign, states)
    val_datasets = [str(item) for item in campaign.get("val_datasets") or []]
    output_path = args.output.expanduser().resolve() if args.output else campaign_root / "search_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(
        campaign_root=campaign_root,
        output_path=output_path,
        campaign=campaign,
        states=states,
        datasets=datasets,
        val_datasets=val_datasets,
        top_k=max(1, args.top_k),
    )
    output_path.write_text(report, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
