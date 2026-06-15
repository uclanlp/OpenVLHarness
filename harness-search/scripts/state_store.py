#!/usr/bin/env python3
"""Small full-file overlay state store for harness search."""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_RSYNC_EXCLUDES = [
    ".git/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".env",
    ".env.latest",
    "outputs/",
    "mini_vlm_toolkit/tool_infra/tests/output/",
    "reference/",
    "harness-optimization/runs/",
    "harness-search/runs/",
    "harness_search_logs/",
]


class ProposalConflict(ValueError):
    """Raised when proposal files cannot be conservatively merged."""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_relpath(path: Path | str) -> str:
    text = str(path).replace("\\", "/").strip("/")
    if not text or text.startswith("../") or "/../" in f"/{text}/":
        raise ValueError(f"unsafe repo-relative path: {path!r}")
    if Path(text).is_absolute():
        raise ValueError(f"absolute paths are not allowed in overlays: {path!r}")
    return text


def is_allowed_edit_path(relpath: str) -> bool:
    relpath = safe_relpath(relpath)
    if relpath == "mini_vlm_toolkit/agent/constant.py":
        return True
    tools_prefix = "mini_vlm_toolkit/tool_infra/src/tools/"
    if not relpath.startswith(tools_prefix):
        return False
    return relpath.endswith("/tool.py") or relpath.endswith("/process_tool_output.py")


def create_base_snapshot(repo_root: Path, base_repo: Path) -> None:
    if base_repo.exists():
        shutil.rmtree(base_repo)
    base_repo.parent.mkdir(parents=True, exist_ok=True)
    repo_root_resolved = repo_root.resolve()
    base_repo_resolved = base_repo.resolve()
    excludes = list(DEFAULT_RSYNC_EXCLUDES)
    try:
        base_repo_parent_rel = base_repo_resolved.parent.relative_to(repo_root_resolved)
    except ValueError:
        base_repo_parent_rel = None
    if base_repo_parent_rel is not None and base_repo_parent_rel.parts:
        excludes.append(f"{base_repo_parent_rel.as_posix().rstrip('/')}/")
    try:
        base_repo_rel = base_repo_resolved.relative_to(repo_root_resolved)
    except ValueError:
        base_repo_rel = None
    if base_repo_rel is not None and base_repo_rel.parts:
        excludes.append(f"{base_repo_rel.as_posix().rstrip('/')}/")
    command = [
        "rsync",
        "-a",
        "--delete",
        *[f"--exclude={item}" for item in excludes],
        f"{repo_root_resolved}/",
        f"{base_repo_resolved}/",
    ]
    subprocess.run(command, check=True)


def create_initial_state(states_root: Path) -> Path:
    state_dir = states_root / "s0000"
    overlay_dir = state_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        state_dir / "state.json",
        {
            "state_id": "s0000",
            "parent_state_id": None,
            "created_round": 0,
            "creation_index": 0,
            "proposal_id": None,
            "overlay_files": [],
            "patch_summary": "baseline repository snapshot",
            "status": "created",
        },
    )
    return state_dir


def overlay_files(state_dir: Path) -> List[str]:
    overlay_dir = state_dir / "overlay"
    if not overlay_dir.exists():
        return []
    files = [
        path.relative_to(overlay_dir).as_posix()
        for path in overlay_dir.rglob("*")
        if path.is_file()
    ]
    return sorted(files)


def materialize_state(base_repo: Path, state_dir: Path, scratch_repo: Path) -> None:
    if scratch_repo.exists():
        shutil.rmtree(scratch_repo)
    scratch_repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "--delete", f"{base_repo.resolve()}/", f"{scratch_repo.resolve()}/"],
        check=True,
    )
    overlay_dir = state_dir / "overlay"
    if not overlay_dir.exists():
        return
    for source in overlay_dir.rglob("*"):
        if not source.is_file():
            continue
        relpath = source.relative_to(overlay_dir)
        target = scratch_repo / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def delete_scratch_repo(scratch_repo: Path) -> None:
    if scratch_repo.exists():
        shutil.rmtree(scratch_repo)


def _copy_overlay(source_overlay: Path, target_overlay: Path) -> None:
    if not source_overlay.exists():
        return
    for source in source_overlay.rglob("*"):
        if not source.is_file():
            continue
        relpath = source.relative_to(source_overlay)
        target = target_overlay / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _read_text_for_diff(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)


def _parent_file_for(base_repo: Path, parent_state_dir: Path, relpath: str) -> Path:
    overlay_path = parent_state_dir / "overlay" / relpath
    if overlay_path.exists():
        return overlay_path
    return base_repo / relpath


def _proposal_file_items(proposal_dir: Path) -> List[Tuple[str, Path]]:
    files_dir = proposal_dir / "files"
    if not files_dir.exists():
        return []
    items: List[Tuple[str, Path]] = []
    for path in files_dir.rglob("*"):
        if not path.is_file():
            continue
        relpath = path.relative_to(files_dir).as_posix()
        safe_relpath(relpath)
        items.append((relpath, path))
    return sorted(items)


def _read_proposal_metadata(proposal_dir: Path) -> Dict[str, Any]:
    metadata_path = proposal_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = read_json(metadata_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _proposal_id(proposal_dir: Path, metadata: Dict[str, Any]) -> str:
    return str(metadata.get("proposal_id") or proposal_dir.name)


def _changed_ops(before: List[str], after: List[str]) -> List[Dict[str, Any]]:
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    ops: List[Dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        ops.append({
            "start": i1,
            "end": i2,
            "replacement": after[j1:j2],
        })
    return ops


def _ops_conflict(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_start = int(left["start"])
    left_end = int(left["end"])
    right_start = int(right["start"])
    right_end = int(right["end"])

    left_insert = left_start == left_end
    right_insert = right_start == right_end
    if left_insert and right_insert:
        return left_start == right_start
    if left_insert:
        return right_start <= left_start <= right_end
    if right_insert:
        return left_start <= right_start <= left_end
    return max(left_start, right_start) < min(left_end, right_end)


def _format_parent_lines(start: int, end: int) -> str:
    if start == end:
        return f"insert@{start + 1}"
    return f"{start + 1}-{end}"


def _build_combo_files(
    *,
    base_repo: Path,
    parent_state_dir: Path,
    proposal_dirs: Sequence[Path],
) -> Tuple[Dict[str, List[str]], List[str], List[Dict[str, Any]]]:
    changes_by_file: Dict[str, List[Dict[str, Any]]] = {}
    proposal_infos: List[Dict[str, Any]] = []

    for proposal_dir in proposal_dirs:
        metadata = _read_proposal_metadata(proposal_dir)
        proposal_id = _proposal_id(proposal_dir, metadata)
        proposal_items = _proposal_file_items(proposal_dir)
        if not proposal_items:
            raise ValueError(f"proposal has no files: {proposal_dir}")

        edited_files: List[str] = []
        for relpath, source in proposal_items:
            if not is_allowed_edit_path(relpath):
                raise ValueError(f"proposal edits disallowed path: {relpath}")

            parent_file = _parent_file_for(base_repo, parent_state_dir, relpath)
            before = _read_text_for_diff(parent_file)
            after = _read_text_for_diff(source)
            if before == after:
                continue

            ops = _changed_ops(before, after)
            if not ops:
                continue
            edited_files.append(relpath)
            for op in ops:
                op = dict(op)
                op["proposal_id"] = proposal_id
                op["relpath"] = relpath
                changes_by_file.setdefault(relpath, []).append(op)

        if not edited_files:
            raise ValueError(f"proposal does not change any allowed file: {proposal_dir}")

        proposal_infos.append({
            "proposal_dir": proposal_dir,
            "proposal_id": proposal_id,
            "metadata": metadata,
            "edited_files": edited_files,
        })

    for relpath, changes in changes_by_file.items():
        for index, left in enumerate(changes):
            for right in changes[index + 1:]:
                if left["proposal_id"] == right["proposal_id"]:
                    continue
                if _ops_conflict(left, right):
                    left_lines = _format_parent_lines(int(left["start"]), int(left["end"]))
                    right_lines = _format_parent_lines(int(right["start"]), int(right["end"]))
                    raise ProposalConflict(
                        f"{left['proposal_id']} and {right['proposal_id']} both edit "
                        f"{relpath} around parent lines {left_lines} and {right_lines}"
                    )

    merged_files: Dict[str, List[str]] = {}
    for relpath, changes in changes_by_file.items():
        parent_file = _parent_file_for(base_repo, parent_state_dir, relpath)
        merged = list(_read_text_for_diff(parent_file))
        for op in sorted(changes, key=lambda item: (int(item["start"]), int(item["end"])), reverse=True):
            merged[int(op["start"]): int(op["end"])] = list(op["replacement"])
        merged_files[relpath] = merged

    edited_files = sorted(merged_files)
    return merged_files, edited_files, proposal_infos


def _combo_summary(proposal_infos: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for info in proposal_infos:
        metadata = info["metadata"]
        summary = metadata.get("summary") or metadata.get("idea") or info["proposal_id"]
        parts.append(f"{info['proposal_id']}: {summary}")
    if len(parts) == 1:
        return parts[0].split(": ", 1)[1] if ": " in parts[0] else parts[0]
    return "Crossover " + "; ".join(parts)


def create_child_state_from_proposals(
    *,
    base_repo: Path,
    states_root: Path,
    parent_state_dir: Path,
    proposal_dirs: Sequence[Path],
    state_id: str,
    created_round: int,
    creation_index: int,
) -> Path:
    proposal_dirs = list(proposal_dirs)
    if not proposal_dirs:
        raise ValueError("at least one proposal is required")

    merged_files, edited_files, proposal_infos = _build_combo_files(
        base_repo=base_repo,
        parent_state_dir=parent_state_dir,
        proposal_dirs=proposal_dirs,
    )
    if not edited_files:
        raise ValueError("proposal combination does not change any allowed file")

    child_dir = states_root / state_id
    if child_dir.exists():
        shutil.rmtree(child_dir)
    child_overlay = child_dir / "overlay"
    child_overlay.mkdir(parents=True, exist_ok=True)

    _copy_overlay(parent_state_dir / "overlay", child_overlay)

    diff_lines: List[str] = []
    for relpath in edited_files:
        target = child_overlay / relpath
        target.parent.mkdir(parents=True, exist_ok=True)

        parent_file = _parent_file_for(base_repo, parent_state_dir, relpath)
        before = _read_text_for_diff(parent_file)
        after = merged_files[relpath]
        if before == after:
            continue
        diff_lines.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"parent/{relpath}",
                tofile=f"{state_id}/{relpath}",
            )
        )
        target.write_text("".join(after), encoding="utf-8")

    source_proposal_ids = [info["proposal_id"] for info in proposal_infos]
    combo_id = "+".join(source_proposal_ids)

    if len(proposal_dirs) == 1 and (proposal_dirs[0] / "patch.diff").exists():
        shutil.copy2(proposal_dirs[0] / "patch.diff", child_dir / "proposal.patch.diff")
    (child_dir / "patch.diff").write_text("".join(diff_lines), encoding="utf-8")

    parent_state = read_json(parent_state_dir / "state.json")
    state_payload: Dict[str, Any] = {
        "state_id": state_id,
        "parent_state_id": parent_state.get("state_id"),
        "created_round": created_round,
        "creation_index": creation_index,
        "proposal_id": combo_id,
        "source_proposal_ids": source_proposal_ids,
        "combo_size": len(proposal_dirs),
        "is_crossover": len(proposal_dirs) > 1,
        "conflict_status": "evaluated",
        "overlay_files": overlay_files(child_dir),
        "edited_files": edited_files,
        "patch_summary": _combo_summary(proposal_infos),
        "proposal_dir": str(proposal_dirs[0]) if len(proposal_dirs) == 1 else None,
        "proposal_dirs": [str(path) for path in proposal_dirs],
        "status": "created",
    }
    write_json(child_dir / "state.json", state_payload)
    return child_dir


def create_child_state_from_proposal(
    *,
    base_repo: Path,
    states_root: Path,
    parent_state_dir: Path,
    proposal_dir: Path,
    state_id: str,
    created_round: int,
    creation_index: int,
) -> Path:
    return create_child_state_from_proposals(
        base_repo=base_repo,
        states_root=states_root,
        parent_state_dir=parent_state_dir,
        proposal_dirs=[proposal_dir],
        state_id=state_id,
        created_round=created_round,
        creation_index=creation_index,
    )


def update_state(state_dir: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    payload = read_json(state_dir / "state.json")
    payload.update(updates)
    if "overlay_files" not in payload:
        payload["overlay_files"] = overlay_files(state_dir)
    write_json(state_dir / "state.json", payload)
    return payload


def rank_states(state_payloads: Iterable[Dict[str, Any]], pool_size: int) -> List[Dict[str, Any]]:
    viable = [
        state for state in state_payloads
        if state.get("status") == "evaluated" and isinstance(state.get("score"), (int, float))
    ]
    viable.sort(
        key=lambda state: (
            -float(state["score"]),
            int(state.get("creation_index") or 0),
            str(state.get("state_id") or ""),
        )
    )
    return viable[:pool_size]
