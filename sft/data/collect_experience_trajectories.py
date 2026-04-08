#!/usr/bin/env python3
"""Collect successful trajectories across experience-learning trials.

Walks an experience-learning run directory, reads each trial's eval xlsx
to find tasks marked as correct, then locates the matching full_trajectory
JSON files and writes a single consolidated JSONL ready to be fed into
``convert_trajectories.py``.

Usage:
    python -m sft.data.collect_experience_trajectories \\
        --run-dir outputs/experience_learning/.../MMSearch \\
        --output sft/data/experience_boosted_train.jsonl

Deduplication: a task may appear in multiple trials (because we re-run
failed samples with refined experience). We keep the LAST trial in which
the task was marked correct — that trajectory was generated with the most
refined experience prompt and is typically the highest quality.

Note: requires the experience-learning run to have been launched with
``trajectory.save_trajectories: true`` in its reflection config, so that
``full_trajectories/{task_id}.json`` files exist for each trial.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _list_trial_dirs(trials_root: Path) -> List[Path]:
    """Return trial_0, trial_1, ... subdirectories sorted by trial index."""
    if not trials_root.is_dir():
        return []
    trials = []
    for entry in sorted(trials_root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("trial_"):
            continue
        try:
            idx = int(entry.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        trials.append((idx, entry))
    trials.sort(key=lambda x: x[0])
    return [t[1] for t in trials]


def _find_eval_xlsx(trial_dir: Path) -> Optional[Path]:
    """Locate the prediction xlsx inside a trial directory.

    Experience learning runs save the xlsx directly under trial_dir
    (use_dataset_subdir=False).
    """
    candidates = list(trial_dir.glob("*.xlsx"))
    if not candidates:
        return None
    # Prefer the one without a known suffix; first match is usually correct.
    return candidates[0]


def _find_full_trajectory(trial_dir: Path, task_id: Any) -> Optional[Path]:
    """Locate full_trajectories/{task_id}.json inside a trial directory."""
    full_dir = trial_dir / "full_trajectories"
    if not full_dir.is_dir():
        return None
    candidate = full_dir / f"{task_id}.json"
    if candidate.is_file():
        return candidate
    # Also try string form (xlsx may store ints)
    candidate_str = full_dir / f"{str(task_id)}.json"
    if candidate_str.is_file():
        return candidate_str
    return None


def _load_correct_task_ids(xlsx_path: Path) -> List[Any]:
    """Read an eval xlsx and return task_ids where is_correct is True."""
    df = pd.read_excel(xlsx_path)
    if "is_correct" not in df.columns or "index" not in df.columns:
        return []
    correct_mask = df["is_correct"].astype(bool)
    return df.loc[correct_mask, "index"].tolist()


def collect_trajectories(
    run_dir: str,
    output_path: str,
    include_trials: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Collect successful trajectories across trials and write a consolidated JSONL.

    Args:
        run_dir: Path to the dataset-level directory under an experience-learning
            run, e.g. ``outputs/experience_learning/.../MMSearch``.
        output_path: Output JSONL path (one record per line, ready for
            ``convert_trajectories.py``).
        include_trials: Optional list of trial indices to include. If None,
            includes all trials.

    Returns:
        A dict with summary stats: ``{"total_trials", "tasks_per_trial",
        "unique_correct", "written"}``.
    """
    run_dir_path = Path(run_dir)
    trials_root = run_dir_path / "trials"
    trial_dirs = _list_trial_dirs(trials_root)

    if not trial_dirs:
        raise FileNotFoundError(
            f"No trial_* directories found under {trials_root}. "
            "Was this an experience-learning run?"
        )

    # task_id -> (trial_idx, trajectory_record). Later trials overwrite earlier
    # ones for the same task, so we end up with the LAST trial that produced
    # a correct trajectory for each task.
    selected: Dict[Any, Tuple[int, Dict[str, Any]]] = {}
    tasks_per_trial: Dict[int, int] = {}

    for trial_dir in trial_dirs:
        try:
            trial_idx = int(trial_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        if include_trials is not None and trial_idx not in include_trials:
            continue

        xlsx = _find_eval_xlsx(trial_dir)
        if xlsx is None:
            print(f"  [trial {trial_idx}] no xlsx found, skipping", file=sys.stderr)
            tasks_per_trial[trial_idx] = 0
            continue

        try:
            correct_ids = _load_correct_task_ids(xlsx)
        except Exception as exc:
            print(f"  [trial {trial_idx}] failed to read {xlsx}: {exc}", file=sys.stderr)
            tasks_per_trial[trial_idx] = 0
            continue

        loaded_count = 0
        for task_id in correct_ids:
            traj_path = _find_full_trajectory(trial_dir, task_id)
            if traj_path is None:
                continue
            try:
                with open(traj_path, "r", encoding="utf-8") as f:
                    record = json.load(f)
            except Exception as exc:
                print(f"  [trial {trial_idx}] failed to load {traj_path}: {exc}", file=sys.stderr)
                continue
            # Tag with originating trial for downstream debugging
            record.setdefault("_source_trial", trial_idx)
            selected[task_id] = (trial_idx, record)
            loaded_count += 1

        tasks_per_trial[trial_idx] = loaded_count

    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path_p, "w", encoding="utf-8") as f:
        for task_id, (trial_idx, record) in selected.items():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return {
        "total_trials": len(trial_dirs),
        "tasks_per_trial": tasks_per_trial,
        "unique_correct": len(selected),
        "written": written,
        "output_path": str(output_path_p),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect successful trajectories across experience-learning trials."
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Dataset-level directory under an experience-learning run, "
             "e.g. outputs/experience_learning/<model>/<agent>/<run_id>/<dataset_subdir>/MMSearch",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--trials", type=int, nargs="+", default=None,
        help="Optional list of trial indices to include (default: all)",
    )
    args = parser.parse_args()

    stats = collect_trajectories(
        run_dir=args.run_dir,
        output_path=args.output,
        include_trials=args.trials,
    )
    print(f"Trials inspected:        {stats['total_trials']}")
    print(f"Correct loaded per trial:")
    for trial_idx, n in sorted(stats["tasks_per_trial"].items()):
        print(f"  trial {trial_idx}: {n}")
    print(f"Unique correct (deduped): {stats['unique_correct']}")
    print(f"Written:                  {stats['written']}")
    print(f"Output:                   {stats['output_path']}")


if __name__ == "__main__":
    main()
