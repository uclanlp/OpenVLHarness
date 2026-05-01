"""Single source of truth for SAM3 baseline output paths."""

import os
import datetime

MODEL_NAME = "sam3-baseline"
RUN_PREFIX = "sam3_baseline"


def make_run_id(tag: str = "default") -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{RUN_PREFIX}_{stamp}_{tag}"


def run_root(work_dir: str, run_id: str) -> str:
    return os.path.join(work_dir, MODEL_NAME, run_id)


def dataset_dir(work_dir: str, run_id: str, dataset_name: str) -> str:
    return os.path.join(run_root(work_dir, run_id), dataset_name)


def prediction_file(work_dir: str, run_id: str, dataset_name: str, fmt: str = "xlsx") -> str:
    return os.path.join(dataset_dir(work_dir, run_id, dataset_name), f"{MODEL_NAME}.{fmt}")


def prompts_csv(work_dir: str, run_id: str, dataset_name: str) -> str:
    return os.path.join(dataset_dir(work_dir, run_id, dataset_name), "sam3_prompts.csv")


def sam3_raw_dir(work_dir: str, run_id: str, dataset_name: str) -> str:
    return os.path.join(dataset_dir(work_dir, run_id, dataset_name), "sam3_raw")


def tool_detections_dir(work_dir: str, run_id: str, dataset_name: str) -> str:
    return os.path.join(dataset_dir(work_dir, run_id, dataset_name), "tool_detections")


def find_latest_run(work_dir: str) -> str | None:
    parent = os.path.join(work_dir, MODEL_NAME)
    if not os.path.isdir(parent):
        return None
    candidates = sorted(
        d for d in os.listdir(parent)
        if d.startswith(RUN_PREFIX) and os.path.isdir(os.path.join(parent, d))
    )
    return candidates[-1] if candidates else None
