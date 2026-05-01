"""Stage 1: dump the SAM3 prompt that would be sent for every example.

No SAM3 calls. Writes one CSV per dataset under
{work_dir}/sam3-baseline/{run_id}/{dataset}/sam3_prompts.csv with columns
(index, image_path, question, sam3_prompt, extraction_method, note) so you
can audit noun-phrase extraction quality before launching real inference.
"""

import argparse
import os
import random
import sys

# Allow running as `python -m baselines.sam3.preview` or
# `python baselines/sam3/preview.py` from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

import pandas as pd  # noqa: E402

from mini_vlm_toolkit.dataset import build_dataset  # noqa: E402
from baselines.sam3 import io_layout, prompts  # noqa: E402


def _filter_dataset(dataset, num_examples, shuffle, seed):
    if num_examples is None:
        return
    n = min(num_examples, len(dataset.data))
    if shuffle:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(dataset.data)), n))
        dataset.data = dataset.data.iloc[indices].reset_index(drop=True)
    else:
        dataset.data = dataset.data.head(n)


def preview_one(dataset_name: str, work_dir: str, run_id: str, num_examples, shuffle, seed) -> dict:
    dataset = build_dataset(dataset_name)
    if dataset is None:
        return {"dataset": dataset_name, "error": "build_dataset returned None"}
    _filter_dataset(dataset, num_examples, shuffle, seed)

    out_rows = []
    method_counts: dict[str, int] = {}
    warn = 0
    for _, row in dataset.data.iterrows():
        ext = prompts.extract(dataset_name, row)
        method_counts[ext.method] = method_counts.get(ext.method, 0) + 1
        if ext.is_warn:
            warn += 1
        out_rows.append({
            "index": row.get("index"),
            "image_path": row.get("image_path"),
            "category_name": row.get("category_name", ""),
            "question": row.get("question", ""),
            "sam3_prompt": ext.sam3_prompt,
            "extraction_method": ext.method,
            "note": ext.note,
        })

    out_path = io_layout.prompts_csv(work_dir, run_id, dataset_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)

    return {
        "dataset": dataset_name,
        "rows": len(out_rows),
        "warnings": warn,
        "methods": method_counts,
        "out_path": out_path,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Preview SAM3 prompts (no model calls)")
    p.add_argument("--data", nargs="+", required=True, help="Dataset names (e.g. CountBenchQA FSC147_val LVIS_bbox_val_positive_only)")
    p.add_argument("--num-examples", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default="./outputs")
    p.add_argument("--run-id", default=None, help="Reuse a specific run_id; otherwise mint a new one")
    p.add_argument("--run-tag", default="preview")
    args = p.parse_args(argv)

    run_id = args.run_id or io_layout.make_run_id(tag=args.run_tag)
    print(f"[preview] work_dir={args.work_dir} run_id={run_id}")

    summaries = []
    for name in args.data:
        try:
            s = preview_one(name, args.work_dir, run_id, args.num_examples, args.shuffle, args.seed)
        except Exception as exc:
            s = {"dataset": name, "error": str(exc)}
        summaries.append(s)
        if "error" in s:
            print(f"  [{name}] ERROR: {s['error']}")
            continue
        method_str = ", ".join(f"{k}={v}" for k, v in sorted(s["methods"].items()))
        warn_tag = f" {s['warnings']} WARN" if s["warnings"] else ""
        print(f"  [{name}] {s['rows']} rows; methods: {method_str};{warn_tag} -> {s['out_path']}")

    return 0 if all("error" not in s for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
