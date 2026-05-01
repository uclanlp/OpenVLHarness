"""Stage 2 + 3: SAM3 inference + dataset.evaluate.

Standalone runner that bypasses run_benchmark.py's agent loop. For each
example: extract noun phrase, POST to SAM3, write a prediction row in the
shape the dataset evaluator expects, and stash a per-task sidecar of raw
SAM3 output (boxes + scores + masks) plus an agent-compatible
tool_detections sidecar so the LVIS/ODinW eval bridge can recover
real SAM3 confidence scores for AP.

Usage:
    python -m baselines.sam3.run \
        --data LVIS_bbox_val_positive_only CountBenchQA \
        --num-examples 300 --shuffle --seed 42 \
        --num-proc 8 \
        --sam3-server-url http://127.0.0.1:8086 \
        --work-dir outputs/ \
        --mode all
"""

import argparse
import json
import os
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Make `mini_vlm_toolkit` and `baselines` importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from mini_vlm_toolkit.dataset import build_dataset  # noqa: E402
from mini_vlm_toolkit.smp import dump, get_pred_file_format  # noqa: E402
from mini_vlm_toolkit.utils import load_judge_config  # noqa: E402

from baselines.sam3 import io_layout, prompts, formatting  # noqa: E402
from baselines.sam3.client import SAM3Client  # noqa: E402

ERROR_TAG = "[SAM3_ERROR]"


def _filter_dataset(dataset, num_examples, shuffle, seed, logger):
    if num_examples is None:
        return
    n = min(num_examples, len(dataset.data))
    if shuffle:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(dataset.data)), n))
        dataset.data = dataset.data.iloc[indices].reset_index(drop=True)
        logger(f"Sampled {n} examples (seed={seed})")
    else:
        dataset.data = dataset.data.head(n)
        logger(f"Limited to first {n} examples")


def _resolve_image_path(dataset, row) -> str:
    """Materialise image to a local file path the SAM3 server can read.

    `dataset.dump_image` handles the full base64/URL/local-path zoo and is
    what the agent path uses too.
    """
    paths = dataset.dump_image(row)
    if isinstance(paths, str):
        return paths
    if isinstance(paths, (list, tuple)) and paths:
        return paths[0]
    raise ValueError(f"dump_image returned no path for row index={row.get('index')}")


def _format_prediction(dataset_name: str, sam3_response: dict) -> str:
    if prompts.is_grounding(dataset_name):
        return formatting.format_grounding_prediction(sam3_response)
    if prompts.is_counting(dataset_name):
        return formatting.format_counting_prediction(sam3_response)
    # Unknown dataset type — emit JSON of boxes as a safe default and let
    # the user notice in the per_row CSV.
    return formatting.format_grounding_prediction(sam3_response)


def _process_row(row, dataset, dataset_name, client, raw_dir, det_dir):
    """Process one row end-to-end. Returns (index, prediction_text, status)."""
    idx = str(row.get("index"))
    raw_path = os.path.join(raw_dir, f"{idx}.json")
    det_path = os.path.join(det_dir, f"{idx}.json")

    # Resume: replay if both sidecars exist.
    if os.path.isfile(raw_path) and os.path.isfile(det_path):
        try:
            with open(raw_path) as f:
                rec = json.load(f)
            prompt = rec.get("sam3_prompt", "")
            pseudo_response = {
                "pred_boxes": rec.get("pred_boxes_pixel") or [],
                "pred_scores": rec.get("pred_scores") or [],
                "pred_masks": rec.get("pred_masks_rle") or [],
                "orig_img_w": rec.get("orig_img_w"),
                "orig_img_h": rec.get("orig_img_h"),
                "num_masks": rec.get("num_masks", 0),
            }
            prediction = _format_prediction(dataset_name, pseudo_response)
            return idx, prediction, "cached", prompt, None
        except Exception:
            pass  # corrupt cache; redo

    ext = prompts.extract(dataset_name, row)
    if not ext.sam3_prompt:
        # No noun phrase extracted — emit empty/zero prediction.
        empty_response = {"pred_boxes": [], "pred_scores": [], "pred_masks": [], "orig_img_w": 0, "orig_img_h": 0, "num_masks": 0}
        prediction = _format_prediction(dataset_name, empty_response)
        rec = formatting.build_sam3_raw_record(empty_response, "", idx, dataset_name, str(row.get("image_path") or ""))
        rec["extraction_method"] = ext.method
        rec["extraction_note"] = ext.note
        with open(raw_path, "w") as f:
            json.dump(rec, f)
        with open(det_path, "w") as f:
            json.dump([], f)
        return idx, prediction, "no_prompt", "", ext.method

    try:
        image_path = _resolve_image_path(dataset, row)
    except Exception as exc:
        return idx, f"{ERROR_TAG} image-resolve: {exc}", "image_error", ext.sam3_prompt, str(exc)

    try:
        sam3_response = client.segment(prompt=ext.sam3_prompt, image_path=image_path)
    except Exception as exc:
        return idx, f"{ERROR_TAG} sam3: {exc}", "sam3_error", ext.sam3_prompt, str(exc)

    prediction = _format_prediction(dataset_name, sam3_response)

    raw_rec = formatting.build_sam3_raw_record(
        sam3_response, ext.sam3_prompt, idx, dataset_name, image_path,
    )
    raw_rec["extraction_method"] = ext.method
    with open(raw_path, "w") as f:
        json.dump(raw_rec, f)

    det_rec = formatting.build_tool_detections_entry(sam3_response, ext.sam3_prompt)
    with open(det_path, "w") as f:
        json.dump(det_rec, f)

    return idx, prediction, "ok", ext.sam3_prompt, None


def run_one_dataset(
    dataset_name: str,
    work_dir: str,
    run_id: str,
    judge_kwargs: dict,
    *,
    num_examples,
    shuffle,
    seed,
    num_proc,
    tool_base,
    timeout_seconds,
    retries,
    mode,
):
    print(f"\n=== {dataset_name} ===")
    dataset = build_dataset(dataset_name)
    if dataset is None:
        print(f"  ERROR: build_dataset({dataset_name}) returned None")
        return
    _filter_dataset(dataset, num_examples, shuffle, seed, lambda m: print(f"  {m}"))

    pred_fmt = get_pred_file_format()
    pred_path = io_layout.prediction_file(work_dir, run_id, dataset_name, pred_fmt)
    raw_dir = io_layout.sam3_raw_dir(work_dir, run_id, dataset_name)
    det_dir = io_layout.tool_detections_dir(work_dir, run_id, dataset_name)
    ds_dir = os.path.dirname(pred_path)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(ds_dir, exist_ok=True)

    if mode in ("infer", "all"):
        client = SAM3Client(tool_base=tool_base, timeout_seconds=timeout_seconds, retries=retries)
        try:
            client.health_check()
        except Exception as exc:
            raise SystemExit(f"Tool server unreachable / Visual_Grounding_Tool unavailable at {client.tool_base}: {exc}")
        print(f"  Tool server: {client.tool_base} (Visual_Grounding_Tool -> SAM3)")

        rows = list(dataset.data.iterrows())
        results: dict[str, str] = {}
        statuses: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max(1, num_proc)) as pool:
            futures = {
                pool.submit(_process_row, row, dataset, dataset_name, client, raw_dir, det_dir): idx
                for idx, (_, row) in enumerate(rows)
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=dataset_name):
                try:
                    idx, prediction, status, _prompt, _err = fut.result()
                except Exception as exc:
                    traceback.print_exc()
                    raise SystemExit(f"Worker died: {exc}")
                results[idx] = prediction
                statuses[status] = statuses.get(status, 0) + 1

        # Build the prediction file: take dataset.data verbatim and add a
        # `prediction` column. This preserves `index`, `image_path`,
        # `question`, `answer`, plus any benchmark-specific columns the
        # evaluator needs (e.g. `category_id`, `image_width`).
        out_df = dataset.data.copy()
        out_df["prediction"] = out_df["index"].astype(str).map(results).fillna("")
        dump(out_df, pred_path)

        status_str = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
        print(f"  inference: {status_str}; wrote {pred_path}")

    if mode in ("eval", "all"):
        if not os.path.isfile(pred_path):
            # smp.dump may have fallen back to .pkl on some xlsx errors.
            fallback = pred_path.rsplit(".", 1)[0] + ".pkl"
            if os.path.isfile(fallback):
                pred_path = fallback
            else:
                raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        print(f"  evaluating {pred_path}")
        eval_results = dataset.evaluate(pred_path, **judge_kwargs)
        try:
            print(f"  metrics: {eval_results}")
        except Exception:
            pass


def main(argv=None):
    p = argparse.ArgumentParser(description="SAM3 baseline inference + eval")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--mode", default="all", choices=["all", "infer", "eval"])
    p.add_argument("--work-dir", default="./outputs")
    p.add_argument("--run-id", default=None,
                   help="Reuse a specific run_id under {work_dir}/sam3-baseline/. "
                        "If omitted, a new one is minted (or with --resume, the latest is reused).")
    p.add_argument("--run-tag", default="default")
    p.add_argument("--resume", action="store_true",
                   help="Reuse the most recent run dir under {work_dir}/sam3-baseline/")
    p.add_argument("--num-examples", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--tool-base", default=None,
                   help="Tool server base URL (default: $TOOL_BASE or http://127.0.0.1:8102). "
                        "This is the same endpoint the agent uses; the tool server forwards "
                        "Visual_Grounding_Tool calls to the SAM3 backend.")
    p.add_argument("--timeout-seconds", type=int, default=300)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--judge-config", default=None,
                   help="Optional. SAM3 outputs are unambiguous so a judge LLM is not "
                        "needed; pass this only if you want the evaluator's LLM-judge path.")
    args = p.parse_args(argv)

    # Resolve run_id.
    if args.run_id:
        run_id = args.run_id
    elif args.resume or args.mode == "eval":
        latest = io_layout.find_latest_run(args.work_dir)
        if not latest:
            raise SystemExit(f"No existing SAM3 baseline runs under {args.work_dir}/{io_layout.MODEL_NAME}/ to resume")
        run_id = latest
        print(f"[run] resuming run_id={run_id}")
    else:
        run_id = io_layout.make_run_id(tag=args.run_tag)
    print(f"[run] work_dir={args.work_dir} run_id={run_id}")

    judge_kwargs = load_judge_config(args.judge_config) if args.judge_config else {}
    if args.judge_config:
        print(f"[run] judge_kwargs: {judge_kwargs}")

    for name in args.data:
        try:
            run_one_dataset(
                dataset_name=name,
                work_dir=args.work_dir,
                run_id=run_id,
                judge_kwargs=judge_kwargs,
                num_examples=args.num_examples,
                shuffle=args.shuffle,
                seed=args.seed,
                num_proc=args.num_proc,
                tool_base=args.tool_base,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
                mode=args.mode,
            )
        except SystemExit:
            raise
        except Exception as exc:
            traceback.print_exc()
            print(f"  [{name}] FAILED: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
