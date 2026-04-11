from dotenv import load_dotenv
load_dotenv(".env")

import json
import os
import argparse
import yaml
from mini_vlm_toolkit.agent import build_agent, load_agent_config
from mini_vlm_toolkit.dataset import build_dataset
import datetime
from mini_vlm_toolkit.smp import *
from mini_vlm_toolkit import infer_video_job, infer_image_job
from mini_vlm_toolkit.utils import load_judge_config
from mini_vlm_toolkit.utils.run_identity import (
    parse_yaml_mapping,
    require_config_version,
    build_run_id,
    build_run_manifest,
    validate_manifest_compatibility,
)

RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE",1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK",1))


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True, allow_unicode=True)


def _find_resumable_run_dir(parent_dir, agent_ver, judge_ver, expected_manifest, dataset_names, model_name, logger, mode="all"):
    """Search backwards (newest→oldest) for a config-matching run directory.

    When *mode* is ``"eval"``, find the newest run that **has** pred files for
    the requested datasets (ready for evaluation).  Otherwise, find the newest
    run that still has unfinished datasets (ready for inference).
    """
    from mini_vlm_toolkit.utils.run_identity import sanitize_for_path
    agent_part = sanitize_for_path(agent_ver, field_name="agent_config_version")
    judge_part = sanitize_for_path(judge_ver, field_name="judge_config_version")
    prefix = f"agent_{agent_part}_judge_{judge_part}_"

    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(
            f"No run directories found under {parent_dir}. Nothing to resume."
        )

    candidates = sorted(
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith(prefix)
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directories matching prefix '{prefix}' found in {parent_dir}. "
            "Nothing to resume."
        )

    for dirname in reversed(candidates):
        pred_root = os.path.join(parent_dir, dirname)
        manifest_path = os.path.join(pred_root, "run_manifest.json")

        # Skip dirs without a manifest
        if not os.path.exists(manifest_path):
            logger.debug(f"Skipping {dirname}: no run_manifest.json")
            continue

        # Check config compatibility
        try:
            existing_manifest = _load_json(manifest_path)
            validate_manifest_compatibility(existing_manifest, expected_manifest)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.debug(f"Skipping {dirname}: {exc}")
            continue

        all_finished = all(
            os.path.exists(get_pred_file_path(pred_root, model_name, ds, use_env_format=True))
            for ds in dataset_names
        )

        if mode == "eval":
            # For eval-only: require that all pred files exist (ready to evaluate)
            if not all_finished:
                logger.debug(f"Skipping {dirname}: not all pred files exist for eval")
                continue
        else:
            # For infer/all: skip runs where everything is already done
            if all_finished:
                logger.debug(f"Skipping {dirname}: all datasets already finished")
                continue

        return pred_root

    raise FileNotFoundError(
        f"No resumable run found in {parent_dir} matching prefix '{prefix}'. "
        "All matching runs either have incompatible configs or are already finished."
    )


# Keys that should be redacted in tool env snapshots (API keys, secrets).
_SENSITIVE_ENV_PATTERNS = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
)


def _snapshot_tool_env(pred_root, logger, agent_config=None):
    """Save tool env snapshot and warn if .env file diverges from running server."""
    import requests as _requests

    tool_base = None
    if agent_config and isinstance(agent_config, dict):
        tool_base = (agent_config.get("agent_kwargs") or {}).get("tool_base")
    tool_base = tool_base or "http://localhost:8102"

    # 1. Read from running server
    server_env = None
    try:
        resp = _requests.get(f"{tool_base}/env", timeout=5)
        resp.raise_for_status()
        server_env = resp.json()
    except Exception:
        pass

    # 2. Read from .env file on disk
    file_env = {}
    tool_infra_dir = os.path.join(os.path.dirname(__file__), "mini_vlm_toolkit", "tool_infra")
    for name in [".env.rui", ".env"]:
        env_path = os.path.join(tool_infra_dir, name)
        if os.path.isfile(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    file_env[k.strip()] = v.strip()
            break

    # 3. Compare and warn on mismatch
    mismatches = {}
    if server_env and file_env:
        for key in server_env:
            if key.startswith("_"):
                continue
            file_val = file_env.get(key)
            if file_val is not None and file_val != server_env[key]:
                mismatches[key] = {"server": server_env[key], "file": file_val}
                msg = (
                    f"⚠ Tool env MISMATCH: {key} — "
                    f"server='{server_env[key]}' vs file='{file_val}'. "
                    f"Did you edit .env without restarting the tool server?"
                )
                logger.warning(msg)
                print(f"\n{'='*60}\nWARNING: {msg}\n{'='*60}\n")

    # 4. Save server env (ground truth) or file env (fallback)
    snapshot = server_env or {
        k: ("<REDACTED>" if any(p in k.upper() for p in _SENSITIVE_ENV_PATTERNS) else v)
        for k, v in file_env.items()
    }
    if snapshot:
        if mismatches:
            snapshot["_MISMATCHES"] = mismatches
        snapshot_path = os.path.join(pred_root, "tool_env.snapshot.json")
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved tool env snapshot: {snapshot_path}")


def _prepare_run_root(pred_root, manifest, logger):
    run_manifest_path = os.path.join(pred_root, "run_manifest.json")
    agent_cfg_path = os.path.join(pred_root, "agent_config.redacted.yaml")
    judge_cfg_path = os.path.join(pred_root, "judge_config.redacted.yaml")

    if os.path.exists(pred_root):
        if not os.path.exists(run_manifest_path):
            raise ValueError(
                f"Run directory exists without manifest: {pred_root}. "
                "Please bump 'agent_config_version' and/or 'judge_config_version' "
                "to start a new run id."
            )
        existing_manifest = _load_json(run_manifest_path)
        validate_manifest_compatibility(existing_manifest, manifest)
        logger.info(
            f"Reusing existing run root with matching identity: {pred_root}"
        )
        if not os.path.exists(agent_cfg_path):
            _write_yaml(agent_cfg_path, manifest["config_snapshot_redacted"]["agent"])
        if not os.path.exists(judge_cfg_path):
            _write_yaml(judge_cfg_path, manifest["config_snapshot_redacted"]["judge"])
        return

    os.makedirs(pred_root, exist_ok=True)
    _write_json(run_manifest_path, manifest)
    _write_yaml(agent_cfg_path, manifest["config_snapshot_redacted"]["agent"])
    _write_yaml(judge_cfg_path, manifest["config_snapshot_redacted"]["judge"])

    # Snapshot tool_infra env for reproducibility (search tool settings, API config, etc.)
    _snapshot_tool_env(pred_root, logger, agent_config=manifest.get("config_snapshot_redacted", {}).get("agent"))

    logger.info(f"Initialized new run root: {pred_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--config', type=str, default='configs/base_agent.yaml', help='Path to agent config file')
    parser.add_argument('--judge-config', type=str, default='configs/judge.yaml', help='Path to judge config file')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'], help='Select running mode')
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    parser.add_argument('--num-proc', type=int, default=4, help='Number of processes for API inference')
    parser.add_argument('--save-html', type=str, default='0', help='Generate HTML reports: 0, N, or all')
    parser.add_argument('--save-images', type=str, default='files', choices=['files', 'base64'],
                        help='How to store images in full_trajectories JSONs: "files" copies '
                             'them to a sibling images/ folder (default), "base64" inlines them as data URIs.')
    parser.add_argument('--incremental-json', action='store_true', help='Enable incremental JSONL output (append results as they complete)')
    parser.add_argument('--num-examples', type=int, default=None, help='Only run on the first N examples (ignores cache for those examples)')
    parser.add_argument('--shuffle', action='store_true', help='Randomly sample --num-examples instances instead of taking the first N')
    parser.add_argument(
        '--sample-filter', type=str, default='all',
        choices=['all', 'image_only', 'text_only'],
        help='Filter samples by modality: all (default), image_only (has image), text_only (no image)',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for --shuffle (default: 42)')
    parser.add_argument('--run-tag', type=str, default='default', help='Custom tag appended to run_id (default: "default")')
    parser.add_argument('--resume', action='store_true', help='Resume the latest existing run instead of creating a new one')
    parser.add_argument(
        '--run-dir', type=str, default=None,
        help='Path to an existing run directory. When set, all current behavior is bypassed: '
             'no new run dir is created, --resume scanning is skipped, and result files are '
             'read/written directly under this directory. Useful for re-evaluating a completed run '
             '(e.g. with --mode eval to apply LLM judge to an existing inference output).'
    )
    args = parser.parse_args()

    # --mode eval always needs an existing run with pred files
    if args.mode == "eval":
        args.resume = True

    logger = get_logger("RUN")

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    agent_config = load_agent_config(args.config)
    judge_config = parse_yaml_mapping(args.judge_config, label="Judge config")
    agent_config_version = require_config_version(
        agent_config,
        field_name="agent_config_version",
        label="Agent config",
    )
    judge_config_version = require_config_version(
        judge_config,
        field_name="judge_config_version",
        label="Judge config",
    )
    agent = build_agent(args.config)
    judge_kwargs = load_judge_config(args.judge_config)
    model_name = agent.model.split('/')[-1]
    agent_class = agent.__class__.__name__

    logger.info(f"LLM Judge Configurations: {judge_kwargs}")
    logger.info(f"Initialized agent of class {agent_class} with model {model_name}")

    parent_dir = os.path.join(args.work_dir, model_name, agent_class.lower())

    if args.run_dir:
        # Use the specified directory directly. Skip resume scanning and
        # new-dir creation. This is the path for re-evaluating completed runs.
        pred_root = os.path.abspath(args.run_dir)
        if not os.path.isdir(pred_root):
            raise FileNotFoundError(f"--run-dir does not exist: {pred_root}")
        run_id = os.path.basename(pred_root)
        run_manifest = build_run_manifest(
            run_id=run_id,
            agent_config_version=agent_config_version,
            judge_config_version=judge_config_version,
            agent_config=agent_config,
            judge_config=judge_config,
            config_path=args.config,
            judge_config_path=args.judge_config,
        )
        logger.info(f"Using specified run directory: {pred_root}")
    elif args.resume:
        # Build a temporary manifest for compatibility checking (run_id filled after finding dir)
        tmp_manifest = build_run_manifest(
            run_id="__resume_pending__",
            agent_config_version=agent_config_version,
            judge_config_version=judge_config_version,
            agent_config=agent_config,
            judge_config=judge_config,
            config_path=args.config,
            judge_config_path=args.judge_config,
        )
        pred_root = _find_resumable_run_dir(
            parent_dir, agent_config_version, judge_config_version,
            tmp_manifest, args.data, model_name, logger, mode=args.mode,
        )
        run_id = os.path.basename(pred_root)
        run_manifest = build_run_manifest(
            run_id=run_id,
            agent_config_version=agent_config_version,
            judge_config_version=judge_config_version,
            agent_config=agent_config,
            judge_config=judge_config,
            config_path=args.config,
            judge_config_path=args.judge_config,
        )
        logger.info(f"Resuming existing run: {pred_root}")
    else:
        run_id = build_run_id(agent_config_version, judge_config_version, run_tag=args.run_tag)
        pred_root = os.path.join(parent_dir, run_id)
        run_manifest = build_run_manifest(
            run_id=run_id,
            agent_config_version=agent_config_version,
            judge_config_version=judge_config_version,
            agent_config=agent_config,
            judge_config=judge_config,
            config_path=args.config,
            judge_config_path=args.judge_config,
        )
        if RANK == 0:
            _prepare_run_root(pred_root, run_manifest, logger)
        if WORLD_SIZE > 1:
            dist.barrier()

    logger.info(
        f"Run identity: run_id={run_id} "
        f"(agent_config_version={agent_config_version}, judge_config_version={judge_config_version})"
    )

    html_arg = str(args.save_html).strip().lower()
    if html_arg == 'all':
        html_limit = -1
    else:
        html_limit = int(html_arg)

    for _, dataset_name in enumerate(args.data):
        dataset_kwargs = dict()
        if WORLD_SIZE > 1:
            if RANK == 0:
                dataset = build_dataset(dataset_name, **dataset_kwargs)
            dist.barrier()

        dataset = build_dataset(dataset_name, **dataset_kwargs)
        if dataset is not None and args.num_examples is not None:
            if args.shuffle:
                import random
                rng = random.Random(args.seed)
                n = min(args.num_examples, len(dataset.data))
                indices = sorted(rng.sample(range(len(dataset.data)), n))
                dataset.data = dataset.data.iloc[indices].reset_index(drop=True)
                logger.info(f"Randomly sampled {n} examples from {dataset_name} (seed={args.seed})")
            else:
                dataset.data = dataset.data.head(args.num_examples)
                logger.info(f"Limiting dataset {dataset_name} to first {args.num_examples} examples")
        # --- Sample modality filter ---
        if dataset is not None and args.sample_filter != "all":
            if "image_path" in dataset.data.columns:
                before = len(dataset.data)
                if args.sample_filter == "image_only":
                    mask = dataset.data["image_path"].apply(
                        lambda v: bool(v) and v != "[]" and v != [] and str(v).strip() not in ("", "nan")
                    )
                else:  # text_only
                    mask = dataset.data["image_path"].apply(
                        lambda v: not v or v == "[]" or v == [] or str(v).strip() in ("", "nan")
                    )
                dataset.data = dataset.data[mask].reset_index(drop=True)
                logger.info(
                    f"Sample filter '{args.sample_filter}': {before} -> {len(dataset.data)} samples in {dataset_name}"
                )

        result_file = get_pred_file_path(pred_root, model_name, dataset_name, use_env_format=True)
        if dataset is None:
            logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
            continue

        if args.mode != "eval":
            logger.info(f"Start inference on dataset {dataset_name} with model {model_name}")
            # Perform the Inference
            if dataset.MODALITY == 'VIDEO':
                infer_video_job(
                    agent,
                    pred_root,
                    model_name,
                    dataset,
                    dataset_name,
                    api_nproc=args.num_proc,
                    html_limit=html_limit,
                    enable_incremental_json=args.incremental_json,
                )
            else:
                infer_image_job(
                    agent,
                    pred_root,
                    model_name,
                    dataset,
                    api_nproc=args.num_proc,
                    html_limit=html_limit,
                    enable_incremental_json=args.incremental_json,
                    save_images=args.save_images,
                )

        
        if args.mode != "infer":
            # Some writers (e.g., xlsx) may fail and dump() falls back to .pkl.
            # Ensure evaluation reads the actual existing result file.
            if not os.path.exists(result_file):
                fallback_pkl = result_file.rsplit('.', 1)[0] + '.pkl'
                if os.path.exists(fallback_pkl):
                    logger.warning(
                        f"Prediction file not found: {result_file}. "
                        f"Using fallback file: {fallback_pkl}"
                    )
                    result_file = fallback_pkl
            logger.info(f"Start evaluation on dataset {dataset_name} with model {model_name}")
            # Perform the Evaluation
            eval_results = dataset.evaluate(result_file, **judge_kwargs)
