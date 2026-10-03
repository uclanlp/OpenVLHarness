from dotenv import load_dotenv
load_dotenv(".env")

import json
import os
import argparse
import yaml
from openvlharness.bootstrap import build_agent, load_agent_config
from mini_vlm_toolkit.dataset import build_dataset
import datetime
from mini_vlm_toolkit.smp import *
from mini_vlm_toolkit import infer_video_job, infer_image_job
from mini_vlm_toolkit.utils import load_judge_config
from mini_vlm_toolkit.utils import file_db
from mini_vlm_toolkit.utils.eval_injection import merge_eval_into_trajectories

# Matches FAIL_MSG in mini_vlm_toolkit.inference_image / .inference_video; the
# sentinel prediction written when an example fails to produce an answer.
FAIL_MSG = 'Failed to obtain answer via API.'
from mini_vlm_toolkit.utils.run_identity import (
    parse_yaml_mapping,
    require_config_version,
    build_run_id,
    build_run_manifest,
    validate_manifest_compatibility,
    redact_sensitive_fields,
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


def _artifact_root_for_sample_filter(pred_root, sample_filter):
    if sample_filter == "all":
        return pred_root
    return os.path.join(pred_root, f"sample_filter={sample_filter}")


def _find_resumable_run_dir(
    parent_dir,
    agent_ver,
    judge_ver,
    expected_manifest,
    dataset_names,
    model_name,
    logger,
    mode="all",
    sample_filter="all",
):
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

        artifact_root = _artifact_root_for_sample_filter(pred_root, sample_filter)
        all_finished = all(
            os.path.exists(get_pred_file_path(artifact_root, model_name, ds, use_env_format=True))
            for ds in dataset_names
        )

        if mode == "eval":
            # For eval-only: require that all pred files exist (ready to evaluate)
            if not all_finished:
                logger.debug(
                    f"Skipping {dirname}: not all pred files exist for eval "
                    f"(sample_filter={sample_filter})"
                )
                continue
        else:
            # For infer/all: skip runs where everything is already done
            if all_finished:
                logger.debug(
                    f"Skipping {dirname}: all datasets already finished "
                    f"(sample_filter={sample_filter})"
                )
                continue

        return pred_root

    raise FileNotFoundError(
        f"No resumable run found in {parent_dir} matching prefix '{prefix}'. "
        "All matching runs either have incompatible configs or are already finished."
    )


def _snapshot_tool_env(pred_root, logger, agent_config=None):
    """Record the selected router and resolved service configuration, without secrets."""
    import requests as _requests
    services = (agent_config or {}).get("services", {})
    base_url = services.get("client", {}).get("base_url")
    snapshot = {"configured": redact_sensitive_fields(services), "server": None}
    try:
        response = _requests.get(f"{base_url}/env", timeout=5)
        response.raise_for_status()
        snapshot["server"] = redact_sensitive_fields(response.json())
    except Exception as exc:
        snapshot["server_error"] = type(exc).__name__
        logger.warning("Router environment snapshot unavailable: %s", type(exc).__name__)
    _write_json(os.path.join(pred_root, "tool_env_snapshot.json"), snapshot)


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
    parser.add_argument('--config', type=str, default='configs/agent/qwen3_paper.yaml', help='Path to agent-system config file')
    parser.add_argument('--judge-config', type=str, default='configs/judge.yaml', help='Path to judge config file')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'], help='Select running mode')
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    parser.add_argument('--num-proc', type=int, default=4, help='Number of processes for API inference')
    parser.add_argument('--save-html', type=str, default='0', help='Generate HTML reports: 0, N, or all')
    parser.add_argument('--save-images', type=str, default='files', choices=['files', 'base64'],
                        help='How to store images in full_trajectories JSONs: "files" copies '
                             'them to a sibling images/ folder (default), "base64" inlines them as data URIs.')
    parser.add_argument('--save-raw-trajectory', action='store_true',
                        help='Also save raw tokenized trajectories under '
                             '<work_dir>/<dataset>/raw_trajectories/ for SFT training. '
                             'Requires the agent config to request '
                             'extra_body.return_token_ids=true on a vLLM-compatible server.')
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
    parser.add_argument('--ignore-failed', action='store_true',
                        help='With --run-dir on a run that already has a prediction file: keep every '
                             'non-failed row and re-run only rows whose prediction is the harness failure '
                             'sentinel (see inference_image.infer_image_job).')
    parser.add_argument(
        '--run-dir', type=str, default=None,
        help='Path to an existing run directory. When set, all current behavior is bypassed: '
             'no new run dir is created, --resume scanning is skipped, and result files are '
             'read/written directly under this directory. Useful for re-evaluating a completed run '
             '(e.g. with --mode eval to apply LLM judge to an existing inference output).'
    )
    parser.add_argument(
        '--keep-file-db', action='store_true',
        help='Keep the per-example file-db (results/, _done/, _claims/) after a '
             'fully-evaluated run. By default (--mode all only), once inference + '
             'evaluation complete and the aggregated prediction file is written, '
             'these are deleted to reclaim disk; the xlsx and full_trajectories/ '
             'are kept and remain sufficient for eval, error analysis, and the '
             'visualizer. Cleanup is also skipped when failures remain (retryable '
             'via --ignore-failed) or a concurrent worker is still active. Set the '
             'env var MVTK_KEEP_FILE_DB=1 to disable globally.'
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

    logger.info(f"LLM Judge Configurations: {redact_sensitive_fields(judge_kwargs)}")
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
        run_manifest_path = os.path.join(pred_root, "run_manifest.json")
        if not os.path.exists(run_manifest_path):
            raise ValueError(f"--run-dir missing run_manifest.json: {pred_root}")
        existing_manifest = _load_json(run_manifest_path)
        validate_manifest_compatibility(existing_manifest, run_manifest)
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
            tmp_manifest, args.data, model_name, logger,
            mode=args.mode,
            sample_filter=args.sample_filter,
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

        artifact_root = _artifact_root_for_sample_filter(pred_root, args.sample_filter)
        if args.sample_filter != "all":
            logger.info(
                f"Using filtered artifact root for {dataset_name}: {artifact_root}"
            )
        result_file = get_pred_file_path(artifact_root, model_name, dataset_name, use_env_format=True)
        if dataset is None:
            logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
            continue

        if args.mode != "eval":
            logger.info(f"Start inference on dataset {dataset_name} with model {model_name}")
            # Perform the Inference
            if dataset.MODALITY == 'VIDEO':
                infer_video_job(
                    agent,
                    artifact_root,
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
                    artifact_root,
                    model_name,
                    dataset,
                    api_nproc=args.num_proc,
                    html_limit=html_limit,
                    enable_incremental_json=args.incremental_json,
                    save_images=args.save_images,
                    save_raw_trajectory=args.save_raw_trajectory,
                    ignore_failed=args.ignore_failed,
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
                elif args.sample_filter != "all":
                    raise FileNotFoundError(
                        f"Filtered prediction file not found for dataset {dataset_name}: {result_file}. "
                        f"Expected artifacts under sample_filter={args.sample_filter}. "
                        "Run inference for that filtered slice first, or evaluate the full run with --sample-filter all."
                    )
            logger.info(f"Start evaluation on dataset {dataset_name} with model {model_name}")
            # Perform the Evaluation
            eval_results = dataset.evaluate(result_file, **judge_kwargs)

            # Inject per-instance eval results into full_trajectories JSONs so
            # each <task_id>.json carries its own correct / score / raw block
            # for debugging and visualization. No-op if no adapter is
            # registered for this dataset, or if full_trajectories/ is absent.
            try:
                merge_eval_into_trajectories(
                    work_dir=os.path.dirname(result_file),
                    dataset_name=dataset_name,
                    dataset_obj=dataset,
                    result_file=result_file,
                    judge_kwargs=judge_kwargs,
                )
            except Exception as exc:
                logger.warning(
                    f"[eval_injection] failed for dataset {dataset_name}: {exc}"
                )

            # Auto-cleanup the per-example file-db once a run is fully
            # evaluated. Only on --mode all (inference + eval both ran this
            # invocation) so eval-only / infer-only never delete it. Never
            # fatal: the xlsx + full_trajectories are the durable outputs.
            keep_db = args.keep_file_db or os.getenv("MVTK_KEEP_FILE_DB") in ("1", "true", "True")
            if args.mode == "all" and not keep_db:
                try:
                    db_work_dir = os.path.dirname(result_file)
                    try:
                        pred_data = load(result_file)
                        if isinstance(pred_data, list):
                            import pandas as pd
                            pred_data = pd.DataFrame(pred_data)
                        preds = pred_data.get("prediction", [])
                        has_failures = any(FAIL_MSG in str(p) for p in preds)
                        expected_n = len(dataset.data)
                    except Exception as exc:
                        # If we can't introspect the result file, err on the
                        # side of keeping the db.
                        has_failures, expected_n = True, 0
                        logger.warning(
                            f"[file-db cleanup] could not read {result_file}: {exc}; "
                            "preserving db"
                        )
                    ok, reason = file_db.is_safe_to_cleanup(
                        db_work_dir,
                        result_file_exists=os.path.exists(result_file),
                        expected_n=expected_n,
                        has_failures=has_failures,
                    )
                    if ok:
                        summary = file_db.cleanup(db_work_dir)
                        logger.info(
                            f"[file-db cleanup] {dataset_name}: removed "
                            f"results/_done/_claims, freed "
                            f"{summary['freed_bytes'] / 1e9:.2f} GB"
                        )
                    else:
                        logger.info(
                            f"[file-db cleanup] skipped for {dataset_name}: {reason}"
                        )
                except Exception as exc:
                    logger.warning(
                        f"[file-db cleanup] failed for dataset {dataset_name}: {exc}"
                    )
