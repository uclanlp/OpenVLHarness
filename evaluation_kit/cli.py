"""Command-line entry point of the evaluation layer.

`openvlharness-eval` (or `python run_benchmark.py` at the repository root) builds the
agent from an agent-system YAML, runs the selected benchmarks through it and scores
the predictions with the dataset evaluators.
"""
import sys

try:
    import pandas  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
    sys.exit(
        "openvlharness-eval needs the evaluation dependencies. "
        "Install them with: pip install -e '.[eval]'  (missing: %s)" % exc.name
    )

from dotenv import load_dotenv
load_dotenv(".env")

import json
import os
import argparse
from pathlib import Path
import yaml
from openvlharness import bootstrap
from evaluation_kit import dataset as datasets
import datetime
from evaluation_kit.smp import *
from evaluation_kit import infer_image_job
from evaluation_kit.utils import load_judge_config
from evaluation_kit.utils import file_db
from evaluation_kit.utils.eval_injection import merge_eval_into_trajectories

# Matches FAIL_MSG in evaluation_kit.runner; the
# sentinel prediction written when an example fails to produce an answer.
FAIL_MSG = 'Failed to obtain answer via API.'
from evaluation_kit.utils.run_identity import (
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
    from evaluation_kit.utils.run_identity import sanitize_for_path
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


def _saved_evaluation_identity(manifest):
    """Read model and agent names from saved metadata without resolving current profiles."""
    config = manifest.get("config_snapshot_redacted", {}).get("agent", {})
    service = config.get("agent", {}).get("model_service")
    profile = manifest.get("resolved_model_profiles", {}).get(service, {}).get("profile", {})
    model = manifest.get("model_id") or profile.get("model_id")
    agent_class = manifest.get("agent_class") or config.get("agent_name")
    if not isinstance(model, str) or not model.strip() or not isinstance(agent_class, str) or not agent_class:
        raise ValueError("Saved run manifest lacks model/agent identity required for evaluation")
    model_name = model.split("/")[-1]
    if model_name in {"", ".", ".."}:
        raise ValueError("Saved model identity has an invalid output filename")
    return model_name, agent_class


def _saved_prediction_files(pred_root, manifest, datasets, sample_filter):
    """Locate saved prediction files independently of the current PRED_FORMAT setting."""
    model_name, _ = _saved_evaluation_identity(manifest)
    root = Path(_artifact_root_for_sample_filter(pred_root, sample_filter))
    predictions = {}
    for dataset in datasets:
        directory = root / dataset
        preferred = manifest.get("prediction_format")
        preferred_path = directory / f"{model_name}.{preferred}"
        if preferred in {"xlsx", "tsv", "json"} and preferred_path.is_file():
            predictions[dataset] = str(preferred_path)
            continue
        candidates = [directory / f"{model_name}.{extension}" for extension in ("xlsx", "tsv", "json", "pkl")]
        candidates = [path for path in candidates if path.is_file()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one saved prediction file for {dataset} under {directory}; "
                             f"found {len(candidates)}. Check --run-dir and --sample-filter.")
        predictions[dataset] = str(candidates[0])
    return predictions


def _prepare_evaluation_run(args, logger):
    """Select saved outputs without expanding inference settings or constructing an agent.

    An explicit run directory is authoritative. Discovery inspects saved manifests
    under work_dir, optionally filtering by the unexpanded agent config's alias and
    version. Multiple model/agent families require an explicit run directory.
    """
    if args.run_dir:
        path = Path(args.run_dir).resolve()
        manifest = _load_json(path / "run_manifest.json")
        predictions = _saved_prediction_files(path, manifest, args.data, args.sample_filter)
        return str(path), manifest, *_saved_evaluation_identity(manifest), predictions
    selector = parse_yaml_mapping(args.config, label="Agent metadata filter") if args.config else None
    candidates = []
    for manifest_path in Path(args.work_dir).glob("*/*/*/run_manifest.json"):
        try:
            manifest = _load_json(manifest_path)
            config = manifest.get("config_snapshot_redacted", {}).get("agent", {})
            if selector:
                service = selector.get("agent", {}).get("model_service")
                alias = selector.get("services", {}).get("models", {}).get(service, {}).get("alias")
                saved_service = config.get("agent", {}).get("model_service")
                saved_alias = config.get("services", {}).get("models", {}).get(saved_service, {}).get("alias")
                if (manifest.get("agent_config_version") != selector.get("agent_config_version")
                        or config.get("agent_name") != selector.get("agent_name") or saved_alias != alias):
                    continue
            identity = _saved_evaluation_identity(manifest)
            predictions = _saved_prediction_files(manifest_path.parent, manifest, args.data, args.sample_filter)
        except (ValueError, OSError, TypeError) as exc:
            logger.debug("Skipping saved run %s: %s", manifest_path.parent, exc)
            continue
        candidates.append((manifest.get("created_at_utc", ""), str(manifest_path.parent), manifest, identity, predictions))
    if not candidates:
        raise FileNotFoundError("No saved run contains the requested predictions; supply --run-dir")
    if len({item[3] for item in candidates}) != 1:
        raise ValueError("Saved runs contain multiple model/agent identities; select one with --run-dir")
    _, root, manifest, identity, predictions = max(candidates, key=lambda item: (item[0], item[1]))
    return root, manifest, *identity, predictions


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

def main(argv=None):
    """Run inference and/or evaluation for the selected datasets (the `openvlharness-eval` command)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True, help='Names of Datasets')
    parser.add_argument('--config', type=str, default=None, help='Agent-system config for inference (default: configs/agent/qwen3_8b_paper.yaml); optional metadata filter for eval discovery')
    parser.add_argument('--judge-config', type=str, default=None, help='Judge config override; inference defaults to configs/judge.yaml, eval uses the saved judge configuration')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'], help='all: infer and evaluate; infer: generate predictions; eval: evaluate saved outputs without loading an inference agent')
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    parser.add_argument('--num-proc', type=int, default=4, help='Number of processes for API inference')
    parser.add_argument('--save-html', type=str, default='0', help='Generate HTML reports: 0, N, or all')
    parser.add_argument('--save-images', type=str, default='files', choices=['files', 'base64'],
                        help='How to store images in full_trajectories JSONs: "files" copies '
                             'them to a sibling images/ folder (default), "base64" inlines them as data URIs.')
    parser.add_argument('--save-raw-trace', action='store_true',
                        help='Also save raw traces reconstructed from tokens under '
                             '<work_dir>/<dataset>/raw_traces/ for training. '
                             'Supported for image/text datasets with qwen3-vl-8b-instruct only; automatically enables '
                             'return_token_ids in the effective run config. Requires the matching '
                             'Qwen tokenizer and a server that returns token IDs. Export failures '
                             'are recorded per sample; GPT/Kimi are rejected before inference.')
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
                             'sentinel (see runner.infer_image_job).')
    parser.add_argument(
        '--run-dir', type=str, default=None,
        help='Existing run directory. Evaluation uses its saved model identity and predictions '
             'without an inference config or agent. Inference resumes still require matching '
             'configuration/profile identity. Skips automatic run discovery and directory creation.'
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
    args = parser.parse_args(argv)

    # --mode eval always needs an existing run with pred files
    if args.mode == "eval":
        args.resume = True

    logger = get_logger("RUN")

    if args.mode == "eval" and WORLD_SIZE > 1:
        raise ValueError("Evaluation-only mode runs in one process; launch without distributed inference settings")


    if args.mode == "eval":
        pred_root, run_manifest, model_name, agent_class, saved_predictions = _prepare_evaluation_run(args, logger)
        run_id = run_manifest["run_id"]
        agent_config_version = run_manifest["agent_config_version"]
        judge_config = (parse_yaml_mapping(args.judge_config, label="Judge config") if args.judge_config
                        else run_manifest["config_snapshot_redacted"]["judge"])
        judge_config_version = require_config_version(judge_config, field_name="judge_config_version", label="Judge config")
        judge_kwargs = load_judge_config(judge_config)
        # Local matching policies do not construct an API judge.
        if (judge_kwargs.get("model") not in (None, "exact_matching", "extract_matching")
                and judge_kwargs.get("key") == "<REDACTED>"):
            raise ValueError("Saved judge credentials are redacted; supply --judge-config for the evaluator's judge")
        logger.info("Evaluating saved %s/%s outputs without an inference agent", agent_class, model_name)
        logger.info("LLM Judge Configurations: %s", redact_sensitive_fields(judge_kwargs))
    else:
        args.config = args.config or "configs/agent/qwen3_8b_paper.yaml"
        args.judge_config = args.judge_config or "configs/judge.yaml"
        agent_config = bootstrap.load_agent_config(args.config, save_raw_trace=args.save_raw_trace)
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
        agent = bootstrap.build_agent(agent_config)
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
                dataset = datasets.build_dataset(dataset_name, **dataset_kwargs)
            dist.barrier()

        dataset = datasets.build_dataset(dataset_name, **dataset_kwargs)
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
        result_file = (saved_predictions[dataset_name] if args.mode == "eval" else
                       get_pred_file_path(artifact_root, model_name, dataset_name, use_env_format=True))
        if dataset is None:
            logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
            continue

        if args.mode != "eval":
            logger.info(f"Start inference on dataset {dataset_name} with model {model_name}")
            # Perform the Inference
            infer_image_job(
                agent,
                artifact_root,
                model_name,
                dataset,
                api_nproc=args.num_proc,
                html_limit=html_limit,
                enable_incremental_json=args.incremental_json,
                save_images=args.save_images,
                save_raw_trace=args.save_raw_trace,
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


if __name__ == "__main__":
    main()
