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
    args = parser.parse_args()

    logger = get_logger("RUN")
        dist.init_process_group(
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
    run_id = build_run_id(agent_config_version, judge_config_version)

    agent = build_agent(args.config)
    judge_kwargs = load_judge_config(args.judge_config)
    model_name = agent.model.split('/')[-1]
    agent_class = agent.__class__.__name__

    logger.info(f"LLM Judge Configurations: {judge_kwargs}")
    logger.info(f"Initialized agent of class {agent_class} with model {model_name}")
    logger.info(
        f"Run identity: run_id={run_id} "
        f"(agent_config_version={agent_config_version}, judge_config_version={judge_config_version})"
    )
    
    pred_root = os.path.join(args.work_dir, model_name, agent_class.lower(), run_id)
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

    html_arg = str(args.save_html).strip().lower()
    if html_arg == 'all':
        html_limit = -1
    else:
        html_limit = int(html_arg)

    for _, dataset_name in enumerate(args.data):
        if args.mode == "eval":
            # Eval-only: use latest run dir (model_timestamp) under dataset_name
            dataset_root = os.path.join(base_pred_root, dataset_name)
            if os.path.isdir(dataset_root):
                run_dirs = [d for d in os.listdir(dataset_root)
                            if os.path.isdir(os.path.join(dataset_root, d))]
                run_dirs.sort(reverse=True)
                pred_root = os.path.join(dataset_root, run_dirs[0]) if run_dirs else None
            else:
                pred_root = None
        elif args.mode != "eval" and args.resume:
            # Resume: use latest run dir if exists, else create new
            dataset_root = os.path.join(base_pred_root, dataset_name)
            if os.path.isdir(dataset_root):
                run_dirs = [d for d in os.listdir(dataset_root)
                            if os.path.isdir(os.path.join(dataset_root, d))]
                run_dirs.sort(reverse=True)
                pred_root = os.path.join(dataset_root, run_dirs[0]) if run_dirs else None
            else:
                pred_root = None
            if pred_root is not None:
                logger.info(f"Resume mode: using existing run dir {pred_root}")
            else:
                run_dir = f'{model_name}_{run_timestamp}'
                pred_root = os.path.join(base_pred_root, dataset_name, run_dir)
        else:
            run_dir = f'{model_name}_{run_timestamp}'
            pred_root = os.path.join(base_pred_root, dataset_name, run_dir)
        if pred_root is None:
            logger.error(f'No existing run found for dataset {dataset_name}, skip eval.')
            continue
        if not os.path.exists(pred_root) and RANK == 0:
            os.makedirs(pred_root, exist_ok=True)
        if WORLD_SIZE > 1:
            dist.barrier()
        dataset_kwargs = dict()
        if WORLD_SIZE > 1:
            if RANK == 0:
                dataset = build_dataset(dataset_name, **dataset_kwargs)
            dist.barrier()

        dataset = build_dataset(dataset_name, **dataset_kwargs)
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
                )
            else:
                infer_image_job(
                    agent,
                    pred_root,
                    model_name,
                    dataset,
                    api_nproc=args.num_proc,
                    html_limit=html_limit,
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
            logger.info(f"Evaluation results for {dataset_name}:\n{eval_results}")

            # Write structured summary JSON (task-aware, machine-readable)
            acc_file = get_intermediate_file_path(result_file, '_acc')
            acc_exists = os.path.exists(acc_file)
            if hasattr(eval_results, 'to_dict') and len(eval_results) > 0:
                metrics = eval_results.to_dict('records')[0]
            elif isinstance(eval_results, dict):
                metrics = eval_results
            else:
                metrics = {'raw': str(eval_results)}
            summary = {
                'model': model_name,
                'agent': agent_class,
                'dataset': dataset_name,
                'task_type': getattr(dataset, 'TYPE', DATASET_TYPE(dataset_name)),
                'modality': getattr(dataset, 'MODALITY', 'IMAGE'),
                'metrics': metrics,
                'files': {
                    'predictions': os.path.basename(result_file),
                    'accuracy': os.path.basename(acc_file) if acc_exists else None,
                },
                'timestamp': dt.now().isoformat(),
            }
            summary_file = os.path.join(pred_root, f'{model_name}_{dataset_name}_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Structured summary saved to {summary_file}")
