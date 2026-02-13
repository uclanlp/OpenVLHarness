from dotenv import load_dotenv
load_dotenv(".env")

import os
import argparse
import json
from datetime import datetime as dt

from mini_vlm_toolkit.agent import build_agent
from mini_vlm_toolkit.dataset import build_dataset, DATASET_TYPE
from mini_vlm_toolkit.smp import *
from mini_vlm_toolkit import infer_video_job, infer_image_job
from mini_vlm_toolkit.utils import load_judge_config

def eval_job():
    pass

RANK = int(os.environ.get('RANK', 0))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE",1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK",1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--config', type=str, default='configs/base_agent.yaml', help='Path to agent config file')
    parser.add_argument('--judge-config', type=str, default='configs/judge.yaml', help='Path to judge config file')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer', 'eval'], help='Select running mode')
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    parser.add_argument('--num-proc', type=int, default=4, help='Number of processes for API inference')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted run: use latest existing run dir, skip completed samples')
    args = parser.parse_args()

    logger = get_logger("RUN")

    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )

    agent = build_agent(args.config)
    judge_kwargs = load_judge_config(args.judge_config)
    model_name = agent.model.split('/')[-1]
    agent_class = agent.__class__.__name__

    logger.info(f"LLM Judge Configurations: {judge_kwargs}")
    logger.info(f"Initialized agent of class {agent_class} with model {model_name}")
    

    base_pred_root = os.path.join(args.work_dir, model_name, agent_class.lower())
    run_timestamp = dt.now().strftime('%Y%m%d_%H%M%S')

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
                infer_video_job(agent, pred_root, model_name, dataset, dataset_name, api_nproc=args.num_proc)
            else:
                infer_image_job(agent, pred_root, model_name, dataset, api_nproc=args.num_proc)

        
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
