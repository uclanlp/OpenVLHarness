"""
CLI entrypoint for the experience learning pipeline.

Usage:
    # Using existing trajectories (trajectory.source: "existing" in reflection config)
    python run_experience_learning.py \\
        --reflection-config configs/cf-testing/reflection.yaml

    # Running inference on the fly (trajectory.source: "infer" in reflection config)
    python run_experience_learning.py \\
        --config configs/base_agent_with_tools.yaml \\
        --reflection-config configs/cf-testing/reflection.yaml \\
        --data RealWorldQA

    # With runtime overrides
    python run_experience_learning.py \\
        --config configs/base_agent_with_tools.yaml \\
        --reflection-config configs/cf-testing/reflection.yaml \\
        --data RealWorldQA \\
        --num-examples 50 --num-proc 4
"""

from dotenv import load_dotenv
load_dotenv(".env")

import argparse
import glob
import json
import os
import os.path as osp
import shutil
from datetime import datetime, timezone

import yaml

from mini_vlm_toolkit.experience.constant import resolve_instance_reflection_prompt
from mini_vlm_toolkit.experience.reflector import (
    filter_for_reflection,
    load_experience,
    load_full_trajectories,
    load_results_with_answers,
    run_reflector,
    save_experience,
)
from mini_vlm_toolkit.smp import get_logger, get_pred_file_path, load as smp_load
from mini_vlm_toolkit.smp.file import get_intermediate_file_path
from mini_vlm_toolkit.utils import load_judge_config
from mini_vlm_toolkit.utils.run_identity import redact_sensitive_fields, sanitize_for_path


logger = get_logger("ExperienceLearning")

_VALID_STRATEGY_COMBOS = {
    ("batch", "experience_file"),
    ("instance", "instance_critique"),
    ("instance_summarize", "experience_file"),
}


def _validate_strategy_combo(generation_strategy: str, passing_strategy: str):
    """Raise ValueError if the generation/passing strategy combo is invalid."""
    if (generation_strategy, passing_strategy) not in _VALID_STRATEGY_COMBOS:
        valid_str = ", ".join(f"{g}+{p}" for g, p in sorted(_VALID_STRATEGY_COMBOS))
        raise ValueError(
            f"Invalid strategy combination: generation_strategy={generation_strategy!r}, "
            f"passing_strategy={passing_strategy!r}. Valid combos: {valid_str}"
        )


# ====================================================================
# Helpers
# ====================================================================

def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True, allow_unicode=True)


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0.5
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"1", "true", "yes", "y", "correct", "hit"}:
            return True
        if val in {"0", "false", "no", "n", "incorrect", "miss"}:
            return False
        try:
            return float(val) > 0.5
        except ValueError:
            return False
    # for vqa
    if isinstance(value, list):
        return sum(value) > 0
    return False


def _extract_correctness_map(data):
    if not hasattr(data, "columns"):
        return {}

    columns_lower = {str(c).lower(): c for c in data.columns}
    idx_col = columns_lower.get("index")
    if idx_col is None:
        return {}

    metric_col = None
    for name in ["is_correct", "correct", "hit", "score", "match", "eval_score", "eval_match"]:
        if name in columns_lower:
            metric_col = columns_lower[name]
            break
    if metric_col is None:
        return {}

    mapping = {}
    for _, row in data.iterrows():
        idx = row[idx_col]
        mapping[str(idx)] = _to_bool(row[metric_col])
    return mapping


def _load_eval_correctness_map(result_file):
    candidates = [result_file]
    stem, _ = osp.splitext(result_file)
    candidates.extend(sorted(glob.glob(f"{stem}_*result.*")))

    seen = set()
    for path in candidates:
        if path in seen or not osp.exists(path):
            continue
        seen.add(path)
        try:
            data = smp_load(path)
        except Exception:
            continue
        mapping = _extract_correctness_map(data)
        if mapping:
            logger.info(f"Using correctness map from: {path} ({len(mapping)} entries)")
            return mapping
    return {}


def _label_trajectories_with_eval(trajectories, results, correctness_map):
    labeled = []
    for record in trajectories:
        raw_task_id = record.get("task_id")
        task_id = str(raw_task_id)
        result_entry = (
            results.get(raw_task_id)
            or results.get(task_id)
            or {}
        )
        answer = result_entry.get("answer", "")
        question = result_entry.get("question", "")

        answer_text = result_entry.get(answer, "")
        if answer_text != "":
            answer = f"{answer}. {answer_text}"
        rec = dict(record)
        rec["ground_truth"] = answer 
        rec["question"] = question or rec.get("question", "")
        rec["is_correct"] = correctness_map.get(task_id, False)
        labeled.append(rec)
    return labeled


def _resolve_dataset_name(source, trajectory_cfg, trajectories, cli_data):
    if source == "infer":
        return cli_data[0]

    dataset_name = trajectory_cfg.get("dataset")
    if dataset_name:
        return dataset_name
    if trajectories:
        first = trajectories[0]
        dataset_name = first.get("dataset")
        if dataset_name:
            return dataset_name
    if cli_data:
        return cli_data[0]
    raise ValueError(
        "Unable to resolve dataset name for existing trajectories. "
        "Set trajectory.dataset in reflection config or pass --data."
    )


def _resolve_model_name(source, trajectories, agent_model):
    if source == "existing" and trajectories:
        model_name = trajectories[0].get("model_name")
        if model_name:
            return str(model_name)
    if agent_model:
        return agent_model.split("/")[-1]
    return "unknown_model"


def _resolve_agent_class_name(source, trajectories, agent_class_name):
    if source == "existing" and trajectories:
        cls = trajectories[0].get("agent_class")
        if cls:
            return str(cls)
    if agent_class_name:
        return str(agent_class_name)
    return "unknown_agent"


def _build_reflection_output_dir(
    work_dir,
    model_name,
    agent_class,
    run_id,
    reflector_model,
    dataset_name=None,
):
    model_part = sanitize_for_path(model_name, field_name="model_name")
    agent_part = sanitize_for_path(agent_class.lower(), field_name="agent_class")
    reflector_part = sanitize_for_path(reflector_model, field_name="reflector_model")
    if dataset_name:
        dataset_part = sanitize_for_path(dataset_name, field_name="dataset_name")
        return osp.join(work_dir, model_part, agent_part, run_id, dataset_part, reflector_part)
    return osp.join(work_dir, model_part, agent_part, run_id, reflector_part)


def _prepare_reflection_run_dir(
    output_dir,
    run_id,
    source,
    dataset_name,
    model_name,
    agent_class,
    reflector_model,
    reflection_config,
    reflection_config_path,
    judge_config_path,
    strategy,
    reflect_on,
):
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "dataset_name": dataset_name,
        "agent_model_name": model_name,
        "agent_class": agent_class,
        "reflector_model": reflector_model,
        "strategy": strategy,
        "reflect_on": reflect_on,
        "config_paths": {
            "reflection": osp.abspath(reflection_config_path),
            "judge": osp.abspath(judge_config_path) if judge_config_path else None,
        },
    }
    _write_json(osp.join(output_dir, "reflection_manifest.json"), manifest)

    redacted_cfg = redact_sensitive_fields(reflection_config)
    _write_yaml(osp.join(output_dir, "reflection_config.redacted.yaml"), redacted_cfg)


def _get_eval_name_str(judge_kwargs):
    """Derive the name_str used for eval pkl filenames (mirrors image_mcq.py logic)."""
    model = judge_kwargs.get('model', 'exact_matching')
    name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
    name_str = name_str_map.get(model, model)
    if '/' in name_str:
        name_str = name_str.split('/')[-1]
    return name_str


def _prepopulate_eval_pkl(result_file, judge_kwargs, carried_forward):
    """Pre-populate the eval result pkl with carried-forward eval results.

    This allows the evaluator's skip logic (checking if index is already in
    the result pkl) to skip re-evaluating carried-forward correct items.
    """
    from mini_vlm_toolkit.smp import dump as smp_dump

    name_str = _get_eval_name_str(judge_kwargs)
    eval_pkl = get_intermediate_file_path(result_file, f'_{name_str}_result', 'pkl')

    eval_entries = {}
    for task_id, entry in carried_forward.items():
        eval_result = entry.get("eval_result")
        if eval_result is not None:
            # Use the original task_id type; also store as int if possible
            # since the evaluator uses int index keys
            eval_entries[task_id] = eval_result
            try:
                int_id = int(task_id)
                if int_id != task_id:
                    eval_entries[int_id] = eval_result
            except (ValueError, TypeError):
                pass

    if eval_entries:
        os.makedirs(osp.dirname(eval_pkl), exist_ok=True)
        smp_dump(eval_entries, eval_pkl)
        logger.info(
            f"Pre-populated eval pkl with {len(eval_entries)} carried-forward results: "
            f"{osp.basename(eval_pkl)}"
        )


def _run_dataset_evaluation(dataset_name, result_file, judge_kwargs):
    from mini_vlm_toolkit.dataset import build_dataset

    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_name} is not valid.")

    logger.info(f"Running dataset evaluation for reflection labels: {dataset_name}")
    dataset.evaluate(result_file, **judge_kwargs)


def _find_result_xlsx(result_file, judge_kwargs):
    """Find the judge-named result xlsx (e.g. ``*_Qwen3.5-9B_result.xlsx``).

    Derives ``name_str`` from ``judge_kwargs['model']`` using the same logic
    as the dataset evaluate methods in ``image_mcq.py``.
    """
    model = judge_kwargs.get("model", "exact_matching")
    name_str_map = {"chatgpt-0125": "openai", "gpt-4-0125": "gpt4"}
    name_str = name_str_map.get(model, model)
    if "/" in name_str:
        name_str = name_str.split("/")[-1]

    stem, ext = osp.splitext(result_file)
    candidate = f"{stem}_{name_str}_result{ext}"
    if osp.exists(candidate):
        return candidate

    # Try common xlsx/tsv variants
    for try_ext in [".xlsx", ".tsv", ".csv"]:
        candidate = f"{stem}_{name_str}_result{try_ext}"
        if osp.exists(candidate):
            return candidate

    # Fallback: glob for any matching result file
    matches = sorted(glob.glob(f"{stem}_*result.*"))
    if matches:
        return matches[0]

    return None


def _get_labeled_trajectories(result_file, dataset_name, judge_kwargs, trajectory_dir):
    """Load trajectories from the result xlsx and full_trajectories/ dir.

    1. Find the judge-named result xlsx (per-instance correct mapping).
    2. Extract correctness from the xlsx.
    3. Load answers from the xlsx.
    4. Load trajectory JSONs from ``full_trajectories/`` using the ``index``
       column from the xlsx.
    """
    # Step 1: Find result xlsx
    result_xlsx = _find_result_xlsx(result_file, judge_kwargs)
    if result_xlsx:
        logger.info(f"Found result xlsx: {result_xlsx}")
    else:
        # Run evaluation to generate the result xlsx
        _run_dataset_evaluation(dataset_name, result_file, judge_kwargs)
        result_xlsx = _find_result_xlsx(result_file, judge_kwargs)

    # Determine which file to use for correctness + answers
    eval_file = result_xlsx or result_file

    # Step 2: Extract correctness
    correctness_map = {}
    if eval_file and osp.exists(eval_file):
        try:
            data = smp_load(eval_file)
            correctness_map = _extract_correctness_map(data)
            if correctness_map:
                logger.info(f"Using correctness map from: {eval_file} ({len(correctness_map)} entries)")
        except Exception:
            pass

    if not correctness_map:
        correctness_map = _load_eval_correctness_map(result_file)

    if not correctness_map:
        raise ValueError(
            "Could not derive per-instance correctness from evaluation outputs. "
            "Please provide an existing_results_file that includes evaluation detail rows."
        )

    # Step 3: Load answers from result xlsx
    results = load_results_with_answers(eval_file)

    # Step 4: Load trajectory JSONs using index from the xlsx
    trajectories = []
    loaded_indices = set()
    for idx in results.keys():
        traj_path = osp.join(trajectory_dir, f"{idx}.json")
        if osp.exists(traj_path):
            with open(traj_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            trajectories.append(record)
            loaded_indices.add(str(idx))

    if not trajectories:
        # Fallback: load all trajectories from directory
        logger.warning(
            "No trajectories matched by index from xlsx. "
            "Falling back to loading all trajectories from directory."
        )
        trajectories = load_full_trajectories(trajectory_dir)
    else:
        logger.info(
            f"Loaded {len(trajectories)} trajectories by index from {trajectory_dir}"
        )

    labeled = _label_trajectories_with_eval(trajectories, results, correctness_map)
    return labeled


# ====================================================================
# Inference helpers
# ====================================================================

def _run_inference_trial(
    agent_config_path,
    dataset_name,
    work_dir,
    trial,
    num_examples,
    api_nproc,
    experience_file,
    save_trajectories,
    instance_critique_dir=None,
    carried_forward=None,
):
    """Run inference on a single dataset for one trial.

    ``carried_forward`` is a dict mapping task_id ->
    {"prediction": ..., "eval_result": ...} for the given dataset.

    Returns (trajectory_dir, result_file, model_name, dataset_name, agent_class_name).
    """
    from mini_vlm_toolkit import infer_image_job
    from mini_vlm_toolkit.agent import get_agent_class, load_agent_config
    from mini_vlm_toolkit.dataset import build_dataset
    from mini_vlm_toolkit.smp import dump

    config = load_agent_config(agent_config_path)
    agent_kwargs = config.get("agent_kwargs", {})
    if experience_file and os.path.exists(experience_file):
        agent_kwargs["experience_file"] = os.path.abspath(experience_file)
    if instance_critique_dir and os.path.isdir(instance_critique_dir):
        agent_kwargs["instance_critique_dir"] = os.path.abspath(instance_critique_dir)

    agent_name = config.get("agent_name", "BaseAgent")
    agent_class = get_agent_class(agent_name)
    agent = agent_class(**agent_kwargs)
    model_name = agent.model.split("/")[-1]
    trial_dir = osp.join(work_dir, f"trial_{trial}")
    os.makedirs(trial_dir, exist_ok=True)

    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise ValueError(f"Dataset {dataset_name} is not valid.")
    if num_examples is not None:
        dataset.data = dataset.data.head(num_examples)
        logger.info(f"Limiting dataset {dataset_name} to first {num_examples} examples")

    # Pre-populate supp file with carried-forward predictions (skip_correct)
    if carried_forward:
        supp_file = osp.join(trial_dir, f'{model_name}_supp.pkl')
        os.makedirs(trial_dir, exist_ok=True)
        supp_data = {k: v["prediction"] for k, v in carried_forward.items()}
        dump(supp_data, supp_file)
        logger.info(
            f"Trial {trial}: skipping {len(carried_forward)} previously-correct "
            f"instances for {dataset_name}"
        )

    logger.info(f"Trial {trial}: running inference on {dataset_name} with {model_name}")

    infer_image_job(
        agent,
        trial_dir,
        model_name,
        dataset,
        api_nproc=api_nproc,
        enable_incremental_json=False,
        use_dataset_subdir=False,
    )

    trajectory_dir = osp.join(trial_dir, "full_trajectories")
    # With use_dataset_subdir=False, files live directly in trial_dir.
    # Pass dataset_name=None to prevent get_pred_file_path from adding a subdir.
    result_file = get_pred_file_path(trial_dir, model_name, dataset_name=None, use_env_format=True)

    # Optionally clean up trajectories to save space
    if not save_trajectories:
        trajectories_jsonl = osp.join(trial_dir, "trajectories.jsonl")
        if os.path.exists(trajectories_jsonl):
            os.remove(trajectories_jsonl)

    return trajectory_dir, result_file, model_name, dataset_name, agent.__class__.__name__


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Learn tool-usage principles from playground data",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to agent config YAML (same as run_benchmark.py, needed when trajectory.source='infer')",
    )
    parser.add_argument(
        "--judge-config", type=str, default="configs/judge.yaml",
        help="Path to judge config YAML used for dataset.evaluate (same as run_benchmark.py)",
    )
    parser.add_argument(
        "--reflection-config", type=str, required=True,
        help="Path to reflection config YAML (all reflection settings)",
    )
    parser.add_argument(
        "--data", type=str, nargs="+", default=None,
        help="Dataset name(s) for the playground (needed when trajectory.source='infer')",
    )
    parser.add_argument(
        "--num-examples", type=int, default=None,
        help="Limit playground to first N examples",
    )
    parser.add_argument(
        "--num-proc", type=int, default=4,
        help="Number of processes for API inference",
    )
    parser.add_argument(
        "--work-dir", type=str, default="./outputs/experience_learning",
        help="Working directory for trial outputs",
    )
    parser.add_argument(
        "--no-reflection", action="store_true",
        help="Run the pipeline without reflection for ablation",
    )
    args = parser.parse_args()

    # Load reflection config
    with open(args.reflection_config, "r", encoding="utf-8") as f:
        rconfig = yaml.safe_load(f)

    reflector_cfg = rconfig.get("reflector", {})
    learning_cfg = rconfig.get("learning", {})
    trajectory_cfg = rconfig.get("trajectory", {})
    output_cfg = rconfig.get("output", {})

    max_tries = learning_cfg.get("max_tries", 1)
    generation_strategy = learning_cfg.get("generation_strategy", "batch")
    passing_strategy = learning_cfg.get("passing_strategy", "experience_file")
    skip_correct = learning_cfg.get("skip_correct", False)
    reflect_on = learning_cfg.get("reflect_on", "all")
    summarize_chunk_size = learning_cfg.get("summarize_chunk_size", 0)
    no_reflection = args.no_reflection

    if not no_reflection:
        _validate_strategy_combo(generation_strategy, passing_strategy)
    source = trajectory_cfg.get("source", "infer")
    save_trajectories = trajectory_cfg.get("save_trajectories", True)
    save_all_intermediate = output_cfg.get("save_all_intermediate", False)
    experience_filename = osp.basename(output_cfg.get("experience_file", "experience.md"))

    judge_kwargs = {}
    if args.judge_config and osp.exists(args.judge_config):
        judge_kwargs = load_judge_config(args.judge_config)
        logger.info(f"Loaded judge config from {args.judge_config}: {judge_kwargs}")
    elif args.judge_config:
        logger.warning(
            f"Judge config not found: {args.judge_config}. "
            "Proceeding with empty judge_kwargs for dataset.evaluate."
        )

    os.makedirs(args.work_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Source: existing trajectories
    # ----------------------------------------------------------------
    if source == "existing":
        traj_dir = trajectory_cfg.get("existing_trajectories_dir")
        result_file = trajectory_cfg.get("existing_results_file")
        if not traj_dir or not result_file:
            raise ValueError(
                "trajectory.source='existing' requires both "
                "'existing_trajectories_dir' and 'existing_results_file' in reflection config."
            )

        logger.info(f"Using existing trajectories from {traj_dir}")
        # Load a sample trajectory to resolve metadata
        _sample_trajectories = load_full_trajectories(traj_dir)
        dataset_name = _resolve_dataset_name(source, trajectory_cfg, _sample_trajectories, args.data)

        model_name = _resolve_model_name(source, _sample_trajectories, None)
        agent_class = _resolve_agent_class_name(source, _sample_trajectories, None)
        reflector_model = reflector_cfg["model"]
        run_id = datetime.now().strftime("reflection_%Y%m%d_%H%M%S")
        output_dir = _build_reflection_output_dir(
            args.work_dir,
            model_name,
            agent_class,
            run_id,
            reflector_model,
            dataset_name=dataset_name,
        )
        _prepare_reflection_run_dir(
            output_dir=output_dir,
            run_id=run_id,
            source=source,
            dataset_name=dataset_name,
            model_name=model_name,
            agent_class=agent_class,
            reflector_model=reflector_model,
            reflection_config=rconfig,
            reflection_config_path=args.reflection_config,
            judge_config_path=args.judge_config,
            strategy=generation_strategy,
            reflect_on=reflect_on,
        )

        instance_reflection_prompt = resolve_instance_reflection_prompt(
            learning_cfg.get("instance_reflection_prompt")
        )

        labeled = _get_labeled_trajectories(
            result_file=result_file,
            dataset_name=dataset_name,
            judge_kwargs=judge_kwargs,
            trajectory_dir=traj_dir,
        )
        filtered = filter_for_reflection(labeled, reflect_on)

        n_correct = sum(1 for r in labeled if r.get("is_correct"))
        accuracy = n_correct / max(len(labeled), 1) * 100
        logger.info(f"Labeled: {n_correct}/{len(labeled)} correct ({accuracy:.1f}%)")
        logger.info(f"Reflecting on {len(filtered)} instances (reflect_on={reflect_on})")

        final_experience_path = osp.join(output_dir, experience_filename)

        # Check for previous experience to refine
        previous_experience = None
        if os.path.exists(final_experience_path):
            previous_experience = load_experience(final_experience_path)
            logger.info(f"Found existing experience at {final_experience_path}, will refine")

        instance_lessons_dir = osp.join(output_dir, "instance_lessons")
        principles = run_reflector(
            filtered,
            generation_strategy=generation_strategy,
            model=reflector_cfg["model"],
            base_url=reflector_cfg["base_url"],
            api_key=reflector_cfg.get("api_key"),
            temperature=reflector_cfg.get("temperature", 0.3),
            max_tokens=reflector_cfg.get("max_tokens", 4096),
            previous_experience=previous_experience,
            instance_lessons_dir=instance_lessons_dir,
            instance_reflection_prompt=instance_reflection_prompt,
            max_examples=learning_cfg.get("max_examples", 50),
            max_result_chars=learning_cfg.get("max_result_chars", 80),
            extra_body=reflector_cfg.get("extra_body"),
            chunk_size=summarize_chunk_size,
        )

        if principles is not None:
            metadata = {
                "num_total": len(labeled),
                "num_correct": n_correct,
                "accuracy": round(accuracy, 2),
                "generation_strategy": generation_strategy,
                "passing_strategy": passing_strategy,
                "reflect_on": reflect_on,
                "reflector_model": reflector_cfg["model"],
            }
            path = save_experience(principles, final_experience_path, metadata=metadata)
            logger.info(f"Experience saved to: {path}")

        logger.info(f"\nFinal experience: {os.path.abspath(final_experience_path)}")
        return

    # ----------------------------------------------------------------
    # Source: infer (run inference on the fly with multi-trial)
    # ----------------------------------------------------------------
    if not (args.config and args.data):
        raise ValueError(
            "trajectory.source='infer' requires --config and --data CLI args."
        )

    from mini_vlm_toolkit.agent import get_agent_class, load_agent_config

    agent_config = load_agent_config(args.config)
    agent_kwargs = agent_config.get("agent_kwargs", {})
    agent_name = agent_config.get("agent_name", "BaseAgent")
    agent_class = get_agent_class(agent_name).__name__
    model_name = str(agent_kwargs.get("model", "unknown_model")).split("/")[-1]
    dataset_names = args.data
    reflector_model = reflector_cfg.get("model", "none") if no_reflection else reflector_cfg["model"]

    if not no_reflection:
        instance_reflection_prompt = resolve_instance_reflection_prompt(
            learning_cfg.get("instance_reflection_prompt")
        )

    run_id = datetime.now().strftime("reflection_%Y%m%d_%H%M%S")
    output_dir = _build_reflection_output_dir(
        args.work_dir,
        model_name,
        agent_class,
        run_id,
        reflector_model,
    )
    _prepare_reflection_run_dir(
        output_dir=output_dir,
        run_id=run_id,
        source=source,
        dataset_name=", ".join(dataset_names),
        model_name=model_name,
        agent_class=agent_class,
        reflector_model=reflector_model,
        reflection_config=rconfig,
        reflection_config_path=args.reflection_config,
        judge_config_path=args.judge_config,
        strategy=generation_strategy,
        reflect_on=reflect_on,
    )

    for dataset_name in dataset_names:
        logger.info(
            f"Running {max_tries} trial(s) on {dataset_name} "
            f"with generation_strategy={generation_strategy}, "
            f"passing_strategy={passing_strategy}, skip_correct={skip_correct}"
        )

        ds_output_dir = osp.join(output_dir, dataset_name)
        infer_work_dir = osp.join(ds_output_dir, "trials")
        os.makedirs(infer_work_dir, exist_ok=True)

        experience_file = None
        instance_critique_dir = None
        previous_experience = None
        carried_forward = {}  # {task_id: {"prediction": ..., "eval_result": ...}}
        trial_stats = []
        instance_lessons_dir = osp.join(ds_output_dir, "instance_lessons")
        final_experience_path = osp.join(ds_output_dir, experience_filename)

        for trial in range(max_tries):
            logger.info(f"\n{'='*60}")
            logger.info(f"TRIAL {trial + 1}/{max_tries} [{dataset_name}]")
            logger.info(f"{'='*60}")

            # Step 1: Run inference
            trajectory_dir, result_file, model_name, _, agent_class = _run_inference_trial(
                agent_config_path=args.config,
                dataset_name=dataset_name,
                work_dir=infer_work_dir,
                trial=trial,
                num_examples=args.num_examples,
                api_nproc=args.num_proc,
                experience_file=experience_file if passing_strategy == "experience_file" else None,
                save_trajectories=save_trajectories,
                instance_critique_dir=instance_critique_dir if passing_strategy == "instance_critique" else None,
                carried_forward=carried_forward if skip_correct else None,
            )

            # Step 1.5: Pre-populate eval pkl with carried-forward results, then eval
            # if skip_correct and carried_forward:
            #     _prepopulate_eval_pkl(result_file, judge_kwargs, carried_forward)
            _run_dataset_evaluation(dataset_name, result_file, judge_kwargs)

            # Step 2: Label and filter via dataset eval
            labeled = _get_labeled_trajectories(
                result_file=result_file,
                dataset_name=dataset_name,
                judge_kwargs=judge_kwargs,
                trajectory_dir=trajectory_dir,
            )
            filtered = filter_for_reflection(labeled, reflect_on)

            n_correct = sum(1 for r in labeled if r.get("is_correct"))
            accuracy = n_correct / max(len(labeled), 1) * 100
            logger.info(
                f"Trial {trial + 1}: {n_correct}/{len(labeled)} correct ({accuracy:.1f}%), "
                f"reflecting on {len(filtered)} instances"
            )
            trial_stats.append({
                "trial": trial + 1,
                "correct": n_correct,
                "total": len(labeled),
                "accuracy": round(accuracy, 2),
            })

            # Track correct for skip_correct (predictions + eval results)
            if skip_correct:
                name_str = _get_eval_name_str(judge_kwargs)
                eval_pkl_path = get_intermediate_file_path(
                    result_file, f'_{name_str}_result', 'pkl'
                )
                eval_pkl_data = {}
                if osp.exists(eval_pkl_path):
                    try:
                        eval_pkl_data = smp_load(eval_pkl_path)
                    except Exception:
                        pass

                for r in labeled:
                    if r.get("is_correct"):
                        task_id = r["task_id"]
                        eval_result = eval_pkl_data.get(task_id)
                        if eval_result is None:
                            try:
                                eval_result = eval_pkl_data.get(int(task_id))
                            except (ValueError, TypeError):
                                pass
                        carried_forward[task_id] = {
                            "prediction": r.get("prediction", ""),
                            "eval_result": eval_result,
                        }

            # Step 3: Reflect (skip if no_reflection baseline)
            if no_reflection:
                logger.info(f"Trial {trial + 1}: no_reflection mode, skipping reflection step")
            elif generation_strategy == "batch":
                principles = run_reflector(
                    filtered,
                    generation_strategy=generation_strategy,
                    model=reflector_cfg["model"],
                    base_url=reflector_cfg["base_url"],
                    api_key=reflector_cfg.get("api_key"),
                    temperature=reflector_cfg.get("temperature", 0.3),
                    max_tokens=reflector_cfg.get("max_tokens", 4096),
                    previous_experience=previous_experience,
                    max_examples=learning_cfg.get("max_examples", 50),
                    max_result_chars=learning_cfg.get("max_result_chars", 80),
                    extra_body=reflector_cfg.get("extra_body"),
                )

                # Save trial experience
                if save_all_intermediate:
                    trial_output = osp.join(ds_output_dir, f"experience_v{trial + 1}.md")
                else:
                    trial_output = osp.join(ds_output_dir, "experience_latest.md")

                metadata = {
                    "trial": trial + 1,
                    "max_tries": max_tries,
                    "correct": n_correct,
                    "total": len(labeled),
                    "accuracy": round(accuracy, 2),
                    "generation_strategy": generation_strategy,
                    "passing_strategy": passing_strategy,
                    "reflector_model": reflector_cfg["model"],
                    "dataset": dataset_name,
                }
                save_experience(principles, trial_output, metadata=metadata)
                logger.info(f"Trial {trial + 1} experience saved to: {trial_output}")

                experience_file = trial_output
                previous_experience = principles

            elif generation_strategy == "instance":
                run_reflector(
                    filtered,
                    generation_strategy=generation_strategy,
                    model=reflector_cfg["model"],
                    base_url=reflector_cfg["base_url"],
                    api_key=reflector_cfg.get("api_key"),
                    temperature=reflector_cfg.get("temperature", 0.3),
                    max_tokens=reflector_cfg.get("max_tokens", 4096),
                    instance_lessons_dir=instance_lessons_dir,
                    instance_reflection_prompt=instance_reflection_prompt,
                    extra_body=reflector_cfg.get("extra_body"),
                )
                instance_critique_dir = instance_lessons_dir
                if save_all_intermediate:
                    snapshot_dir = osp.join(ds_output_dir, f"instance_lessons_v{trial + 1}")
                    shutil.copytree(instance_lessons_dir, snapshot_dir, dirs_exist_ok=True)
                    logger.info(f"Trial {trial + 1} instance lessons snapshot: {snapshot_dir}")
                logger.info(
                    f"Trial {trial + 1}: updated {len(filtered)} instance lessons "
                    f"in {instance_lessons_dir}"
                )

            elif generation_strategy == "instance_summarize":
                principles = run_reflector(
                    filtered,
                    generation_strategy=generation_strategy,
                    model=reflector_cfg["model"],
                    base_url=reflector_cfg["base_url"],
                    api_key=reflector_cfg.get("api_key"),
                    temperature=reflector_cfg.get("temperature", 0.3),
                    max_tokens=reflector_cfg.get("max_tokens", 4096),
                    previous_experience=previous_experience,
                    instance_lessons_dir=instance_lessons_dir,
                    instance_reflection_prompt=instance_reflection_prompt,
                    extra_body=reflector_cfg.get("extra_body"),
                    chunk_size=summarize_chunk_size,
                )

                # Save trial experience
                if save_all_intermediate:
                    trial_output = osp.join(ds_output_dir, f"experience_v{trial + 1}.md")
                else:
                    trial_output = osp.join(ds_output_dir, "experience_latest.md")

                metadata = {
                    "trial": trial + 1,
                    "max_tries": max_tries,
                    "correct": n_correct,
                    "total": len(labeled),
                    "accuracy": round(accuracy, 2),
                    "generation_strategy": generation_strategy,
                    "passing_strategy": passing_strategy,
                    "reflector_model": reflector_cfg["model"],
                    "dataset": dataset_name,
                }
                save_experience(principles, trial_output, metadata=metadata)
                logger.info(f"Trial {trial + 1} experience saved to: {trial_output}")

                if save_all_intermediate:
                    snapshot_dir = osp.join(ds_output_dir, f"instance_lessons_v{trial + 1}")
                    shutil.copytree(instance_lessons_dir, snapshot_dir, dirs_exist_ok=True)
                    logger.info(f"Trial {trial + 1} instance lessons snapshot: {snapshot_dir}")

                experience_file = trial_output
                previous_experience = principles

        # Copy final experience to the output path in this run directory
        if experience_file and os.path.exists(experience_file) and experience_file != final_experience_path:
            with open(experience_file, "r", encoding="utf-8") as src:
                with open(final_experience_path, "w", encoding="utf-8") as dst:
                    dst.write(src.read())

        logger.info(f"\nFinal experience for {dataset_name}: {os.path.abspath(final_experience_path)}")

        # Save accuracy trends
        os.makedirs(ds_output_dir, exist_ok=True)
        trends_path = osp.join(ds_output_dir, "accuracy_trends.json")
        _write_json(trends_path, trial_stats)
        logger.info(f"Accuracy trends saved to: {trends_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info(f"LEARNING SUMMARY [{dataset_name}]")
        logger.info("=" * 60)
        for stat in trial_stats:
            logger.info(
                f"  Trial {stat['trial']}: accuracy={stat['accuracy']:.1f}% "
                f"({stat['correct']}/{stat['total']})"
            )
        if len(trial_stats) > 1:
            delta = trial_stats[-1]["accuracy"] - trial_stats[0]["accuracy"]
            sign = "+" if delta >= 0 else ""
            logger.info(f"  Change: {sign}{delta:.1f}%")

        # Clean up full_trajectories dirs to save disk
        if not save_trajectories:
            for traj_dir in glob.glob(osp.join(infer_work_dir, "trial_*", "full_trajectories")):
                shutil.rmtree(traj_dir, ignore_errors=True)
            logger.info("Cleaned up full_trajectories directories")

if __name__ == "__main__":
    main()
