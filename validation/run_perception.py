"""Validate or run the frozen ODinW/FSC147 integration subset using local datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# The evaluator is used from the checkout; openvlharness must be installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def selected_datasets(manifest, data_root):
    """Load the frozen indices and verify source image and question hashes."""
    from openvlharness_eval.dataset import build_dataset

    selected = []
    for name, spec in manifest["datasets"].items():
        dataset = build_dataset(name)
        dataset.data["index"] = dataset.data["index"].astype(str)
        indexed = dataset.data.set_index("index")
        for sample in spec["samples"]:
            image = data_root / sample["image"]
            if hashlib.sha256(image.read_bytes()).hexdigest() != sample["image_sha256"]:
                raise ValueError(f"Frozen image hash mismatch: {sample['index']}")
            question = str(indexed.loc[sample["index"], "question"])
            if hashlib.sha256(question.encode()).hexdigest() != sample["question_sha256"]:
                raise ValueError(f"Frozen question hash mismatch: {sample['index']}")
        dataset.data = indexed.loc[[sample["index"] for sample in spec["samples"]]].reset_index()
        selected.append((name, dataset))
    return selected


def main():
    """Run direct-metric evaluation with a fresh output directory and saved identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("perception_samples.json"))
    parser.add_argument("--data-root", type=Path, required=True, help="Prepared LMUData directory")
    parser.add_argument("--config", default="configs/agent/qwen3_paper.yaml", help="Agent-system YAML for live runs")
    parser.add_argument("--output-dir", type=Path, help="New run directory; required for live runs")
    parser.add_argument("--check-only", action="store_true", help="Verify inputs without initializing model or tool clients")
    args = parser.parse_args()
    os.environ["LMUData"] = str(args.data_root.resolve())
    os.environ.setdefault("GLIP_ROOT", str(args.data_root.resolve() / "GLIP"))
    os.environ["FULL_TRAJECTORIES_LIMIT"] = "-1"
    manifest = json.loads(args.manifest.read_text())
    datasets = selected_datasets(manifest, args.data_root)
    if args.check_only:
        print(json.dumps({name: len(dataset.data) for name, dataset in datasets}))
        return
    if args.output_dir is None:
        parser.error("Live validation requires --output-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use an empty --output-dir; validation never overwrites earlier results")

    from openvlharness.bootstrap import build_agent, load_agent_config
    from openvlharness_eval import infer_image_job
    from openvlharness_eval.smp import get_pred_file_path
    from openvlharness_eval.utils.run_identity import build_run_manifest

    config = load_agent_config(args.config)
    agent = build_agent(config)
    model = agent.model.split("/")[-1]
    judge = {"judge_config_version": "direct_metrics_v1", "model_name": None, "score_mode": "constant"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = build_run_manifest(run_id=args.output_dir.name,
        agent_config_version=config["agent_config_version"], judge_config_version=judge["judge_config_version"],
        agent_config=config, judge_config=judge, config_path=args.config,
        judge_config_path="configs/judge_no_judge.yaml")
    (args.output_dir / "run_manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    (args.output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, dataset in datasets:
        infer_image_job(agent, str(args.output_dir), model, dataset, api_nproc=1, html_limit=-1, save_images="files")
        predictions = get_pred_file_path(str(args.output_dir), model, name, use_env_format=True)
        dataset.evaluate(predictions, model=None, score_mode="constant", nproc=1)
    print(f"Validation artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
