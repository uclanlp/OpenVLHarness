"""python -m openvlharness.toolsmith.pipeline --config PIPELINE_YAML --rollouts DIR... --work-dir DIR [--rebuild-gaps]"""
import argparse
import sys

import yaml

from ...bootstrap import expand_environment
from . import Pipeline


def main():
    """Parse the command line, run the pipeline and print the decision counts."""
    parser = argparse.ArgumentParser(prog="python -m openvlharness.toolsmith.pipeline",
                                     description="Grow a ToolSmith catalog from failed rollouts.")
    parser.add_argument("--config", required=True, help="Pipeline YAML, e.g. configs/toolsmith/pipeline.yaml")
    parser.add_argument("--rollouts", nargs="+", required=True,
                        help="Dataset directories of an evaluated openvlharness-eval run of the baseline agent")
    parser.add_argument("--work-dir", required=True, help="Directory for gaps.jsonl, mining.json and decisions.jsonl")
    parser.add_argument("--rebuild-gaps", action="store_true",
                        help="Mine the rollouts again instead of reusing gaps.jsonl, which is reused only for the "
                             "rollouts and mining settings recorded in mining.json")
    args = parser.parse_args()
    try:
        with open(args.config) as stream:
            config = expand_environment(yaml.safe_load(stream), label="pipeline configuration", source=f" {args.config}")
        pipeline = Pipeline(config, args.work_dir)
        counts = pipeline.run(args.rollouts, rebuild_gaps=args.rebuild_gaps)
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    for decision, count in sorted(counts.items()):
        print(f"{decision}: {count}")


if __name__ == "__main__":
    main()
