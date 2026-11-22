"""python -m openvlharness.toolsmith.pipeline --config PIPELINE_YAML --rollouts DIR... --work-dir DIR"""
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
    parser.add_argument("--work-dir", required=True, help="Directory for gaps.jsonl and decisions.jsonl")
    args = parser.parse_args()
    try:
        with open(args.config) as stream:
            config = expand_environment(yaml.safe_load(stream), label="pipeline configuration", source=f" {args.config}")
        pipeline = Pipeline(config, args.work_dir)
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    for decision, count in sorted(pipeline.run(args.rollouts).items()):
        print(f"{decision}: {count}")


if __name__ == "__main__":
    main()
