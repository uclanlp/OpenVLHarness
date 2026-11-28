"""python -m openvlharness.toolsmith.pipeline --config PIPELINE_YAML --rollouts DIR... --work-dir DIR
[--rebuild-gaps] [--retry-failed]

Exit status:
  0  Done. The summary lists the gaps and self-critiques set aside after failing in
     FAILED_RUNS runs; --retry-failed retries them.
  1  The pipeline cannot run; the error says why.
  2  The command line is wrong (argparse).
  3  A self-critique or a gap failed in this run (a model or network call failed); running
     the same command again retries it, until it has failed in FAILED_RUNS runs.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

from ...bootstrap import expand_environment
from . import FAILED_RUNS, Pipeline, json_lines

FAILED = 3  # the exit status of a run in which a self-critique or a gap failed; argparse uses 2


def main():
    """Parse the command line, run the pipeline, print what it did and exit with its status."""
    parser = argparse.ArgumentParser(prog="python -m openvlharness.toolsmith.pipeline",
                                     description="Grow a ToolSmith catalog from failed rollouts.")
    parser.add_argument("--config", required=True, help="Pipeline YAML, e.g. configs/toolsmith/pipeline.yaml")
    parser.add_argument("--rollouts", nargs="+", required=True,
                        help="Dataset directories of an evaluated openvlharness-eval run of the baseline agent")
    parser.add_argument("--work-dir", required=True,
                        help="Directory for gaps.jsonl, mining.json, critiques.jsonl and decisions.jsonl")
    parser.add_argument("--rebuild-gaps", action="store_true",
                        help="Mine the rollouts again instead of reusing gaps.jsonl, which is reused only for the "
                             "rollouts and mining settings recorded in mining.json; saved self-critiques are reused")
    parser.add_argument("--retry-failed", action="store_true",
                        help=f"Retry the gaps and self-critiques set aside after failing in {FAILED_RUNS} runs with "
                             "the same settings")
    args = parser.parse_args()
    try:
        with open(args.config) as stream:
            config = expand_environment(yaml.safe_load(stream), label="pipeline configuration",
                                        source=f" {args.config}",
                                        guide='step 3 of "Grow your own catalog" in toolsmith/README.md')
        pipeline = Pipeline(config, args.work_dir)
        counts = pipeline.run(args.rollouts, rebuild_gaps=args.rebuild_gaps, retry_failed=args.retry_failed)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(f"error: {exc}")
    print(summary(counts, pipeline))
    if counts.get("failed") or getattr(pipeline, "critique_failures", 0):
        sys.exit(FAILED)


def summary(counts, pipeline):
    """Describe what a run did with the gaps, also when it had nothing left to decide."""
    work = Path(pipeline.work_dir)
    path, record = work / "gaps.jsonl", work / "mining.json"
    gaps, processed = len(json_lines(path)), sum(counts.values())
    mined = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
    unmined, given_up = len(mined.get("failed_critiques") or []), len(mined.get("abandoned_critiques") or [])
    abandoned = len(getattr(pipeline, "abandoned", []))
    labels = mined.get("labels")
    if not gaps:
        if labels is None:
            lines = [f"No gaps to decide ({path} is empty)."]
        elif labels:
            lines = [f"No gaps to decide: none of the {sum(labels.values())} critiqued failed queries was labeled "
                     f"NEW_TOOL_NEEDED ({path} is empty)."]
        elif unmined or given_up:
            lines = [f"No gaps to decide yet: every self-critique failed, so no failed query has a label ({path} is "
                     "empty)."]
        else:
            lines = [f"No gaps to decide: the rollouts hold no failed query ({path} is empty)."]
    elif not processed:
        lines = [f"Nothing left to decide: all {gaps} gaps in {path} have a decision"
                 + (f", or failed in {FAILED_RUNS} runs." if abandoned else ".")]
    else:
        before = gaps - processed - abandoned
        lines = [f"Processed {processed} of the {gaps} gaps in {path}"
                 + (f"; the other {before} had a decision already." if before > 0 else ".")]
        lines += [f"  {decision}: {count}" for decision, count in sorted(counts.items())]
    if unmined:
        lines.append(f"{unmined} self-critiques failed, so their queries are not mined yet; run the same command "
                     "again to retry them.")
    if given_up:
        lines.append(f"{given_up} self-critiques failed in {FAILED_RUNS} runs with these settings and are set aside; "
                     "pass --retry-failed to try them again, also after a new API key (a change to the analyst's "
                     "other settings or max_tokens retries them too).")
    if abandoned:
        lines.append(f"{abandoned} gaps failed in {FAILED_RUNS} runs with these settings and are set aside; pass "
                     "--retry-failed to try them again, also after a new API key (a change to the models' other "
                     "settings, the gates' or max_tokens retries them too).")
    if counts.get("admitted"):
        lines.append(f"New tools are in {pipeline.catalog}; restart the router to serve them.")
    if counts.get("failed"):
        lines.append(f"The failed gaps have no decision yet; run the same command again to retry them (each is "
                     f"retried until it has failed in {FAILED_RUNS} runs).")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
