#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

DATASET="${DATASET:-MMStar}"
CONFIG="${CONFIG:-configs/base_agent_with_tools.yaml}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/judge.yaml}"
NUM_PROC="${NUM_PROC:-4}"
RUN_TAG="${RUN_TAG:-rl_eval}"

# Serve trained checkpoints with your serving stack using Hermes parser before running:
#   --tool-call-parser hermes

python run_benchmark.py \
  --config "$CONFIG" \
  --judge-config "$JUDGE_CONFIG" \
  --data "$DATASET" \
  --num-proc "$NUM_PROC" \
  --run-tag "$RUN_TAG"
