#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WORK_ROOT="/local/<user>/mini-vlm-toolkit/harness_search_logs"
N_EX_PER_DS=20
NUM_PROC="${NUM_PROC:-50}"
PARALLEL_STATES="${PARALLEL_STATES:-4}"
POOL_SIZE=3
MAX_PROPOSALS=3
NUM_ROUNDS=5
PLAYGROUND_MAX_TASKS=20
PLAYGROUND_TRIALS_PER_TASK=5
ROUTER_PORT="${ROUTER_PORT:-8102}"
CODEX_USAGE_THRESHOLD_PERCENT="${CODEX_USAGE_THRESHOLD_PERCENT:-90}"
CODEX_USAGE_SLEEP_CUSHION_SECONDS="${CODEX_USAGE_SLEEP_CUSHION_SECONDS:-60}"
CODEX_STATUS_FAILURE_MODE="${CODEX_STATUS_FAILURE_MODE:-warn}"
CODEX_STATUS_STARTUP_DELAY="${CODEX_STATUS_STARTUP_DELAY:-8}"
CODEX_EXEC_TIMEOUT_SECONDS="${CODEX_EXEC_TIMEOUT_SECONDS:-3600}"
CODEX_EXEC_RETRIES="${CODEX_EXEC_RETRIES:-1}"

CODEX_REASONING_EFFORT="high"

mkdir -p "${WORK_ROOT}"
cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "vlmevalkit" ]]; then
  PYTHON_CMD=(python)
else
  PYTHON_CMD=(conda run --no-capture-output -n vlmevalkit python)
fi

"${PYTHON_CMD[@]}" harness-search/scripts/run_search.py \
  --datasets ERQA MMSIBench_wo_circular \
  --num-examples "${N_EX_PER_DS}" \
  --num-proc "${NUM_PROC}" \
  --parallel-states "${PARALLEL_STATES}" \
  --num-rounds "${NUM_ROUNDS}" \
  --pool-size "${POOL_SIZE}" \
  --max-proposals "${MAX_PROPOSALS}" \
  --playground-max-tasks "${PLAYGROUND_MAX_TASKS}" \
  --playground-trials-per-task "${PLAYGROUND_TRIALS_PER_TASK}" \
  --work-root "${WORK_ROOT}" \
  --router-env-file mini_vlm_toolkit/tool_infra/.env.latest \
  --router-port "${ROUTER_PORT}" \
  --codex-bin /home/<user>/.local/bin/codex \
  --codex-model gpt-5.5 \
  --codex-reasoning-effort "${CODEX_REASONING_EFFORT}" \
  --codex-usage-threshold-percent "${CODEX_USAGE_THRESHOLD_PERCENT}" \
  --codex-usage-sleep-cushion-seconds "${CODEX_USAGE_SLEEP_CUSHION_SECONDS}" \
  --codex-status-failure-mode "${CODEX_STATUS_FAILURE_MODE}" \
  --codex-status-startup-delay "${CODEX_STATUS_STARTUP_DELAY}" \
  --codex-exec-timeout-seconds "${CODEX_EXEC_TIMEOUT_SECONDS}" \
  --codex-exec-retries "${CODEX_EXEC_RETRIES}"
