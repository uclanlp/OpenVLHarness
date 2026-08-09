#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Usage:
#   ./scripts/run_base_agent_with_tools_wise.sh [model_preset]
# Examples:
#   ./scripts/run_base_agent_with_tools_wise.sh                         # default: v5 umbrella + Qwen3-VL-8B
#   ./scripts/run_base_agent_with_tools_wise.sh v5-oh                   # openvl-harness aligned
#   ./scripts/run_base_agent_with_tools_wise.sh gpt5                    # gpt-5 baseline for comparison
#   NUM_EXAMPLES=20 ./scripts/run_base_agent_with_tools_wise.sh         # smoke
#
# Dataset: WISE-Verified (arXiv 2503.07265, Niu et al. 2025)
#   1000 knowledge-intensive text-to-image prompts across 6 domains
#   (CULTURE / TIME / SPACE / BIOLOGY / PHYSICS / CHEMISTRY). Each prompt
#   requires world-knowledge inference (not stated in the prompt surface)
#   to produce a semantically correct image. No GT reference image; the
#   LLM judge sees (PROMPT, EXPLANATION, generated_image) and emits a
#   binary Score in {0, 1} per the paper's protocol. See
#   dataset/utils/judge_wise.py for the byte-exact judge prompt.
#
# Data source: a WISE repo checkout at $LMUData/WISE/data_verified/
#   Override via env var WISE_DATA_DIR to point to a different checkout.
#
# Paper compare: Qwen-Image-Agent (arXiv 2606.26907) Table 2 reports:
#   Qwen-Image-2.0 base: 0.7954 overall
#   Qwen-Image-Agent  : 0.9020 overall (+10.7 points via agentic search)
#
# Judge: only judge_wise (paper protocol, binary). No override.
#
# Presets (v5 nightly family, all use BaseAgentWithToolsAndCode):
#   v5              -> configs/rui-testing/base_agent_with_tools_v5_rui.yaml       (umbrella, DEFAULT)
#   v5-under        -> configs/rui-testing/base_agent_with_tools_v5_under_rui.yaml (understanding-only; NOT useful for gen)
#   v5-oh           -> configs/rui-testing/base_agent_with_tools_v5_oh_rui.yaml    (openvl-harness aligned)
#   gpt5            -> configs/rui-testing/base_agent_with_tools-gpt5_rui.yaml     (gpt-5 comparison)

MODEL_PRESET="${1:-v5}"
DATASET="${DATASET:-WISE}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/rui-testing/judge_rui.yaml}"
NUM_PROC="${NUM_PROC:-4}"
SAVE_HTML="${SAVE_HTML:-10}"
SAVE_RAW_TRAJECTORY="${SAVE_RAW_TRAJECTORY:-1}"  # default ON; override with SAVE_RAW_TRAJECTORY= bash ... to disable
NUM_EXAMPLES="${NUM_EXAMPLES:-}"
MODE="${MODE:-all}"
SAMPLE_FILTER="${SAMPLE_FILTER:-all}"
SAVE_IMAGES="${SAVE_IMAGES:-files}"                # REQUIRED for WISE evaluate() to resolve generated images
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-default}"

if [[ -n "${CONFIG:-}" ]]; then
  AGENT_CONFIG="$CONFIG"
else
  case "$MODEL_PRESET" in
    v5|tools-v5|v5-all|v5_all)
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools_v5_rui.yaml"
      ;;
    v5-under|v5_under|tools-v5-under)
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools_v5_under_rui.yaml"
      ;;
    v5-under-qwen|v5_under_qwen|tools-v5-under-qwen)
      # v5-under with official Qwen3-VL-Instruct sampling (temp0.7/top_p0.8/
      # top_k20/rep1.0/presence1.5) — comparable to cf v8_instruct history.
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools_v5_under_qwen_rui.yaml"
      ;;
    v5-oh|v5_oh|tools-v5-oh)
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools_v5_oh_rui.yaml"
      ;;
    gpt5|gpt-5)
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools-gpt5_rui.yaml"
      ;;
    *)
      echo "Unknown model preset: $MODEL_PRESET"
      echo "Supported presets: v5 (default) | v5-under | v5-under-qwen | v5-oh | gpt5"
      echo "Or pass CONFIG=<path-to-yaml> to use a custom config."
      exit 1
      ;;
  esac
fi

# Fail fast with a clear message: the preset paths above point at a local
# configs/rui-testing/ tree that is not part of this repo, so a bare preset run
# would otherwise die later with a confusing error.
for _cfg_var in AGENT_CONFIG JUDGE_CONFIG; do
  _cfg="${!_cfg_var}"
  if [[ ! -f "$_cfg" ]]; then
    echo "ERROR: $_cfg_var not found: $_cfg" >&2
    echo "Pass CONFIG=<agent.yaml> / JUDGE_CONFIG=<judge.yaml> pointing at configs that exist" >&2
    echo "in your checkout, e.g. CONFIG=configs/base_agent_with_tools.yaml" >&2
    exit 1
  fi
done

# WISE only supports judge_wise (no override).
if [[ -n "${MMDR_JUDGE_PROMPT:-}" && "${MMDR_JUDGE_PROMPT}" != "judge_wise" ]]; then
  echo "WARNING: MMDR_JUDGE_PROMPT=${MMDR_JUDGE_PROMPT} overridden — WISE supports only judge_wise." >&2
fi
export MMDR_JUDGE_PROMPT="judge_wise"

# Ensure --save-images files is set (needed for generated image resolution)
if [[ "$SAVE_IMAGES" != "files" ]]; then
  echo "WARNING: SAVE_IMAGES=${SAVE_IMAGES} — WISE evaluate() requires 'files' to resolve generated images from disk." >&2
fi

# Verify WISE data is reachable. WISE runs from the cached $LMUData/WISE.tsv when
# present (load_data reads it directly); only the FIRST build needs merge.json.
# Source dir defaults under $LMUData (portable across checkouts) — override via
# WISE_DATA_DIR. No machine-specific absolute paths.
_LMU="${LMUData:-$HOME/LMUData}"
_WISE_DIR="${WISE_DATA_DIR:-$_LMU/WISE/data_verified}"
_WISE_TSV="$_LMU/WISE.tsv"
if [[ ! -f "$_WISE_TSV" && ! -f "$_WISE_DIR/merge.json" ]]; then
  echo "ERROR: WISE data not found — need cached $_WISE_TSV OR $_WISE_DIR/merge.json" >&2
  echo "Clone https://github.com/PKU-YuanGroup/WISE and set WISE_DATA_DIR to its data_verified/ dir (or provide the cached WISE.tsv)." >&2
  exit 1
fi

echo "Running BaseAgentWithToolsAndCode on WISE-Verified"
echo "  config:        $AGENT_CONFIG"
echo "  judge-config:  $JUDGE_CONFIG"
echo "  judge-prompt:  judge_wise (only supported)"
echo "  dataset:       $DATASET"
echo "  wise-data-dir: $_WISE_DIR"
echo "  num-proc:      $NUM_PROC"
echo "  save-html:     $SAVE_HTML"
echo "  save-images:   $SAVE_IMAGES (must be 'files' for WISE)"
echo "  num-examples:  ${NUM_EXAMPLES:-all}"
echo "  mode:          $MODE"

python run_benchmark.py \
  --config "$AGENT_CONFIG" \
  --judge-config "$JUDGE_CONFIG" \
  --data "$DATASET" \
  --num-proc "$NUM_PROC" \
  --incremental-json \
  --save-html "$SAVE_HTML" --save-images "$SAVE_IMAGES" \
  $( [[ "$SAVE_RAW_TRAJECTORY" == "1" ]] && echo --save-raw-trajectory ) \
  ${NUM_EXAMPLES:+--num-examples "$NUM_EXAMPLES"} \
  ${SHUFFLE:+--shuffle} \
  --seed "$SEED" --run-tag "$RUN_TAG" \
  ${RESUME:+--resume} ${RUN_DIR:+--run-dir "$RUN_DIR"} \
  --sample-filter "$SAMPLE_FILTER" --mode "$MODE"
