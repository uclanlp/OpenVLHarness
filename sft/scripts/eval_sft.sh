#!/usr/bin/env bash
set -euo pipefail

# Post-SFT Evaluation on MMSearch
#
# Usage:
#   bash sft/scripts/eval_sft.sh <model_path>
#   bash sft/scripts/eval_sft.sh /path/to/sft_model
#
# Environment variables:
#   MODEL_PATH       — path to the trained model (or first positional arg)
#   EVAL_GPU         — GPUs for vLLM serving (default: 6,7)
#   EVAL_PORT        — vLLM server port (default: 8020)
#   EVAL_DATASET     — MMSearch dataset (default: MMSearch)
#   NUM_EXAMPLES     — number of samples to eval (default: 20, set "all" for full eval)
#   AGENT_CONFIG     — custom agent config (auto-generated if not set)
#   JUDGE_CONFIG     — judge config (default: configs/judge.yaml)
#   SKIP_SERVE       — set to 1 if vLLM server is already running
#   TP_SIZE          — tensor parallel size (default: auto from GPU count)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Load .env
if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: bash sft/scripts/eval_sft.sh <model_path>"
    echo "  e.g. bash sft/scripts/eval_sft.sh /path/to/sft_model"
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

EVAL_GPU="${EVAL_GPU:-6,7}"
EVAL_PORT="${EVAL_PORT:-8020}"
EVAL_DATASET="${EVAL_DATASET:-MMSearch}"
NUM_EXAMPLES="${NUM_EXAMPLES:-20}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/judge.yaml}"
SKIP_SERVE="${SKIP_SERVE:-0}"
NUM_PROC="${NUM_PROC:-4}"

# Auto-detect TP size from GPU count
IFS=',' read -ra GPU_ARRAY <<< "$EVAL_GPU"
TP_SIZE="${TP_SIZE:-${#GPU_ARRAY[@]}}"

MODEL_NAME="$(basename "$MODEL_PATH")"
SERVED_MODEL_NAME="${MODEL_NAME}"

echo "============================================="
echo "  Post-SFT Evaluation (MMSearch)"
echo "============================================="
echo "  model:        $MODEL_PATH"
echo "  model_name:   $MODEL_NAME"
echo "  gpu:          $EVAL_GPU (tp=$TP_SIZE)"
echo "  port:         $EVAL_PORT"
echo "  dataset:      $EVAL_DATASET"
echo "  num_examples: $NUM_EXAMPLES"
echo ""

# --- Step 1: Start vLLM server ---
VLLM_PID=""
if [[ "$SKIP_SERVE" != "1" ]]; then
    echo "[Step 1] Starting vLLM server..."

    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port "$EVAL_PORT" \
        --model "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --gpu-memory-utilization 0.9 \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len 32768 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        &
    VLLM_PID=$!

    # Wait for server to be ready
    echo "  Waiting for vLLM server (pid=$VLLM_PID) on port $EVAL_PORT..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:${EVAL_PORT}/health" > /dev/null 2>&1; then
            echo "  vLLM server ready."
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM server exited unexpectedly."
            exit 1
        fi
        sleep 5
    done

    if ! curl -s "http://localhost:${EVAL_PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: vLLM server did not start within 10 minutes."
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    echo ""
else
    echo "[Step 1] Skipping vLLM server (SKIP_SERVE=1)"
    echo ""
fi

# Cleanup: kill vLLM on exit
cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "Shutting down vLLM server (pid=$VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- Step 2: Generate agent config ---
EVAL_AGENT_CONFIG="${AGENT_CONFIG:-}"
if [[ -z "$EVAL_AGENT_CONFIG" ]]; then
    echo "[Step 2] Generating agent config..."

    EVAL_CONFIG_DIR="$ROOT_DIR/sft/configs/eval"
    mkdir -p "$EVAL_CONFIG_DIR"
    EVAL_AGENT_CONFIG="${EVAL_CONFIG_DIR}/eval_${MODEL_NAME}.yaml"

    cat > "$EVAL_AGENT_CONFIG" <<YAML
# Auto-generated eval config for SFT model: ${MODEL_NAME}
agent_name: BaseAgentWithTools
agent_config_version: v1_0215

agent_kwargs:
  base: "http://localhost:${EVAL_PORT}"
  key: "empty"
  model: "${SERVED_MODEL_NAME}"
  timeout: 600
  custom_prompt: false
  max_tokens: 2048
  temperature: 0.3
  top_p: 0.95
  repetition_penalty: 1.1
  retry: 3
  verbose: true
  debug_tool_loop: true
  tool_base: "http://localhost:8102"
  tool_choice: "auto"
  tool_list:
    - "Python_Executor_Tool"
    - "Crop_Tool"
    - "Gemini_Google_Search_Tool"
    - "Serper_Google_Search_Tool"
    - "SerpApi_Google_Lens_Tool"
    - "Webpage_Visit_Tool"
    - "LLM_Direct_Generator_Tool"
    - "Get_Tool_Schema"
  prompt_version: "v2"
  reasoning_effort: null
  persistent_state_memory:
    name: "image"
    emit_policy: "always"
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
  max_iterations: 15
YAML

    echo "  Generated: $EVAL_AGENT_CONFIG"
    echo ""
else
    echo "[Step 2] Using provided agent config: $EVAL_AGENT_CONFIG"
    echo ""
fi

# --- Step 3: Run evaluation ---
echo "[Step 3] Running MMSearch evaluation..."
echo ""

NUM_EXAMPLES_ARG=""
if [[ "$NUM_EXAMPLES" != "all" ]]; then
    NUM_EXAMPLES_ARG="--num-examples $NUM_EXAMPLES"
fi

python run_benchmark.py \
    --config "$EVAL_AGENT_CONFIG" \
    --judge-config "$JUDGE_CONFIG" \
    --data "$EVAL_DATASET" \
    --num-proc "$NUM_PROC" \
    --incremental-json \
    --save-html 10 \
    $NUM_EXAMPLES_ARG \
    --mode all

echo ""
echo "============================================="
echo "  Evaluation complete."
echo "  Model: $MODEL_NAME"
echo "  Results saved to outputs/ directory."
echo "============================================="
