#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

# ---------- Model size ----------
MODEL_SIZE="${MODEL_SIZE:-8b}"   # 2b | 4b | 8b
MODEL_SIZE_UPPER="${MODEL_SIZE^^}"  # 2B | 4B | 8B
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-${MODEL_SIZE_UPPER}-Instruct}"

# ---------- Config ----------
CONFIG_NAME="grpo_training_fsc147_${MODEL_SIZE}"

# ---------- Data ----------
LMU_DATA_DIR="${LMU_DATA_DIR:-/data/<user>/lmu_data}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_DIR/rl/data/fsc147_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$PROJECT_DIR/rl/data/fsc147_val.parquet}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-50}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

# ---------- Tools ----------
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$PROJECT_DIR/rl/configs/tool_config.yaml}"
SKIP_TOOL_CONFIG_GENERATION="${SKIP_TOOL_CONFIG_GENERATION:-1}"

# ---------- Training ----------
ALGORITHM="${ALGORITHM:-grpo}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-}"

# ---------- GPU / parallelism ----------
N_GPUS="${N_GPUS:-8}"
NNODES="${NNODES:-1}"
TP_SIZE="${TP_SIZE:-}"

# ---------- Logging ----------
WANDB_PROJECT="${WANDB_PROJECT:-visual-tooluse-grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl${MODEL_SIZE}_fsc147_grpo}"
PPO_DRY_CFG="${PPO_DRY_CFG:-0}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"

# ---------- Stage 1: Generate tool config ----------
if [[ "$SKIP_TOOL_CONFIG_GENERATION" != "1" ]]; then
  TOOL_BASE="${TOOL_BASE:-http://127.0.0.1:8101}"
  python -m rl.tools.generate_tool_config \
    --tool-base "$TOOL_BASE" \
    --output "$TOOL_CONFIG_PATH"
fi

# ---------- Stage 2: Preprocess FSC147 TSV → parquet ----------
if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  preprocess_args=(
    --lmu-data-dir "$LMU_DATA_DIR"
    --output-train "$TRAIN_PARQUET"
    --output-val "$VAL_PARQUET"
    --val-max-samples "$VAL_MAX_SAMPLES"
  )
  if [[ "$TRAIN_MAX_SAMPLES" != "-1" ]]; then
    preprocess_args+=(--train-max-samples "$TRAIN_MAX_SAMPLES")
  fi
  python -m rl.data.preprocess_fsc147 "${preprocess_args[@]}"
fi

# ---------- Stage 3: verl GRPO training ----------
ppo_args=(
  --config-path "$PROJECT_DIR/rl/configs"
  --config-name "$CONFIG_NAME"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "data.train_files=$TRAIN_PARQUET"
  "data.val_files=$VAL_PARQUET"
  "trainer.project_name=$WANDB_PROJECT"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.resume_mode=$RESUME_MODE"
  "algorithm.adv_estimator=$ALGORITHM"
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH"
  "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "trainer.n_gpus_per_node=$N_GPUS"
  "trainer.nnodes=$NNODES"
)
if [[ -n "$TP_SIZE" ]]; then
  ppo_args+=("actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE")
fi
if [[ -n "$USE_REMOVE_PADDING" ]]; then
  ppo_args+=("actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING")
fi
if [[ -n "$MAX_PROMPT_LENGTH" ]]; then
  ppo_args+=("data.max_prompt_length=$MAX_PROMPT_LENGTH")
fi
if [[ -n "$TOTAL_TRAINING_STEPS" ]]; then
  ppo_args+=("trainer.total_training_steps=$TOTAL_TRAINING_STEPS")
fi
if [[ -n "$TOTAL_EPOCHS" ]]; then
  ppo_args+=("trainer.total_epochs=$TOTAL_EPOCHS")
fi
if [[ -n "$RESUME_FROM_PATH" ]]; then
  ppo_args+=("trainer.resume_from_path=$RESUME_FROM_PATH")
fi
if [[ "$PPO_DRY_CFG" == "1" ]]; then
  ppo_args+=(--cfg job)
fi

python -m verl.trainer.main_ppo "${ppo_args[@]}" "$@"
