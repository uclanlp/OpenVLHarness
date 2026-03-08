#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

# ---------- Data ----------
TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_DIR/rl/data/geo3k_train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$PROJECT_DIR/rl/data/geo3k_test.parquet}"
LOCAL_DATASET_PATH="${LOCAL_DATASET_PATH:-}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
TEST_MAX_SAMPLES="${TEST_MAX_SAMPLES:--1}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

# ---------- Tools ----------
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$PROJECT_DIR/rl/configs/tool_config_geo3k.yaml}"
SKIP_TOOL_CONFIG_GENERATION="${SKIP_TOOL_CONFIG_GENERATION:-1}"

# ---------- Model ----------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"

# ---------- Training ----------
ALGORITHM="${ALGORITHM:-grpo}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-}"

# ---------- GPU / parallelism ----------
N_GPUS="${N_GPUS:-6}"
NNODES="${NNODES:-1}"
TP_SIZE="${TP_SIZE:-1}"

# ---------- Logging ----------
WANDB_PROJECT="${WANDB_PROJECT:-visual-tooluse-grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl2b_geo3k_grpo}"
PPO_DRY_CFG="${PPO_DRY_CFG:-0}"

if [[ "$SKIP_TOOL_CONFIG_GENERATION" != "1" ]]; then
  TOOL_BASE="${TOOL_BASE:-http://127.0.0.1:8101}"
  python -m rl.tools.generate_tool_config \
    --tool-base "$TOOL_BASE" \
    --output "$TOOL_CONFIG_PATH"
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  preprocess_args=(
    --output-train "$TRAIN_PARQUET"
    --output-test "$VAL_PARQUET"
    --train-max-samples "$TRAIN_MAX_SAMPLES"
    --test-max-samples "$TEST_MAX_SAMPLES"
  )
  if [[ -n "$LOCAL_DATASET_PATH" ]]; then
    preprocess_args+=(--local-dataset-path "$LOCAL_DATASET_PATH")
  fi
  python -m rl.data.preprocess_geo3k "${preprocess_args[@]}"
fi

ppo_args=(
  --config-path "$PROJECT_DIR/rl/configs"
  --config-name grpo_training_geo3k
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "data.train_files=$TRAIN_PARQUET"
  "data.val_files=$VAL_PARQUET"
  "trainer.project_name=$WANDB_PROJECT"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "algorithm.adv_estimator=$ALGORITHM"
  "actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH"
  "actor_rollout_ref.rollout.n=$ROLLOUT_N"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "trainer.n_gpus_per_node=$N_GPUS"
  "trainer.nnodes=$NNODES"
)
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
if [[ "$PPO_DRY_CFG" == "1" ]]; then
  ppo_args+=(--cfg job)
fi

python -m verl.trainer.main_ppo "${ppo_args[@]}"
