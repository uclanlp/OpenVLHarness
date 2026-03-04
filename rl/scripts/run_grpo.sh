#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

TOOL_BASE="${TOOL_BASE:-http://127.0.0.1:8101}"
DATASET_NAME="${DATASET_NAME:-MMStar}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$PROJECT_DIR/rl/data/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$PROJECT_DIR/rl/data/val.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
WANDB_PROJECT="${WANDB_PROJECT:-visual-tooluse-grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl_grpo}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-256}"
SKIP_TOOL_CONFIG_GENERATION="${SKIP_TOOL_CONFIG_GENERATION:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
PPO_DRY_CFG="${PPO_DRY_CFG:-0}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-}"

if [[ "$SKIP_TOOL_CONFIG_GENERATION" != "1" ]]; then
  python -m rl.tools.generate_tool_config \
    --tool-base "$TOOL_BASE" \
    --output "$PROJECT_DIR/rl/configs/tool_config.yaml"
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  preprocess_train_args=(
    --dataset "$DATASET_NAME"
    --output "$TRAIN_PARQUET"
  )
  if [[ "$TRAIN_MAX_SAMPLES" != "-1" ]]; then
    preprocess_train_args+=(--max-samples "$TRAIN_MAX_SAMPLES")
  fi
  python -m rl.data.preprocess "${preprocess_train_args[@]}"

  preprocess_val_args=(
    --dataset "$DATASET_NAME"
    --output "$VAL_PARQUET"
  )
  if [[ "$VAL_MAX_SAMPLES" != "-1" ]]; then
    preprocess_val_args+=(--max-samples "$VAL_MAX_SAMPLES")
  fi
  python -m rl.data.preprocess "${preprocess_val_args[@]}"
fi

ppo_args=(
  # Use an absolute config path. Hydra resolves relative config paths from the
  # module package (`verl/trainer`), not from the shell CWD.
  --config-path "$PROJECT_DIR/rl/configs"
  --config-name grpo_training
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "data.train_files=$TRAIN_PARQUET"
  "data.val_files=$VAL_PARQUET"
  "trainer.project_name=$WANDB_PROJECT"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
)
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
