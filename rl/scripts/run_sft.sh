#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

TRAJECTORY_JSONL="${TRAJECTORY_JSONL:-$PROJECT_DIR/outputs/latest/trajectories.jsonl}"
PRED_FILE="${PRED_FILE:-}"
SFT_PARQUET="${SFT_PARQUET:-$PROJECT_DIR/rl/data/sft_train.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

if [[ ! -f "$TRAJECTORY_JSONL" ]]; then
  echo "Missing trajectory file: $TRAJECTORY_JSONL" >&2
  exit 1
fi

collect_args=(
  --trajectory-jsonl "$TRAJECTORY_JSONL"
  --output "$SFT_PARQUET"
  --require-correct
)
if [[ -n "$PRED_FILE" ]]; then
  collect_args+=(--pred-file "$PRED_FILE")
fi
python -m rl.data.collect_sft_data "${collect_args[@]}"

python -m verl.trainer.sft_trainer \
  model.path="$MODEL_PATH" \
  data.train_files="$SFT_PARQUET" \
  data.val_files=null \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  data.max_length=8192 \
  data.pad_mode=no_padding \
  data.truncation=error \
  trainer.project_name=visual-tooluse-sft \
  trainer.experiment_name=qwen3vl_toolcall_sft \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1
