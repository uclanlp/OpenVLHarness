#!/usr/bin/env bash
set -euo pipefail

# SFT Training via LlamaFactory
#
# Usage:
#   bash sft/scripts/run_sft.sh                                    # default: LoRA on Qwen3-VL-8B
#   bash sft/scripts/run_sft.sh sft/configs/sft_full_qwen3vl_8b.yaml  # full fine-tuning
#   CONFIG=path/to/config.yaml bash sft/scripts/run_sft.sh         # custom config
#
# Environment variables:
#   CONFIG          — path to LlamaFactory YAML config (default: sft/configs/sft_lora_qwen3vl_8b.yaml)
#   TRAJECTORY_INPUT — path to a full_trajectories/ folder for data conversion (optional)
#   SFT_DATA_OUTPUT  — output path for converted SFT data (default: sft/data/sft_train.json)
#   SFT_DATASET_NAME — dataset name in dataset_info.json (default: sft_train)
#   SKIP_CONVERT     — set to 1 to skip data conversion step
#   RESUME           — path to a previous output_dir or checkpoint to resume from
#   NUM_GPUS         — number of GPUs (default: auto-detect)
#   CUDA_VISIBLE_DEVICES — GPU selection

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Load environment variables (.env)
if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    source "$ROOT_DIR/.env"
    set +a
fi

CONFIG="${1:-${CONFIG:-sft/configs/sft_lora_qwen3vl_8b.yaml}}"
TRAJECTORY_INPUT="${TRAJECTORY_INPUT:-}"
SFT_DATA_OUTPUT="${SFT_DATA_OUTPUT:-sft/data/sft_train.json}"
SFT_DATASET_NAME="${SFT_DATASET_NAME:-sft_train}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
RESUME="${RESUME:-}"

# --- Auto-create user-editable copies from *_example templates ---
# The repo ships *_example templates (tracked in git) and gitignores the
# non-example counterparts so each user has a private editable copy.
# On first run, copy any missing template -> editable file. NEVER overwrites
# an existing file.
_maybe_copy_from_example() {
    local target="$1"
    local example="$2"
    if [[ ! -f "$target" && -f "$example" ]]; then
        cp "$example" "$target"
        echo "[setup] Created $target from $example"
        echo "[setup] Edit it to customize."
    fi
}

# Auto-create dataset_info.json (always needed by LlamaFactory).
_maybe_copy_from_example \
    "sft/data/dataset_info.json" \
    "sft/data/dataset_info_example.json"

# Auto-create the requested config from its *_example counterpart, if any.
# Only triggers when CONFIG points at a template-style filename without _example.
if [[ "$CONFIG" == *.yaml && "$CONFIG" != *_example.yaml ]]; then
    _example_candidate="${CONFIG%.yaml}_example.yaml"
    _maybe_copy_from_example "$CONFIG" "$_example_candidate"
fi

# Validate config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo "Hint: copy a template from sft/configs/*_example.yaml to your desired path."
    exit 1
fi

# --- Auto-generate timestamped output_dir ---
# Convention: sft/saves/{model}_{type}_sft_{MMDD}_{HHMM}
# Set SFT_OUTPUT_DIR to override. If unset, derives from config's output_dir + timestamp.
# Set SFT_SAVE_ROOT to change the base directory (default: sft/saves).
SFT_SAVE_ROOT="${SFT_SAVE_ROOT:-sft/saves}"
TIMESTAMP="$(date +%m%d_%H%M)"
if [[ -z "${SFT_OUTPUT_DIR:-}" ]]; then
    # Extract output_dir from YAML config as base name (e.g. "qwen3vl-8b-full-vdr")
    BASE_NAME="$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
base = cfg.get('output_dir', 'sft_run').rsplit('/', 1)[-1]
print(base.replace('-', '_'))
" "$CONFIG" 2>/dev/null || echo "sft_run")"
    SFT_OUTPUT_DIR="${SFT_SAVE_ROOT}/${BASE_NAME}_sft_${TIMESTAMP}"
fi

# --- Resume support ---
# RESUME can be:
#   1. A previous output_dir (auto-detects last checkpoint inside)
#   2. A specific checkpoint path (e.g. .../checkpoint-500)
# When resuming, reuse that output_dir instead of creating a new one.
RESUME_ARGS=""
if [[ -n "$RESUME" ]]; then
    if [[ -d "$RESUME" ]]; then
        # Check if it's a checkpoint dir (has trainer_state.json) or an output_dir
        if [[ -f "$RESUME/trainer_state.json" ]]; then
            # Specific checkpoint dir — output_dir is its parent
            SFT_OUTPUT_DIR="$(dirname "$RESUME")"
            RESUME_ARGS="resume_from_checkpoint=$RESUME"
        else
            # Output dir — let LlamaFactory auto-detect last checkpoint
            SFT_OUTPUT_DIR="$RESUME"
            RESUME_ARGS="overwrite_output_dir=false"
        fi
    else
        echo "ERROR: RESUME path does not exist: $RESUME"
        exit 1
    fi
fi

echo "============================================="
echo "  SFT Training (LlamaFactory)"
echo "============================================="
echo "  config:     $CONFIG"
echo "  output_dir: $SFT_OUTPUT_DIR"
if [[ -n "$RESUME" ]]; then
    echo "  resume:     $RESUME"
fi

# --- Step 1: Data conversion (optional) ---
if [[ -n "$TRAJECTORY_INPUT" && "$SKIP_CONVERT" != "1" ]]; then
    echo "  trajectory: $TRAJECTORY_INPUT"
    echo "  sft output: $SFT_DATA_OUTPUT"
    echo ""
    echo "[Step 1] Converting trajectories → LlamaFactory format..."

    python -m sft.data.convert_trajectories \
        --input "$TRAJECTORY_INPUT" \
        --output "$SFT_DATA_OUTPUT" \
        --filter-success \
        --skip-env-updates

    # Update dataset_info.json with the new dataset entry.
    python3 -c "
import json, os
info_path = 'sft/data/dataset_info.json'
with open(info_path) as f:
    info = json.load(f)
info['${SFT_DATASET_NAME}'] = {
    'file_name': os.path.basename('${SFT_DATA_OUTPUT}'),
    'formatting': 'sharegpt',
    'columns': {'messages': 'conversations', 'images': 'images'}
}
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(f'Updated {info_path} with dataset: ${SFT_DATASET_NAME}')
"
    echo ""
else
    if [[ "$SKIP_CONVERT" == "1" ]]; then
        echo "  [Skipping data conversion]"
    fi
    echo ""
fi

# --- Step 2: Save config snapshot ---
mkdir -p "$SFT_OUTPUT_DIR"
cp "$CONFIG" "$SFT_OUTPUT_DIR/train_config.yaml"
# Append runtime overrides to snapshot
cat >> "$SFT_OUTPUT_DIR/train_config.yaml" <<EOF

### Runtime overrides (auto-appended by run_sft.sh)
# output_dir: $SFT_OUTPUT_DIR
# run_name: $(basename "$SFT_OUTPUT_DIR")
# resume: ${RESUME:-none}
# timestamp: $(date '+%Y-%m-%d %H:%M:%S')
# gpu: ${CUDA_VISIBLE_DEVICES:-all}
EOF
echo "[Step 2] Config snapshot saved to $SFT_OUTPUT_DIR/train_config.yaml"

# --- Step 3: Launch training ---
echo "[Step 3] Starting SFT training..."
echo ""

RUN_NAME="$(basename "$SFT_OUTPUT_DIR")"
llamafactory-cli train "$CONFIG" output_dir="$SFT_OUTPUT_DIR" run_name="$RUN_NAME" $RESUME_ARGS
