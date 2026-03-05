# RL Integration (integrate-verl + verl)

RL training pipeline for visual tool use.

## Setup

Create environment and install dependencies:

```bash
conda create -n tool python=3.12 -y
conda activate tool

# In your locally cloned verl repo:
git checkout 2815eea5d6a5153594dcd71637c423984a268791
pip install -e ".[vllm]"
pip install flash-attn --no-build-isolation

`sglang` and `vllm` currently have incompatible transitive pins in this stack, so install only one backend extra per env (`.[vllm]` or `.[sglang]`, not both).

# Back in this repo (integrate-verl):
pip install -r rl/requirements.txt
```

Optional (only if you need to generate parquet data locally):

```bash
pip install -e .
```


## Running Training

Start the tool server in a separate terminal/env:

```bash
bash mini_vlm_toolkit/tool_infra/scripts/run_server.sh
```

Launch training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATASET_NAME=MMStar \
TRAIN_MAX_SAMPLES=16 \
VAL_MAX_SAMPLES=4 \
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
N_GPUS=4 TP_SIZE=2 ROLLOUT_N=4 \
TRAIN_BATCH_SIZE=4 TOTAL_EPOCHS=1 \
EXPERIMENT_NAME=smoke_test \
SKIP_PREPROCESS=1 \
SKIP_TOOL_CONFIG_GENERATION=1 \
bash rl/scripts/run_grpo.sh
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATASET_NAME` | `MMStar` | Dataset for `build_dataset()` |
| `TRAIN_MAX_SAMPLES` | `-1` (all) | Cap training samples |
| `VAL_MAX_SAMPLES` | `256` | Cap validation samples |
| `SPLIT_SEED` | `42` | Seed for train/val split |
| `MODEL_PATH` | `Qwen/Qwen3-VL-8B-Instruct` | HF model path |
| `ALGORITHM` | `grpo` | `grpo`, `reinforce_plus_plus`, `rloo`, etc. |
| `ROLLOUT_N` | `8` | Rollouts per prompt (GRPO needs >1) |
| `N_GPUS` | `1` | GPUs per node |
| `NNODES` | `1` | Number of nodes |
| `TP_SIZE` | `1` | Tensor parallel size |
| `TRAIN_BATCH_SIZE` | `128` | Training batch size |
| `TOTAL_EPOCHS` | 3 | Training epochs |
| `MAX_PROMPT_LENGTH` | unset | Optional override for `data.max_prompt_length`; by default use value from `rl/configs/grpo_training.yaml` |
| `USE_REMOVE_PADDING` | unset | Optional override for `actor_rollout_ref.model.use_remove_padding`; by default use value from `rl/configs/grpo_training.yaml` |
| `TOOL_CONFIG_PATH` | `rl/configs/tool_config.yaml` | Tool schema file for multi-turn rollout; can point to `rl/configs/tool_config.smoke.yaml` |
| `WANDB_PROJECT` | `visual-tooluse-grpo` | WandB project name |
| `EXPERIMENT_NAME` | `qwen3vl_grpo` | WandB run name |
| `SKIP_PREPROCESS` | `0` | `1` to reuse existing parquets |
| `SKIP_TOOL_CONFIG_GENERATION` | `0` | `1` to reuse existing tool config |



## Configuring Tools

Active tools are defined in `rl/configs/tool_config.yaml` (auto-generated from tool server).
The current curated default is `SAM3_Segment_Tool` only.
To change the set, either edit `CURATED_V1_TOOLS` in `rl/tools/generate_tool_config.py`,
pass `--tools "OCR_Tool,Crop_Tool"` manually, or edit the YAML directly.

## Pinned Versions

| Component | Version |
|---|---|
| verl | `2815eea5` (from source, NOT pip) |
| torch | `2.6.0` (cu124) |
| sglang | `0.5.6` (pinned by verl) |
| vllm | `0.8.5–0.12.0` (pinned by verl) |
