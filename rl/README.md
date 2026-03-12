# RL Integration (integrate-verl + verl)

RL training pipeline for visual tool use.

## Setup

Create environment and install dependencies:

```bash
conda create -n tool python=3.12 -y
conda activate tool

# In your locally cloned verl repo:
git clone https://github.com/tool-vision/verl.git && cd verl
pip install -e ".[vllm]"
pip install flash-attn --no-build-isolation

```

Optional (only if you need to generate parquet data locally):

```bash
# Back in this repo:
pip install -e .
```


## Running Training

Start the tool server in a separate terminal/env, then launch training (example with FSC147 dataset on 8 A100 GPUs):

```bash
MODEL_SIZE=2b \
SKIP_TOOL_CONFIG_GENERATION=1 \
SKIP_PREPROCESS=1 \
TRAIN_PARQUET=rl/data/fsc147_train.parquet \
VAL_PARQUET=rl/data/fsc147_val.parquet \
bash rl/scripts/run_grpo_fsc147.sh
```

Use `MODEL_SIZE=4b` or `MODEL_SIZE=8b` to switch model variants (configs exist for all three).

### Data

FSC147 is a visual object-counting dataset (3,659 train / 1,286 val images). On any <cluster> server the data is already available at:

```
/data/<user>/lmu_data/FSC147_train.tsv
/data/<user>/lmu_data/FSC147_val.tsv
/data/<user>/lmu_data/images/FSC147/
```

To preprocess the TSVs into parquet (only needed once, or if the parquets don't exist):

```bash
LMUData=/data/<user>/lmu_data \
/data/<shared>/conda_envs/sam3/bin/python -m rl.data.preprocess_fsc147 \
  --lmu-data-dir /data/<user>/lmu_data \
  --output-train rl/data/fsc147_train.parquet \
  --output-val   rl/data/fsc147_val.parquet \
  --val-max-samples 50
```

Notes:
- `run_grpo_fsc147.sh` will run preprocessing automatically unless `SKIP_PREPROCESS=1`.
- Set `LMU_DATA_DIR` to override the default data path if running on a different server.
- GPU memory is tight on A100s if other processes are running; pass
  `actor_rollout_ref.rollout.gpu_memory_utilization=0.40` if you see OOM on startup.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `8b` | Model variant: `2b`, `4b`, or `8b` |
| `MODEL_PATH` | `Qwen/Qwen3-VL-{SIZE}-Instruct` | HF model path (auto-derived from `MODEL_SIZE`) |
| `LMU_DATA_DIR` | `/data/<user>/lmu_data` | Root directory containing FSC147 TSVs and images |
| `TRAIN_PARQUET` | `rl/data/fsc147_train.parquet` | Preprocessed train parquet path |
| `VAL_PARQUET` | `rl/data/fsc147_val.parquet` | Preprocessed val parquet path |
| `TRAIN_MAX_SAMPLES` | `-1` (all) | Cap training samples |
| `VAL_MAX_SAMPLES` | `50` | Number of val samples used during training |
| `ALGORITHM` | `grpo` | `grpo`, `reinforce_plus_plus`, `rloo`, etc. |
| `ROLLOUT_N` | `4` | Rollouts per prompt |
| `N_GPUS` | `8` | GPUs per node |
| `NNODES` | `1` | Number of nodes |
| `TP_SIZE` | unset | Tensor parallel size (defaults to value in config) |
| `TRAIN_BATCH_SIZE` | `32` | Training batch size |
| `TOTAL_EPOCHS` | `3` | Training epochs |
| `MAX_PROMPT_LENGTH` | unset | Optional override for `data.max_prompt_length` |
| `USE_REMOVE_PADDING` | unset | Optional override for `actor_rollout_ref.model.use_remove_padding` |
| `TOOL_CONFIG_PATH` | `rl/configs/tool_config.yaml` | Tool schema file for multi-turn rollout |
| `WANDB_PROJECT` | `visual-tooluse-grpo` | WandB project name |
| `EXPERIMENT_NAME` | `qwen3vl{SIZE}_fsc147_grpo` | WandB run name (auto-derived from `MODEL_SIZE`) |
| `SKIP_PREPROCESS` | `0` | `1` to reuse existing parquets |
| `SKIP_TOOL_CONFIG_GENERATION` | `1` | `1` to reuse existing tool config |
| `RESUME_MODE` | `auto` | verl resume mode |
| `RESUME_FROM_PATH` | unset | Path to checkpoint to resume from |



## Configuring Tools

Active tools are defined in `rl/configs/tool_config.yaml` (auto-generated from tool server).
The current curated default is `SAM3_Segment_Tool` only.
To change the set, either edit `CURATED_V1_TOOLS` in `rl/tools/generate_tool_config.py`,
pass `--tools "OCR_Tool,Crop_Tool"` manually, or edit the YAML directly.

## Pinned Versions

| Component | Version |
|---|---|
| verl | `6dc2d23` (from source, NOT pip) |
| torch | `2.6.0` (cu124) |
| sglang | `0.5.6` (pinned by verl) |
| vllm | `0.8.5–0.12.0` (pinned by verl) |
