# SFT — Supervised Fine-Tuning

SFT module for mini-vlm-toolkit, powered by [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory).

## Setup

```bash
pip install llamafactory
```

## File layout: `_example` templates

User-customizable files (training configs, dataset registry) ship as
`_example` templates in git. On the first run of `bash sft/scripts/run_sft.sh`,
any missing non-example counterpart is auto-created from its template.

| Template (in git) | Local file (auto-created on first run, gitignored) |
|-------------------|---------------------------------------------------|
| `sft/configs/sft_lora_qwen3vl_8b_example.yaml` | `sft/configs/sft_lora_qwen3vl_8b.yaml` |
| `sft/configs/sft_full_qwen3vl_8b_example.yaml` | `sft/configs/sft_full_qwen3vl_8b.yaml` |
| `sft/data/dataset_info_example.json` | `sft/data/dataset_info.json` |

Edit the auto-created files to set hyperparameters, register your datasets,
etc. Your edits won't be overwritten by future updates because the local
copies are gitignored. The templates themselves stay clean for everyone.

## Data Paths & .env

**Important:** Data paths and credentials are managed via the `.env` file at the project root. `run_sft.sh` automatically loads `.env` on startup.

Copy `sft/.env.example` to the project root as `.env` and fill in your values:

```bash
cp sft/.env.example .env
# Edit .env with your paths and API keys
```

Required variables:

```bash
SFT_DATA_ROOT=/path/to/your/sft/training/data   # SFT training data root
LMUData=/path/to/your/evaluation/data            # Evaluation data root

# WandB — each user MUST set their own API key
WANDB_API_KEY=your_key_here          # Get your key at https://wandb.ai/authorize
WANDB_PROJECT=visual-tooluse-sft     # WandB project name
# WANDB_ENTITY=your_team_or_username # Optional: WandB team or username
```

> **Important:** Configs with `report_to: wandb` will upload training logs to WandB. **Each user must set their own `WANDB_API_KEY` in `.env`**, otherwise training will prompt for login or fail. Set `report_to: none` in the YAML config to disable WandB.

Optional variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SFT_SAVE_ROOT` | `sft/saves` | Base directory for model checkpoints |
| `SFT_OUTPUT_DIR` | auto-generated | Override full output path |

When registering datasets in `sft/data/dataset_info.json`, use absolute paths for data outside `sft/data/`:

```json
"my_dataset": {
  "file_name": "/absolute/path/to/my_data.jsonl",
  ...
}
```

> **Note:** `dataset_info.json` is static JSON and does not support environment variable substitution. When switching machines, update the paths in `dataset_info.json` accordingly.

## Quick Start

### 1. Convert trajectories to SFT data

```bash
python -m sft.data.convert_trajectories \
    --input outputs/.../full_trajectories \
    --output sft/data/my_dataset.json \
    --filter-success --success-criteria exact_match
```

`--input` must be a `full_trajectories/` folder produced by
`run_benchmark.py` (one `<task_id>.json` per sample). With `--filter-success`,
pass `--success-criteria` to choose how `prediction` is compared against
`ground_truth`:

| Criterion | Status |
|-----------|--------|
| `exact_match` (default) | implemented — string-equal after strip |
| `llm_judge` | not yet implemented (raises `NotImplementedError`) |
| `mean_iou_point_5` | not yet implemented (raises `NotImplementedError`) |

Run `run_benchmark.py` with `--save-images files` (the default) so each user
and tool image lives at `full_trajectories/images/<task_id>_<idx>.<ext>`. The
converter resolves those paths, prepends one `<image>` placeholder per image
to the originating user message, and appends the resolved paths to the
ShareGPT `images` list in the same order — this is the alignment LlamaFactory
expects for multimodal training. Folders produced with `--save-images base64`
will load but their inline data URIs are skipped (no `images` field), so
prefer `files` for SFT.

### 2. Register dataset

`sft/data/dataset_info.json` is auto-created from `dataset_info_example.json`
on first run. Open it and add an entry for your dataset:

```json
{
  "my_dataset": {
    "file_name": "my_dataset.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "images"}
  }
}
```

If your data uses OpenAI format (`role`/`content` instead of ShareGPT's `from`/`value`), add `tags`:

```json
{
  "my_dataset": {
    "file_name": "/path/to/data.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system",
      "observation_tag": "tool"
    }
  }
}
```

Then update your training YAML to reference the dataset by name:
`dataset: my_dataset`.

### 3. Train

```bash
# LoRA (default, single GPU)
bash sft/scripts/run_sft.sh

# Full fine-tuning (multi-GPU recommended)
bash sft/scripts/run_sft.sh sft/configs/sft_full_qwen3vl_8b.yaml

# Custom config
bash sft/scripts/run_sft.sh path/to/your_config.yaml

# End-to-end: convert + train
TRAJECTORY_INPUT=outputs/.../full_trajectories bash sft/scripts/run_sft.sh

# Save checkpoints to a custom directory
SFT_SAVE_ROOT=/data/models bash sft/scripts/run_sft.sh
```

Output directory is auto-generated with timestamps (e.g. `sft/saves/qwen3vl_8b_lora_sft_0402_1503/`). A config snapshot is saved alongside the checkpoint for reproducibility.

> **Note:** LlamaFactory does not use the `-c` flag. The YAML config file is passed directly as the first positional argument.

### 4. Resume Training

Resume from a checkpoint after training is interrupted:

```bash
# Pass output_dir — auto-detects the last checkpoint
RESUME=/path/to/previous_output_dir \
  bash sft/scripts/run_sft.sh sft/configs/sft_full_qwen3vl_8b.yaml

# Pass a specific checkpoint path
RESUME=/path/to/previous_output_dir/checkpoint-500 \
  bash sft/scripts/run_sft.sh sft/configs/sft_full_qwen3vl_8b.yaml
```

### 5. Evaluate

Run MMSearch evaluation after training to verify SFT effectiveness:

```bash
# Quick validation (20 samples, auto-starts vLLM -> eval -> shutdown)
bash sft/scripts/eval_sft.sh /path/to/trained_model

# Specify GPUs and sample count
EVAL_GPU=0,1 NUM_EXAMPLES=50 bash sft/scripts/eval_sft.sh /path/to/model

# Full evaluation
NUM_EXAMPLES=all bash sft/scripts/eval_sft.sh /path/to/model

# Skip vLLM startup if already running
SKIP_SERVE=1 EVAL_PORT=8020 bash sft/scripts/eval_sft.sh /path/to/model
```

The script automatically: starts vLLM server -> generates agent config -> runs MMSearch -> shuts down vLLM on exit.

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_GPU` | `6,7` | GPUs for vLLM serving |
| `EVAL_PORT` | `8020` | vLLM server port |
| `EVAL_DATASET` | `MMSearch` | Evaluation dataset |
| `NUM_EXAMPLES` | `20` | Number of samples (`all` for full eval) |
| `SKIP_SERVE` | `0` | Set to `1` to skip vLLM startup |
| `TP_SIZE` | auto | Tensor parallel size |

## Data Formats

### From trajectories (convert_trajectories.py)

Converts a `full_trajectories/` folder -> LlamaFactory ShareGPT format:
- System/user/assistant/tool messages are mapped to `system/human/gpt/tool` roles
- Tool calls are serialized as `<tool_call>` XML blocks
- Each user message that carried images gets one `<image>` placeholder
  prepended per image; resolved paths (from the sibling `images/` folder)
  are appended to the sample's `images` list in the same order
- Environment Update messages are stripped by default
- `--filter-success` keeps only correct samples; `--success-criteria`
  selects the comparator (default `exact_match`)

### External datasets

Place any LlamaFactory-compatible dataset (ShareGPT, Alpaca, OpenAI format) in `sft/data/`
and register it in `dataset_info.json`. See [LlamaFactory docs](https://llamafactory.readthedocs.io/) for format details.

## Configs

| Template (in git) | Description |
|-------------------|-------------|
| `sft_lora_qwen3vl_8b_example.yaml` | LoRA SFT on Qwen3-VL-8B (default; first run auto-creates `sft_lora_qwen3vl_8b.yaml`) |
| `sft_full_qwen3vl_8b_example.yaml` | Full fine-tuning on Qwen3-VL-8B (first run auto-creates `sft_full_qwen3vl_8b.yaml`) |

Key config fields:
- `model_name_or_path` — HuggingFace model ID or local path
- `template` — chat template (`qwen3_vl` for Qwen3-VL, `qwen3_nothink` to disable thinking, `llama3` for LLaMA)
- `dataset` — comma-separated dataset names from `dataset_info.json`
- `dataset_dir` — directory containing `dataset_info.json` (default: `sft/data`)
- `finetuning_type` — `lora` or `full`
- `deepspeed` — DeepSpeed config path (e.g. `sft/configs/deepspeed/ds_z3_config.json`); recommended for full fine-tuning on multi-GPU
- `output_dir` — base name for checkpoints (timestamped directory is auto-generated by `run_sft.sh`)
- `report_to` — `wandb` or `none`

## GPU Requirements

| Method | Minimum GPUs | Recommended | Notes |
|--------|-------------|-------------|-------|
| LoRA (8B) | 1x 24GB | 1x 48GB | Single GPU sufficient |
| Full tuning (8B) | 2x 48GB | 4x 48GB | Requires DeepSpeed ZeRO-3 |
| Full tuning + offload | 2x 24GB | 2x 48GB | Slower, uses CPU RAM via `ds_z3_offload_config.json` |

## Directory Structure

```
sft/
├── .env.example                          # Template for environment variables
├── configs/
│   ├── deepspeed/                        # DeepSpeed configs (ZeRO-3, ZeRO-3 + offload)
│   ├── eval/                             # Auto-generated eval configs (gitignored)
│   ├── sft_lora_qwen3vl_8b_example.yaml  # template (tracked)
│   └── sft_full_qwen3vl_8b_example.yaml  # template (tracked)
├── data/
│   ├── dataset_info_example.json         # template (tracked)
│   ├── convert_trajectories.py           # trajectory -> ShareGPT SFT data
│   └── collect_experience_trajectories.py  # gather correct trajectories across experience-learning trials
├── scripts/
│   ├── run_sft.sh                        # Training launcher (loads .env, auto-copies templates)
│   └── eval_sft.sh                       # Post-training evaluation (vLLM + MMSearch)
├── saves/                                # Training outputs (checkpoints, gitignored)
└── README.md
```
