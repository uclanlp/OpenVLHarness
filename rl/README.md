# RL Integration (integrate-verl + verl)

This folder contains the RL training bridge for visual tool use:

1. `scripts/run_sft.sh` collects trajectory-based SFT data and runs `verl` SFT trainer.
2. `scripts/run_grpo.sh` generates tool config, preprocesses dataset rows, and runs GRPO with multi-turn Hermes format.
3. `scripts/run_eval.sh` runs benchmark evaluation against a served checkpoint.

## Key modules

- `tools/tool_server_adapter.py`: `BaseTool` adapter that forwards calls to `tool_infra` `/execute`.
- `tools/generate_tool_config.py`: builds `configs/tool_config.yaml` from tool server `/tools` + `/jsonschema`.
- `rewards/compute_score.py`: lightweight online reward.
- `data/preprocess.py`: benchmark dataset -> RL parquet conversion.
- `data/collect_sft_data.py`: trajectory JSONL -> SFT parquet conversion.

## Notes

- Train-time tool calling uses Hermes `<tool_call>...</tool_call>` format.
- Inference-time native OpenAI `tool_calls` remains unchanged when serving with a Hermes tool parser.
- For `main_ppo`, prefer `scripts/run_grpo.sh` (it passes an absolute `--config-path` to avoid Hydra relative-path resolution issues).
