# Model services

Each configuration names its main model (the orchestrator) by an alias from the
[model registry](../openvlharness/services/model_registry.yaml). The alias fixes the
exact model ID sent to the server, the API protocol, the system prompt and the
accepted request options. Every preset uses **Qwen3-VL-8B-Instruct for multimodal
search summarization and webpage extraction**, whatever the main model. With the 8B
preset, the main and auxiliary roles can share one server; any other main model needs
a separate Qwen3-VL-8B-Instruct server for the auxiliary role.

## Paper presets

One preset per backbone in the paper, with the paper's decoding settings:

| Preset | Config alias | Exact model ID sent to the server | API | Request settings |
| --- | --- | --- | --- | --- |
| [qwen3_8b_paper.yaml](../configs/agent/qwen3_8b_paper.yaml) | `qwen3-vl-8b-instruct` | `Qwen/Qwen3-VL-8B-Instruct` | Chat Completions | temperature 0.7, top_p 0.8, top_k 20, repetition penalty 1.0, presence penalty 1.5, 12800 tokens |
| [qwen3_32b_paper.yaml](../configs/agent/qwen3_32b_paper.yaml) | `qwen3-vl-32b-instruct` | `Qwen/Qwen3-VL-32B-Instruct` | Chat Completions | same as 8B |
| [gpt5_paper.yaml](../configs/agent/gpt5_paper.yaml) | `gpt-5-2025-08-07` | `gpt-5-2025-08-07` | Responses | medium reasoning, 16000 tokens |
| [gpt6_luna_paper.yaml](../configs/agent/gpt6_luna_paper.yaml) | `gpt-6-luna` | `gpt-6-luna` | Responses | medium reasoning, 16000 tokens |
| [gpt6_sol_paper.yaml](../configs/agent/gpt6_sol_paper.yaml) | `gpt-6-sol` | `gpt-6-sol` | Responses | medium reasoning, 16000 tokens |
| [kimi_k3_paper.yaml](../configs/agent/kimi_k3_paper.yaml) | `kimi-k3` | `kimi-k3` | Chat Completions | temperature 0.6, top_p 0.95, 65536 tokens |

The vLLM served name must match the exact model ID, including capitalization;
the shorter config alias is not the served name.

## Other registered models

These aliases are registered with the same contract as the preset of their
family:

| Family | Aliases | Start from |
| --- | --- | --- |
| Qwen3-VL Instruct | `qwen3-vl-2b-instruct`, `qwen3-vl-4b-instruct`, `qwen3-vl-8b-instruct`, `qwen3-vl-32b-instruct`, `qwen3-vl-30b-a3b-instruct`, `qwen3-vl-235b-a22b-instruct`, and each with an `-fp8` suffix for the official FP8 builds | `qwen3_8b_paper.yaml` |
| OpenAI GPT | `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.1`, `gpt-5.2`, `gpt-5.2-pro`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.4-pro`, `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, `gpt-6-luna`, `gpt-6-sol`, `gpt-6-astra`, `gpt-6.1-sol`, and the paper's snapshot `gpt-5-2025-08-07` | `gpt6_luna_paper.yaml` |
| Kimi | `kimi-k3` | `kimi_k3_paper.yaml` |

To use one, copy the family's preset and change the orchestrator's `alias`; leave
the auxiliary model on `qwen3-vl-8b-instruct`:

```bash
sed 's/alias: gpt-6-luna/alias: gpt-6.1-sol/' configs/agent/gpt6_luna_paper.yaml > configs/agent/gpt61_sol.yaml
```

Options outside the family's list (for example `top_k` on a GPT model) fail when
the configuration loads. For the GPT models, keep `reasoning_effort` set: GPT-5.1,
5.2 and 5.4 default to no reasoning, and GPT-6 Astra and GPT-6.1 Sol reject
`none`. The `-pro` variants cost far more per token. `gpt-5-pro` is not registered
because it accepts only `high` effort, while the coding generator requests
`medium`. OpenAI lists `gpt-5`, `gpt-5-mini`, `gpt-5-nano` and the
`gpt-5-2025-08-07` snapshot (used by `gpt5_paper.yaml`) for shutdown on December
11, 2026, and `gpt-5.1` and `gpt-5.4-nano` on April 1, 2027; see OpenAI's
[deprecations page](https://developers.openai.com/api/docs/deprecations).

The paper ran the six preset models; the other aliases are registered from the
providers' published model IDs and checked offline (request construction and
configuration loading), not benchmarked. To add a model outside these families,
register its served ID, API protocol, prompt bundle and supported options in the
registry.

## Output budget and context

When a model service omits `max_tokens`, the adapter uses 4096 for both local
context reservation and the provider's output limit (`max_tokens` for Chat
Completions, `max_output_tokens` for Responses). Explicit limits and the paper
presets retain their configured budgets.

When `agent.max_context_length` is set, it must be greater than the orchestrator's
effective `max_tokens` plus a 512-token margin. Configurations with no remaining
input budget fail during loading, before tokenizer loading or inference. Increase
the context window or lower `max_tokens` to resolve the error. Omitting `max_context_length` or setting it to
`null` leaves context enforcement to the provider.

## Raw trace export (Qwen only)

Pass `--save-raw-trace` to `openvlharness-eval` with the `qwen3-vl-8b-instruct`
profile. GPT, Kimi and other profiles are rejected before their inference begins. The flag enables
`services.models.<orchestrator-service>.return_token_ids: true` in the effective
configuration saved with the run and included in its resume identity. The normal
paper preset remains `false`; there is no `extra_body` configuration to edit.

```bash
openvlharness-eval --data FSC147_val \
  --config configs/agent/qwen3_8b_paper.yaml \
  --judge-config configs/judge_no_judge.yaml \
  --num-examples 1 --num-proc 1 --save-raw-trace --save-html all
```

Install `openvlharness[qwen]` and use the matching tokenizer/model snapshot from
the serving setup below. Raw export pins tokenizer revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` in the model profile and requires a
vLLM-compatible endpoint that returns `prompt_token_ids` and
`choices[0].token_ids`. Missing tokenizer dependencies or files cause a startup
error.

The terminating response's tokens reconstruct Qwen ChatML, including its rendered
tool calls and tool responses. Image bytes come from that response's exact request,
after any context compaction or fallback. Successful exports populate
`trajectory.raw_trace` and write `raw_traces/<task_id>.json` with relative image
references under `raw_traces/images/`. This format is separate from the
ordinary full-trajectory JSON and HTML reports.

If token IDs are missing, decoding fails, or image alignment is invalid, the agent
records `trajectory.raw_trace_error`. The writer logs an error and saves
`raw_traces/errors/<task_id>.json`. Predictions and normal reports remain
available. `_summary.json` reports exports and failures for the writer's latest
invocation; the per-sample error files persist across subsequent samples. Check
those errors before using an export for training. Resume requires the same
effective token-ID setting; it cannot add raw exports to an ordinary partial run.

## Verified Qwen serving setup

Tested setup: vLLM **0.18.1**, PyTorch **2.10.0**, Transformers **4.57.3**,
OpenAI Python **2.6.0** and huggingface-hub **0.35.3**, serving model revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` on **two NVIDIA RTX A6000 48 GB GPUs**.
A fresh installation and smaller GPU configurations have not been validated.
Keep the serving environment separate from the harness and heavy perception
services.

Activate that serving environment, select a new dedicated cache on a fast local
disk, and download the pinned snapshot before enabling offline mode. The harness
loads its tokenizer by model ID, so the cache's `refs/main` must resolve to the
same pinned revision; downloading by commit alone does not create that reference.

```bash
export MODEL_CACHE_ROOT=/path/to/local-disk/model-cache
export HF_HOME="$MODEL_CACHE_ROOT/huggingface"
export QWEN_SNAPSHOT="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
snapshot = Path(snapshot_download(
    "Qwen/Qwen3-VL-8B-Instruct",
    revision=revision,
    cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
    max_workers=8,
))
assert snapshot.name == revision
assert (snapshot / "tokenizer.json").is_file()
assert (snapshot / "tokenizer_config.json").is_file()
# Pin model-ID tokenizer lookup in this dedicated cache to the downloaded commit.
main_ref = snapshot.parent.parent / "refs" / "main"
main_ref.parent.mkdir(parents=True, exist_ok=True)
main_ref.write_text(revision)
PY

export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$MODEL_CACHE_ROOT/vllm"
export TORCH_HOME="$MODEL_CACHE_ROOT/torch"
export TRITON_CACHE_DIR="$MODEL_CACHE_ROOT/triton"
export XDG_CACHE_HOME="$MODEL_CACHE_ROOT/xdg"
export TMPDIR="$MODEL_CACHE_ROOT/tmp"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
mkdir -p "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

python -m vllm.entrypoints.openai.api_server \
  --model "$QWEN_SNAPSHOT" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port 8222 \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 168000 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-num-seqs 4 --max-num-batched-tokens 16384 \
  --enforce-eager --limit-mm-per-prompt '{"video":0}' \
  > "$MODEL_CACHE_ROOT/qwen-server.log" 2>&1
```

The launch runs in the foreground. From another terminal, check
`curl --fail http://127.0.0.1:8222/v1/models` and verify the exact served ID above.
The tested service passed text, image, native tool-call/result continuation, and
multimodal summarization probes. Keep the 168000 server context aligned with
`agent.max_context_length` in the Qwen preset. The harness can use the same
`HF_HOME` for its tokenizer cache; install the harness's `qwen` extra to enable
local context counting.

### Other Qwen3-VL sizes

Serve another Qwen3-VL Instruct model with the same command, replacing the
snapshot, `--served-model-name` and the alias in the configuration with the
model's entry from the registry (for example `Qwen/Qwen3-VL-32B-Instruct` and
`qwen3-vl-32b-instruct`). Keep a Qwen3-VL-8B-Instruct server running for the
auxiliary role and point `AUXILIARY_MODEL_BASE_URL` at it. Only the 8B serving
setup above was validated; size `--tensor-parallel-size` so the weights and the
168000-token KV cache fit. The bf16 weights take about 2 bytes per parameter
(roughly 4 GB for 2B, 9 GB for 4B, 17 GB for 8B, 62 GB for 30B-A3B, 67 GB for 32B
and 470 GB for 235B-A22B); the FP8 builds take about half. On smaller GPUs, lower
`--max-model-len` and the preset's `agent.max_context_length` together.

## Connect the presets

Set these variables in the processes that run the harness and tool router. For
the Qwen preset and the unauthenticated local server shown above:

```bash
export MODEL_BASE_URL=http://127.0.0.1:8222/v1
export MODEL_API_KEY=empty
export AUXILIARY_MODEL_BASE_URL=http://127.0.0.1:8222/v1
export AUXILIARY_MODEL_API_KEY=empty
```

For the GPT presets, keep the auxiliary variables pointing to Qwen and set:

```bash
export MODEL_BASE_URL=https://api.openai.com/v1
export MODEL_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment}"
```

For Kimi K3 on Moonshot's API, keep the auxiliary variables pointing to Qwen and set:

```bash
export MODEL_BASE_URL=https://api.moonshot.ai/v1
export MODEL_API_KEY="${MOONSHOT_API_KEY:?Set MOONSHOT_API_KEY in your environment}"
```

Each preset reads the credential from `MODEL_API_KEY`.

Model bases accept either a bare service base, such as
`http://127.0.0.1:8222`, or a base ending in `/v1`; the adapter constructs the
profile's `/v1/chat/completions` or `/v1/responses` endpoint. A full endpoint is
also accepted when it matches the profile's protocol. The GPT presets send
`store=false`, medium reasoning, and an output budget of 16000 through Responses.
Access to the exact model ID must be available on the chosen endpoint;
`reasoning_effort` accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh` and
`max`, and each model rejects the levels it does not support.

Kimi K3 always reasons, and its reasoning tokens count against `max_tokens`; the
paper runs used 65536 because smaller budgets lost answers to reasoning. The paper
used an OpenAI-compatible provider that accepted temperature 0.6 and top_p 0.95.
Moonshot documents both as fixed, so remove them from the preset if your endpoint
rejects them. Set `reasoning_effort` (`low`, `high` or `max`) in the orchestrator
service to change the provider default. Tool-call turns are sent back with their
`reasoning_content`, as Moonshot requires.

Tool server addresses and search/storage credentials are configured separately;
these model variables alone do not start the tool services.

## Per-role settings

| Role | Output-token budget | Request settings | Timeout / additional retries |
| --- | ---: | --- | --- |
| Qwen orchestrator | 12800 | temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5, repetition penalty 1.0 | 600 s / 8 |
| GPT orchestrator (GPT-5, Luna, Sol) | 16000 | medium reasoning, Responses | 600 s / 8 |
| Kimi K3 orchestrator | 65536 | temperature 0.6, top_p 0.95, Chat Completions | 1800 s / 8 |
| Qwen text/image search summarizer | 4096 | temperature 0, top_p 1, thinking disabled; multimodal messages for image-aware summaries | 600 s / 2 |
| Qwen webpage extractor | 16384 | temperature 0, top_p 1, thinking disabled | 600 s / 2 |
| Coding generator | 4096 | same model and native protocol as orchestrator; fresh prompt; temperature 0 on Chat Completions (Qwen, Kimi) or medium reasoning on Responses (GPT) | 600 s / 8 |

Auxiliary calls use the OpenAI SDK with a separate five-second connection
timeout. The coding generator does not inherit the orchestrator's output budget;
its request timeout is fixed at 600 s, and its retries come from the orchestrator
service. Generated code executes locally with a separate 30-second default
limit; the generator receives binding descriptors, while the executor receives
actual in-memory values. See [runtime contracts](runtime.md).

To experiment with another summarizer, change the auxiliary model service to an
explicitly registered compatible profile. New model families require a registry
entry defining the exact served ID, API protocol, prompt and supported options;
the harness does not infer these from a model-name prefix. Qwen multimodal
summarization remains the official default in every preset.
