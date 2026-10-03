# Model services

The paper presets use native Chat Completions for Qwen and native Responses for
GPT Luna. Both use **Qwen3-VL-8B-Instruct for multimodal search summarization and
webpage extraction**. The Qwen orchestrator and auxiliary model can share one
server; their request settings are separate.

| Preset | Config alias | Exact model ID sent to the server | API |
| --- | --- | --- | --- |
| [Qwen](../configs/agent/qwen3_paper.yaml) | `qwen3-vl-8b-instruct` | `Qwen/Qwen3-VL-8B-Instruct` | Chat Completions |
| [GPT Luna](../configs/agent/gpt6_luna_paper.yaml) | `gpt-6-luna` | `gpt-6-luna` | Responses |

Aliases resolve through the explicit [model registry](../src/openvlharness/services/model_registry.yaml).
The vLLM served name must match the exact model ID, including capitalization;
the shorter config alias is not the served name.

When a model service omits `max_tokens`, the adapter uses 4096 for both local
context reservation and the provider's output limit (`max_tokens` for Chat
Completions, `max_output_tokens` for Responses). Explicit limits and the paper
presets retain their configured budgets.

## Verified Qwen serving setup

The before/after validation used an existing environment with vLLM **0.18.1**,
PyTorch **2.10.0**, Transformers **4.57.3**, OpenAI Python **2.6.0**, and
huggingface-hub **0.35.3**. It served model revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` on **two NVIDIA RTX A6000 48 GB GPUs**.
These are the tested versions and hardware; a fresh installation and smaller GPU
configurations have not been validated. Keep the serving environment separate
from the harness and heavy perception services.

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

## Connect the presets

Set these variables in the processes that run the harness and tool router. For
the Qwen preset and the unauthenticated local server shown above:

```bash
export MODEL_BASE_URL=http://127.0.0.1:8222/v1
export MODEL_API_KEY=empty
export AUXILIARY_MODEL_BASE_URL=http://127.0.0.1:8222/v1
export AUXILIARY_MODEL_API_KEY=empty
```

For the GPT Luna preset, keep the auxiliary variables pointing to Qwen and set:

```bash
export MODEL_BASE_URL=https://api.openai.com/v1
export MODEL_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment}"
```

The preset reads the credential from `MODEL_API_KEY`.

Model bases accept either a bare service base, such as
`http://127.0.0.1:8222`, or a base ending in `/v1`; the adapter constructs the
profile's `/v1/chat/completions` or `/v1/responses` endpoint. A full endpoint is
also accepted when it matches the profile's protocol. The GPT preset sends
`store=false`, medium reasoning, and an output budget of 16000 through Responses.
Access to the exact `gpt-6-luna` model ID must be available on the chosen endpoint.

Tool server addresses and search/storage credentials are configured separately;
these model variables alone do not start the tool services.

## Per-role settings

| Role | Output-token budget | Request settings | Timeout / additional retries |
| --- | ---: | --- | --- |
| Qwen orchestrator | 12800 | temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5, repetition penalty 1.0 | 600 s / 8 |
| GPT Luna orchestrator | 16000 | medium reasoning, Responses | 600 s / 8 |
| Qwen text/image search summarizer | 4096 | temperature 0, top_p 1, thinking disabled; multimodal messages for image-aware summaries | 600 s / 2 |
| Qwen webpage extractor | 16384 | temperature 0, top_p 1, thinking disabled | 600 s / 2 |
| Coding generator | 4096 | same model and native protocol as orchestrator; fresh prompt; Qwen temperature 0 or GPT medium reasoning | 120 s / 8 |

Auxiliary calls use the OpenAI SDK with a separate five-second connection
timeout. The coding generator's budget and timeout do not inherit the
orchestrator's larger values. Generated code executes locally with a separate
30-second default limit; the generator receives binding descriptors, while the
executor receives actual in-memory values. See [runtime contracts](runtime.md).

To experiment with another summarizer, change the auxiliary model service to an
explicitly registered compatible profile. New model families require a registry
entry defining the exact served ID, API protocol, prompt and supported options;
the harness does not infer these from a model-name prefix. Qwen multimodal
summarization remains the official default in both paper presets.
