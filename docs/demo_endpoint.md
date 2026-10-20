# Use your own model in the live demo

The [interactive demo](https://huggingface.co/spaces/uclanlp/OpenVLHarness) runs the agent loop
and perception tools on our server. Your model does the rest: it orchestrates, writes
the coding tool's Python, and summarizes search results. This guide covers the
**OpenAI-compatible** option, where the model runs on a server you host yourself.
It explains how to get the two values the demo asks for: **Base URL** and **Model**.

## What the endpoint needs

- **Chat Completions API.** The endpoint is OpenAI-compatible at `/v1/chat/completions`.
  vLLM and SGLang provide this.
- **Image input.** Use a vision-language model; the demo sends your images and the
  tools' output images.
- **Native tool calling.** The model must return `tool_calls`, not tool calls written
  as plain text. With vLLM this means `--enable-auto-tool-choice` and a
  `--tool-call-parser` that matches the model.
- **A public HTTPS address.** Our server calls your endpoint, so it must be reachable
  from the internet over `https://`. The demo rejects `http://`, `localhost` and
  private-network addresses.
- **Enough context.** A run collects several tool results and images. The demo requests
  up to 12,800 output tokens for Qwen3-VL models and 8,192 for other models, so serve
  at least 32,768 tokens of context. 65,536 is better.

Qwen3-VL models are recognized by name: any model name containing `Qwen3-VL` gets the
paper's Qwen prompt and sampling settings. Other models get the generic prompt.

## Quick start: Qwen3-VL-8B with vLLM and a Cloudflare tunnel

This uses one GPU with at least 24 GB of memory. The 8B model's weights take about
17 GB; on a 48 GB GPU you can raise `--max-model-len`.

**1. Serve the model.** Choose an API key so that only you can use the endpoint:

```bash
pip install vllm
export VLLM_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
echo "API key: $VLLM_KEY"

vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port 8000 --api-key "$VLLM_KEY" \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --limit-mm-per-prompt '{"video": 0}'
```

`--served-model-name` sets the **Model** value you will enter in the demo. Other
Qwen3-VL sizes use the same flags; see [Qwen serving](models.md#verified-qwen-serving-setup)
for weight sizes and multi-GPU settings.

**2. Check it locally** from another terminal. The `id` in the reply is the exact
model name:

```bash
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $VLLM_KEY"
# {"object":"list","data":[{"id":"Qwen/Qwen3-VL-8B-Instruct", ...}]}
```

**3. Give it a public HTTPS address.** A
[Cloudflare quick tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
needs no account. Install `cloudflared` from that page, then run:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
# ... Your quick Tunnel has been created! Visit it at:
# https://<random-words>.trycloudflare.com
```

Keep it running while you use the demo. Every restart produces a new address. Any
HTTPS reverse proxy or a server with a public certificate works too.

**4. Check the public address:**

```bash
curl -s https://<random-words>.trycloudflare.com/v1/models -H "Authorization: Bearer $VLLM_KEY"
```

**5. Fill in the demo** with *OpenAI-compatible* selected:

| Field | Value |
| --- | --- |
| Base URL | `https://<random-words>.trycloudflare.com/v1` |
| Model | the `id` from step 2, e.g. `Qwen/Qwen3-VL-8B-Instruct` |
| Model API key | `$VLLM_KEY`; leave it empty if you served without `--api-key` |

Click **Test connection**. It should reply `✅ Connected to …`. Then pick a sample and
run it. Enter the base URL without `/chat/completions`; the demo adds it.

## Other servers and hosted APIs

- **SGLang** works the same way. Enable its tool-call parser for your model and use
  the `/v1` address it serves.
- **Hosted APIs.** Any provider with an OpenAI-compatible Chat Completions API,
  image input and tool calling works. The base URL and model name come from the
  provider's documentation. For OpenAI models, choose *OpenAI* instead; it uses the
  Responses API.

## Troubleshooting

| Message or symptom | Fix |
| --- | --- |
| `Enter an https base URL for your endpoint.` | Use the `https://` tunnel or proxy address, not `http://` or the local port. |
| `The endpoint must be publicly reachable…` | The address resolves to a private or local IP. Expose it through a tunnel or a public host. |
| `404` from the endpoint | The base URL should end at `/v1` (or the server root), without `/chat/completions`. |
| `The model … does not exist` | The Model field must equal the `id` from `/v1/models`, including capitalization. |
| `401` | The Model API key does not match the server's `--api-key`. |
| The model describes tools but never calls them | Start vLLM with `--enable-auto-tool-choice` and the right `--tool-call-parser` (`hermes` for Qwen3-VL). |
| Context-length errors late in a run | Raise `--max-model-len` (and GPU memory) on your server. |
| Errors about images | Serve a vision-language model and keep image input enabled. |

While the tunnel runs, anyone who has the address can reach your server, so set
`--api-key` and stop the tunnel when you are done. The demo uses your key only for
the run's requests and does not store it.
