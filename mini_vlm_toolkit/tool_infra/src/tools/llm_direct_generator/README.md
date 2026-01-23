# LLM Direct Generator Tool

`LLM_Direct_Generator_Tool` forwards OpenAI-compatible chat or embeddings requests to
servers (such as vLLM) and returns the full JSON response payload. Treat it as a heavy
tool: you must launch the model servers yourself and configure `.env`.

## Start model servers yourself

This tool does not launch model servers. Start your vLLM servers separately
and then set the model-to-base-URL mapping in `.env`.

Example (adjust to your environment):
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --port 8004

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Embedding-8B \
  --port 8811
```

Then update `.env`:
```bash
LLM_DIRECT_MODEL_BASE_URLS=Qwen/Qwen3-8B:http://127.0.0.1:8004/v1,Qwen/Qwen3-Embedding-8B:http://127.0.0.1:8811/v1
LLM_DIRECT_MODEL_API_KEYS=Qwen/Qwen3-8B:EMPTY,Qwen/Qwen3-Embedding-8B:EMPTY
LLM_DIRECT_API_KEY=EMPTY
```

If your vLLM server does not require auth, keep API keys as `EMPTY`.

## Embeddings support

Set `request_type="embeddings"` and supply `input` with a string or list of strings.
