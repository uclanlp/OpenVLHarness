# OCR Tool Setup

`OCR_Tool` calls a vLLM-served PaddleOCR-VL model directly via the
OpenAI-compatible `/v1/chat/completions` endpoint. There is no separate
wrapper process — `tool.py` does image preprocessing (EXIF transpose, RGB,
PNG re-encode → base64 data URL) and posts to vLLM. Weights for
`PaddlePaddle/PaddleOCR-VL-1.5` are downloaded by vLLM on first run and
cached under the standard Hugging Face cache directory.

## Image inputs

`OCR_Tool` accepts either `image_path` (server-local path) or `image_base64`
(base64-encoded image or data URL). Provide exactly one.

## 1) Install dependencies

```bash
./src/tools/ocr/install.sh
```

This creates a `conda` env named `ocr` with `paddleocr`, the vLLM stack
required by PaddleOCR-VL, and the small client deps used by `tool.py`.

## 2) Launch the OCR server

Optional environment variables:
- `OCR_MODEL_ID` (default: `PaddlePaddle/PaddleOCR-VL-1.5`)
- `OCR_HOST` (default: `127.0.0.1`)
- `OCR_PORT` (default: `8085`)

Start the server:

```bash
./src/tools/ocr/run_ocr_server.sh
```

First run downloads weights — wait for vLLM to log
`Application startup complete` / `Uvicorn running on http://127.0.0.1:8085`.

Health check:

```bash
curl http://127.0.0.1:8085/health
curl http://127.0.0.1:8085/v1/models
```

Smoke transcription (matches the OpenAI-client snippet from the PaddleOCR-VL
docs):

```bash
curl http://127.0.0.1:8085/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "PaddlePaddle/PaddleOCR-VL-1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"}},
          {"type": "text", "text": "OCR:"}
        ]
      }
    ],
    "temperature": 0
  }'
```

`OCR_Tool.execute()` returns:

```json
{
  "text": "...transcribed text...",
  "model_id": "PaddlePaddle/PaddleOCR-VL-1.5"
}
```

## 3) Launch the mini VLM tool server with the wrapper

Optional environment variables for the wrapper side:
- `OCR_SERVER_URL` (default: `http://127.0.0.1:8085`) — base URL `tool.py` hits
- `OCR_MODEL_ID` (default: `PaddlePaddle/PaddleOCR-VL-1.5`) — sent as the
  `model` field in chat completions; should match what vLLM is serving
- `OCR_MAX_NEW_TOKENS` (default: `2048`) — forwarded as `max_tokens`
- `OCR_REPETITION_PENALTY` (default: `1.0`; PaddleOCR-VL recommends `1.05+`;
  forwarded only when ≠ 1.0)

```bash
export OCR_SERVER_URL="http://127.0.0.1:8085"
export PYTHONPATH="$(pwd)/src"
export LIGHT_TOOL_MODULES="tools.ocr.tool"
./scripts/run_server.sh
```
