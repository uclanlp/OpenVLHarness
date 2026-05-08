# OCR Tool Setup

This tool wraps a dedicated PaddleOCR-VL model server. The model server
runs separately from the mini VLM tool server. Weights for
`PaddlePaddle/PaddleOCR-VL-1.5` are downloaded by `transformers` on first
run and cached under the standard Hugging Face cache directory.

## Image inputs

`OCR_Tool` accepts either `image_path` (server-local path) or `image_base64`
(base64-encoded image or data URL). Provide exactly one.

## 1) Install dependencies (model server env)

```bash
pip install "transformers>=5.0" torch Pillow
```

`torch` must match your CUDA toolkit (CU≤12.6 recommended for the model
weights distribution). CPU inference works but is significantly slower.

## 2) Launch the OCR model server

Optional environment variables:
- `OCR_MODEL_ID` (default: `PaddlePaddle/PaddleOCR-VL-1.5`)
- `OCR_DEVICE` (`auto`, `cuda`, or `cpu`, default: `auto`)
- `OCR_TORCH_DTYPE` (default: `bfloat16`; e.g. `float16`, `float32`)
- `OCR_MAX_NEW_TOKENS` (default: `2048`)
- `OCR_REPETITION_PENALTY` (default: `1.0`; the PaddleOCR-VL team
  recommends `1.05+` to avoid repetition loops on certain inputs)
- `OCR_HOST` (default: `127.0.0.1`)
- `OCR_PORT` (default: `8085`)

Start the server:

```bash
./src/tools/ocr/run_ocr_server.sh
```

Health check:

```bash
curl http://127.0.0.1:8085/health
```

Smoke transcription:

```bash
curl -F image=@/path/to/scan.png http://127.0.0.1:8085/v1/ocr
```

Response shape:

```json
{
  "text": "...transcribed text...",
  "model_id": "PaddlePaddle/PaddleOCR-VL-1.5",
  "device": "cuda"
}
```

## 3) Launch the mini VLM tool server with the wrapper

Point the wrapper to the model server:

```bash
export OCR_SERVER_URL="http://127.0.0.1:8085"
export PYTHONPATH="$(pwd)/src"
export LIGHT_TOOL_MODULES="tools.ocr.tool"
./scripts/run_server.sh
```
