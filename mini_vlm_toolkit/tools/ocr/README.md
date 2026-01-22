# OCR Tool Setup

This tool is a wrapper around a dedicated EasyOCR model server. The model server
runs separately from the mini VLM tool server. EasyOCR downloads model weights on
first run (by default under `~/.EasyOCR`).

## Image inputs

`OCR_Tool` accepts either `image_path` (server-local path) or `image_base64`
(base64-encoded image or data URL). Provide exactly one.

## 1) Install dependencies (model server env)

```bash
pip install easyocr
```

## 2) Launch the OCR model server

Set optional environment variables:
- `OCR_LANGS` (default: `ch_sim,en`)
- `OCR_GPU` (`true`, `false`, or `auto`, default: `auto`)
- `OCR_MODEL_DIR` (optional model storage directory)
- `OCR_DOWNLOAD_ENABLED` (`true` or `false`, default: `true`)
- `OCR_HOST` (default: `127.0.0.1`)
- `OCR_PORT` (default: `8085`)

Start the server:

```bash
./mini_vlm_toolkit/tools/ocr/run_ocr_server.sh
```

Health check:
```
curl http://127.0.0.1:8085/health
```

## 3) Launch the mini VLM tool server with the wrapper

Point the wrapper to the model server:

```bash
export OCR_SERVER_URL="http://127.0.0.1:8085"
export PYTHONPATH="$(pwd)"
export LIGHT_TOOL_MODULES="mini_vlm_toolkit.tools.ocr.tool"
./scripts/run_server.sh
```
