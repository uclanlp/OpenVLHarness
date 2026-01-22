# Depth Tool Setup

This tool is a wrapper around a dedicated Depth-Anything model server (Hugging Face
`AutoModelForDepthEstimation`). The model server runs separately from the mini VLM
tool server.

## Image inputs

`Depth_Tool` accepts either `image_path` (server-local path) or `image_base64`
(base64-encoded image or data URL). Provide exactly one.

## 1) Install dependencies (model server env)

Install compatible `torch` and `transformers`:

```bash
pip install torch
pip install transformers
```

Optional colormap support:
- `matplotlib` (for the Inferno colormap). If missing, the server falls back to grayscale.

## 2) Launch the Depth-Anything model server

Set optional environment variables:
- `DEPTH_ANYTHING_MODEL_ID` (default: `LiheYoung/depth-anything-base-hf`)
- `DEPTH_ANYTHING_DEVICE` (`cuda`, `cpu`, or `auto`, default: `cpu`)
- `DEPTH_ANYTHING_COLORMAP` (`inferno` or `grayscale`, default: `inferno`)
- `DEPTH_ANYTHING_HOST` (default: `127.0.0.1`)
- `DEPTH_ANYTHING_PORT` (default: `8082`)

Start the server:

```bash
./mini_vlm_toolkit/tools/depth/run_depth_anything_server.sh
```

Health check:
```
curl http://127.0.0.1:8082/health
```

## 3) Launch the mini VLM tool server with the wrapper

Point the wrapper to the model server:

```bash
export DEPTH_ANYTHING_SERVER_URL="http://127.0.0.1:8082"
export PYTHONPATH="$(pwd)"
export LIGHT_TOOL_MODULES="mini_vlm_toolkit.tools.depth.tool"
./scripts/run_server.sh
```
