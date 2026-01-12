# GroundingDINO Tool Setup

This tool is a wrapper around a dedicated GroundingDINO model server (Hugging Face
`AutoModelForZeroShotObjectDetection`). The model server runs separately from the
mini VLM tool server.

## 1) Install dependencies (model server env)

Install compatible `torch`, `torchvision`, and `transformers`:

```bash
pip install torch torchvision
pip install transformers
```

## 2) Launch the GroundingDINO model server

Set optional environment variables:
- `GROUNDING_DINO_MODEL_ID` (default: `IDEA-Research/grounding-dino-base`)
- `GROUNDING_DINO_DEVICE` (`cuda`, `cpu`, or `auto`, default: `auto`)
- `GROUNDING_DINO_HOST` (default: `127.0.0.1`)
- `GROUNDING_DINO_PORT` (default: `8084`)

Start the server:

```bash
export PYTHONPATH="$(pwd)"
./scripts/run_grounding_dino_server.sh
```

Health check:
```
curl http://127.0.0.1:8084/health
```

## 3) Launch the mini VLM tool server with the wrapper

Point the wrapper to the model server:

```bash
export GROUNDING_DINO_SERVER_URL="http://127.0.0.1:8084"
export PYTHONPATH="$(pwd)"
export TOOL_MODULES="mini_vlm_toolkit.tools.grounding_dino.tool"
./scripts/run_server.sh
```

Optional:
- `GROUNDING_DINO_TEST_QUERY` (default: `person`)
