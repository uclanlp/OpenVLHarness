# Tool service deployment

The agent calls one configured router. SAM3, DA3 and OCR run in separate model
processes; the router holds their HTTP wrappers. Install `openvlharness[server]`
on the router host and each raw model's dependencies in its own environment.
The basic client install does not require Torch, vLLM or raw model packages.

From the source checkout, prepare the router environment:

```bash
python -m venv /path/to/envs/router
source /path/to/envs/router/bin/activate
python -m pip install '.[server]'
```

The server extra includes Matplotlib because the camera wrapper uses it on
this host to render the bird's-eye view. The renderer tolerates a missing plotting
package by retaining numeric evidence and omitting its image, but the paper
installation includes it to preserve visual evidence.
The DA3 raw environment also needs Matplotlib for the paper's inferno depth
visualization. Neither installation requires a GPU on the router host.

## Configure and launch the router

Use `configs/agent/qwen3_paper.yaml` or `configs/agent/gpt6_luna_paper.yaml`.
Both are complete agent-system configurations. Set every `${...}` value named
by the selected file, including `TOOL_SERVER_URL`, `MODEL_BASE_URL`,
`MODEL_API_KEY`, `AUXILIARY_MODEL_BASE_URL`, `AUXILIARY_MODEL_API_KEY`,
`SAM3_SERVER_URL`, `OCR_SERVER_URL`, and `DEPTH_ESTIMATION_SERVER_URL`.
Empty API-key values are acceptable only for endpoints that need no key.

Model connection values belong in `services.models`. Raw service base URLs
belong in `services.tools`; omit `/v1/depth`, `/v1/pose`, `/v1/segment`, and
`/v1/chat/completions` from those raw URLs. Configure addresses reachable from
the router host. The agent sends image bytes after resolving runtime handles.
A path local to the agent is not assumed to exist on a remote router.

```bash
python -m openvlharness.services.server --config configs/agent/qwen3_paper.yaml
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8101/health
curl --fail http://127.0.0.1:8101/tools
curl --fail http://127.0.0.1:8101/llm-schema
curl --fail http://127.0.0.1:8101/backend-status
curl --fail http://127.0.0.1:8101/env
```

`/env` reports redacted configured services and fixed search settings. Only
selected implementations load. Removing a capability entry implements an
ablation; enabling `extensions` alone does not load GroundingDINO.
Depth and camera share the router's `depth` capacity gate. A gate is local to
one router process, so multiple router processes do not share that limit.

## Service environments

| Service | Guide | Default port | Router-to-raw timeout |
| --- | --- | --- | --- |
| SAM3 | [SAM3 setup](sam3_segment/README.md) | 8086 | 300 seconds |
| DA3 depth and pose | [DA3 setup](depth_estimation/README.md) | 8087 | depth 120, pose 300 seconds |
| PaddleOCR-VL | [OCR setup](ocr/README.md) | 8085 | 600 seconds |
| GroundingDINO extension | [DINO setup](grounding_dino/README.md) | 8084 | 120 seconds |
| Search and webpage extraction | [Search setup](serper_text_search/README.md) | external APIs | implementation-specific |
| Code generation and local execution | [Coding setup](python_coding_agent/README.md) | router plus model endpoint | local execution 30 seconds |

The model request timeout, router client timeout/retries and raw wrapper
request timeout are separate. Paper presets use a 600-second router client
and eight additional native transport retries. Raw timeout values above are
private wrapper defaults, not public tool arguments.

## Observed stage-1 environment

These versions were observed in the successful old-code raw-service checks on
2026-10-03 UTC. They describe an existing environment, not a clean-install lock.
All three environments used Python 3.10.14, Torch 2.10.0, Torchvision 0.25.0,
FastAPI 0.128.0, Uvicorn 0.40.0 and Pillow 12.0.0. OCR used vLLM 0.18.1 and
Transformers 4.57.3. DA3 used NumPy 1.26.4; SAM3 and OCR used NumPy 2.2.6.
Matplotlib was 3.9.2. The environments inherited some packages from an existing
vLLM installation; the installer recipes here use separate environments and
have been syntax-checked, not verified as fresh-install reproductions.

| Component | Tested source/checkpoint revision |
| --- | --- |
| SAM3 source | `2345a4ad109ac29c569da749c91d84f10dc08c40` |
| `facebook/sam3` checkpoint | `3c879f39826c281e95690f02c7821c4de09afae7` |
| DA3 source | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| `depth-anything/DA3NESTED-GIANT-LARGE` checkpoint | `8615eefb62f2db4f8d6ebaa59160086981672829` |
| `PaddlePaddle/PaddleOCR-VL-1.5` checkpoint | `2a4195faa5e7914c12f2fc601d72c81caf8d2da5` |

Stage 1 ran the original service modules with these checkpoints and obtained
real SAM3 masks, OCR text, depth arrays, and ordered camera matrices. The
relocated SAM3 and DA3 module entrypoints were then started in those same raw
environments without dependency upgrades, and the same three fixture requests
passed. SAM3 boxes/scores/RLE and depth values were identical; camera matrices
kept their shapes with a maximum extrinsics difference of approximately
2.4e-5 after restart. This validates those small raw fixtures, not a full model
benchmark or a clean installation. OCR retained its existing external vLLM
process while the router wrapper was relocated. DINO was not part of the GPU
checks; its launcher received a shell/argument check only.

Place `HF_HOME`, `TMPDIR`, `MPLCONFIGDIR`, `TORCH_HOME` and model-server caches
on a fast local disk when available. Set `HF_HUB_OFFLINE=1` only after all
required files are cached. Keep one worker per raw GPU process: importing a
raw server loads its model, and extra workers duplicate that allocation.

## Source license

The original tool infrastructure MIT notice is preserved verbatim in
[LICENSE.tool-infra](../LICENSE.tool-infra). Model weights and upstream raw-model
packages retain their own licenses and access requirements.
