# OpenVLHarness

OpenVLHarness runs a visual language model through nine tool capabilities with
per-question image, numerical-data and webpage memory. The paper presets use
Qwen3-VL-8B-Instruct through Chat Completions or GPT-6-Luna through Responses.
Search summarization and webpage extraction use greedy Qwen3-VL-8B-Instruct,
including image inputs for multimodal search, with either orchestrator.

## Install

The client package is independent of the benchmark and GPU-service environments:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# For local Qwen context counting:
pip install -e '.[qwen]'
# On the tool-router host:
pip install -e '.[server]'
```

Raw SAM3, PaddleOCR-VL and Depth Anything 3 servers use separate environments.
See [service setup](src/openvlharness/services/tools/README.md) for launch commands,
provider credentials and the fixed paper tool settings. For the retained benchmark
runner, install `requirements.txt` in an evaluation environment and install this
package there. Those broader dependencies are unnecessary for client-only use.

## Configure and run

See [model serving](docs/models.md) for the tested Qwen launch and native model
connections.

Choose [Qwen](configs/agent/qwen3_paper.yaml) or
[Luna](configs/agent/gpt6_luna_paper.yaml). Each YAML separates agent policy,
memory, capability selection, and deployment connections. Set the referenced
environment variables before loading it:

```bash
export MODEL_BASE_URL=http://127.0.0.1:8222/v1  # Qwen; use your Responses endpoint for Luna
export MODEL_API_KEY=empty                   # Set your provider key for a remote API
export AUXILIARY_MODEL_BASE_URL=http://127.0.0.1:8222/v1
export AUXILIARY_MODEL_API_KEY=empty
export TOOL_SERVER_URL=http://127.0.0.1:8101
export SAM3_SERVER_URL=http://127.0.0.1:8086
export OCR_SERVER_URL=http://127.0.0.1:8085
export DEPTH_ESTIMATION_SERVER_URL=http://127.0.0.1:8087
```

After starting the raw services and configuring search credentials, start the
router with the same resolved agent-system configuration:

```bash
python -m openvlharness.services.server --config configs/agent/qwen3_paper.yaml
```

Then run a question in another process:

```python
from openvlharness.bootstrap import build_agent

agent = build_agent("configs/agent/qwen3_paper.yaml")
result = agent.generate([
    {"type": "image", "value": "/path/to/image.jpg"},
    {"type": "text", "value": "How many bottles are visible?"},
])
print(result["prediction"])
```

`generate` creates fresh memory for every question. Its return value includes
`extra_records.trajectory` and grounding detections for reporting. The client
imports only the selected capability files; the router imports only configured
tool implementations. Unregistered model aliases and unsupported options fail
at startup. To add another served model, explicitly register its ID, protocol,
prompt and context policy in
[model_registry.yaml](src/openvlharness/services/model_registry.yaml). Kimi's
planned transport is Chat Completions; this release's tested profiles are Qwen
and Luna. An alternative auxiliary model can use a registered profile, while the
paper default remains multimodal Qwen.

## Capabilities

| Public tool | Paper implementation and result |
| --- | --- |
| `Visual_Grounding_Tool` | SAM3 boxes, scores, RLE masks and numbered overlay |
| `OCR_Tool` | PaddleOCR-VL text, also available as `ocr_result["text"]` |
| `Depth_Estimation_Tool` | Depth Anything 3 visualization and float32 array binding |
| `Camera_Trajectory_Tool` | Ordered DA3 poses, float32 intrinsics/extrinsics and visualization |
| `Zoom_In_Tool` | Crop using 0–1000 coordinates and padding |
| `Text_Search_Tool` | Serper web/image/news/scholar/maps/places/shopping/patent search |
| `Image_Search_Tool` | Serper Lens reverse image search |
| `Webpage_Visit_Tool` | Serper scrape and Qwen evidence extraction, up to five URLs |
| `Python_Coding_Agent_Tool` | Orchestrator-matched code generation, then local execution on copied bindings |

The optional `extensions` registry enables `grounding_dino` as a replacement
for `grounding`; it keeps the public `Visual_Grounding_Tool` signature. DINO stores
`segment_<query>` with `pred_boxes` and `pred_scores`; masks and mask count are
null. It does not invoke SAM3 to manufacture masks. Remove a capability from
`capabilities.tools` to omit it from the agent and router.

Code execution copies input bindings and has a 30-second limit. It executes
Python in the calling process and **is not a security sandbox**. The worker-thread
timeout cannot necessarily interrupt a blocked native-library or network call.
Use an isolated deployment appropriate for the data and generated code.

See [runtime design](docs/runtime.md) for contracts, ownership, retries and the
two intentional changes from the old runtime.

## Benchmark evaluation

The existing dataset loaders and evaluators remain under `mini_vlm_toolkit`.
Set `LMUData` to a local disk for downloads and generated artifacts. FSC147 and
ODinW have deterministic counting/box metrics and do not require an LLM judge:

```bash
export LMUData=/path/to/local/datasets
python run_benchmark.py \
  --config configs/agent/qwen3_paper.yaml \
  --judge-config configs/judge_no_judge.yaml \
  --data FSC147_test --num-examples 12 --shuffle --seed 42 \
  --num-proc 1 --save-html all --work-dir /path/to/local/runs
```

Use `ODinW13_bbox_test_qwen3` for ODinW, and `configs/judge.yaml` for benchmarks
that require a judge. The no-judge preset passes an actual null model into the
evaluator. `--num-examples` limits the evaluated sample, `--shuffle --seed`
selects a repeatable subset, and `--num-proc` controls concurrent examples.

Each run creates a unique directory under
`<work-dir>/<model>/<agent>/<run_id>/<dataset>/`, containing predictions,
`trajectories.jsonl`, full trajectories and optional HTML reports. The parent
contains the resume manifest, redacted configurations and a router environment
snapshot. `--resume` requires matching run configuration. `--save-images files`
(the default) writes attachments beside full trajectories; `base64` keeps them
inline. `--incremental-json` writes results as examples complete.

### Benchmarks hosted on HuggingFace
The splits below (prompts, annotations and images) are hosted at
[uclanlp/OpenVLHarness-Evaluation-Datasets](https://huggingface.co/datasets/uclanlp/OpenVLHarness-Evaluation-Datasets)
and are downloaded into `LMUData` automatically the first time you pass them to `--data`:

- Counting & grounding (about 0.8 GB of images): `FSC147_val`, `FSC147_test`, `FSCD_147_bbox_val`,
  `FSCD_147_bbox_test`, `FSCD_LVIS_bbox_val`, `FSCD_LVIS_bbox_test`, `ODinW13_bbox_val_qwen3`,
  `ODinW13_bbox_test_qwen3`
- General VQA: `RealWorldQA`, `HallusionBench`, `VStarBench`, `FREAK_MCQ`, `FREAK_VQA`
- Spatial reasoning: `MMSIBench_wo_circular`, `ERQA`, `SparBench`, `SparBench_tiny`,
  `MindCubeBench_raw_qa`, `MindCubeBench_tiny_raw_qa`, `EmbSpatialBench`, `OmniSpatialBench`
  (and its `_default` / `_zeroshot_cot` / `_manual_cot` prompt variants)

ODinW evaluation also downloads its ground-truth annotations to `LMUData/GLIP` unless
`GLIP_ROOT` is set. The scripts in `mini_vlm_toolkit/dataset/scripts/prepare_*.py`
document how each split was built from its original source. `CountQA` and `PixmoCount_*`
are instead built from their original sources on first use (see `countqa.py`, `pixmo_count.py`).

## Validation and historical source

```bash
pip install -e '.[test,server]'
python -m pytest tests -q
```

The fixtures preserve old public schemas, prompts, model payloads, memory values
and tool evidence. They cover both native protocols, numeric dtype/shape, local
execution, error envelopes and shared backend concurrency. Regression evidence
and the small live comparison are described in [release validation](docs/validation.md).

The old runtime is recorded at commit
`b98e5a540c4867eab20e07166c274e7ca8b8b9fe` on `opensource`. Its agent classes,
configuration format and cache artifacts are not compatibility APIs for the new
package. Dataset preparation and evaluation remain available in their original
locations.
