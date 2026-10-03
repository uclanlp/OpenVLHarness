# OpenVLHarness

**Real-World Visual Agentic Tool-Use with Multimodal Memory**

[Project page](https://tool-vision.github.io/openvlharness.github.io/) ·
[Interactive demo](https://tool-vision.github.io/openvlharness.github.io/#playground) ·
[Results](https://tool-vision.github.io/openvlharness.github.io/#results) ·
[Evaluation datasets](https://huggingface.co/datasets/uclanlp/OpenVLHarness-Evaluation-Datasets) ·
[Citation](#citation)

Yu Zhou\*, Cheng-Fu Yang\*, Rui Sun\*, Di Wu\*, Sibo Peng, Mingyang Zhang,
Zhen Yang, Ruiyi Zhang, Nanyun Peng, Zhe Gan, Kai-Wei Chang

University of California, Los Angeles · Apple · \*Equal contribution

OpenVLHarness is a **training-free harness for visual language models** that
combines perception, web search, and Python computation to answer questions about
images. It makes tool evidence readable by the model and reusable across reasoning
steps, without updating the backbone's weights.

The paper evaluates six backbones on **23 benchmarks** spanning counting and
grounding, multimodal search, general visual question answering, and spatial
understanding. Average gains over the corresponding base models range from
**9.2 to 12.6 percentage points**. Explore the
[results and recorded trajectories](https://tool-vision.github.io/openvlharness.github.io/#method)
on the project page.

## How it works

The model chooses tools, inspects their evidence, and continues until it can answer.
Three components let it combine evidence across steps:

- **Capability layer:** nine task-level interfaces turn tool outputs into readable
  observations, including annotated images, summaries, and compact numerical data.
- **Multimodal memory:** images, structured data, and web references remain
  addressable throughout a question. Each new question starts with fresh memory.
- **Python Coding Agent:** the backbone writes code in a separate context to
  calculate over stored boxes, masks, depth maps, and other tool results.

For example, an agent can locate objects, estimate depth, and compute their spatial
relationship; or crop an unfamiliar object, search for visual matches, and verify
the answer on a webpage.

| Capability | What it provides | Default backend |
| --- | --- | --- |
| Visual Grounding | Locate objects by text; return boxes, masks, and an annotated image | SAM 3 |
| OCR | Read text in an image | PaddleOCR-VL |
| Depth Estimation | Estimate per-pixel depth; retain the numerical array and visualization | Depth Anything 3 |
| Camera Trajectory | Estimate camera poses across ordered images | Depth Anything 3 |
| Zoom-in | Crop a region for closer inspection | Built-in image processing |
| Text Search | Retrieve and summarize web, image, news, and other search results | Serper + Qwen |
| Image Search | Find visual matches with reverse image search | Serper Lens + Qwen |
| Webpage Visit | Extract evidence relevant to a goal from webpages | Serper scrape + Qwen |
| Python Coding Agent | Generate and execute computations over stored tool results | Orchestrator backbone + local Python |

Both paper presets use **Qwen3-VL-8B-Instruct for multimodal search summarization
and webpage extraction**, independently of the main model. The coding agent uses
the same backbone as the main model.

## Quick start

### 1. Install the harness

Use Python **3.10 or newer**. From a source checkout:

```bash
git clone --branch opensource https://github.com/tool-vision/mini-vlm-toolkit.git
cd mini-vlm-toolkit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[qwen,server]'
```

This installs the agent, Qwen tokenizer support, and the tool router. The model
servers have separate environments and dependencies. For a client connecting to
an existing router, `pip install -e .` is sufficient; add the `qwen` extra when
using Qwen for local context counting or raw trace export.

### 2. Prepare models and tools

The release includes two tested agent presets:

| Main model | Configuration | API |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct | [qwen3_paper.yaml](configs/agent/qwen3_paper.yaml) | Chat Completions |
| GPT-6-Luna | [gpt6_luna_paper.yaml](configs/agent/gpt6_luna_paper.yaml) | Responses |

To run all nine capabilities, prepare these services before starting the agent:

- **Models:** serve Qwen locally or connect to a GPT endpoint, following the
  [model setup guide](docs/models.md). Both presets also need a Qwen endpoint for
  search tools; the Qwen main and auxiliary roles can share one server. The guide
  includes the tested model revision, serving versions, and GPU configuration.
- **Perception:** launch SAM 3, PaddleOCR-VL, and Depth Anything 3 using the
  [tool service guide](src/openvlharness/services/tools/README.md). These run as
  separate GPU services behind the router.
- **Search:** configure a Serper key using `SERPER_API_KEY` or
  `SERPER_API_KEY_FILE`. Reverse image search additionally needs a GCS or S3
  bucket for image uploads. Follow the
  [search and storage setup](src/openvlharness/services/tools/serper_text_search/README.md).

For the local Qwen setup, set these variables in both the router and agent
terminals. Adjust the addresses to match your deployment:

```bash
export MODEL_BASE_URL=http://127.0.0.1:8222/v1
export MODEL_API_KEY=empty
export AUXILIARY_MODEL_BASE_URL=http://127.0.0.1:8222/v1
export AUXILIARY_MODEL_API_KEY=empty
export TOOL_SERVER_URL=http://127.0.0.1:8101
export SAM3_SERVER_URL=http://127.0.0.1:8086
export OCR_SERVER_URL=http://127.0.0.1:8085
export DEPTH_ESTIMATION_SERVER_URL=http://127.0.0.1:8087
```

The `empty` keys above are for unauthenticated local endpoints. For GPT, select
`configs/agent/gpt6_luna_paper.yaml` in the commands and Python example below,
set `MODEL_BASE_URL` and `MODEL_API_KEY` for your Responses endpoint, and keep the
auxiliary connection pointing to Qwen. See the [GPT connection example](docs/models.md#connect-the-presets).

### 3. Run a question

Start the router with the same configuration that the agent will use:

```bash
python -m openvlharness.services.server --config configs/agent/qwen3_paper.yaml
```

In another terminal with the environment configured, check the router:

```bash
curl --fail http://127.0.0.1:8101/health
curl --fail http://127.0.0.1:8101/backend-status
```

Then ask a question from Python:

```python
from openvlharness.bootstrap import build_agent

agent = build_agent("configs/agent/qwen3_paper.yaml")
result = agent.generate([
    {"type": "image", "value": "/path/to/image.jpg"},
    {"type": "text", "value": "How many bottles are visible?"},
])
print(result["prediction"])
trajectory = result["extra_records"]["trajectory"]
```

Replace the image path with your own image. Inputs can include multiple images;
image values can be local paths, HTTP(S) URLs, or data URLs. The returned trajectory
records the model's turns, tool calls, and evidence for inspection.

The coding tool executes generated Python locally in the agent process. Run it in
an isolated environment appropriate for your data; its execution timeout is not a
security sandbox. See [execution details](src/openvlharness/services/tools/python_coding_agent/README.md).

## Evaluate on benchmarks

### Run a small evaluation

Install the benchmark dependencies in a separate evaluation environment, keeping
the model and tool services running:

```bash
python -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[qwen]'
```

Use the same connection variables as above. Set `LMUData` and the work directory
to a fast local disk when available. This example evaluates 12 reproducibly
sampled questions **per dataset** from the FSC147 and ODinW validation splits,
using direct counting and box metrics without an LLM judge:

```bash
export LMUData=/path/to/local/datasets
python run_benchmark.py \
  --config configs/agent/qwen3_paper.yaml \
  --judge-config configs/judge_no_judge.yaml \
  --data FSC147_val ODinW13_bbox_val_qwen3 \
  --num-examples 12 --shuffle --seed 42 \
  --num-proc 1 --save-html all --work-dir /path/to/local/runs
```

`--num-proc` controls concurrent examples. Remove `--num-examples` and `--shuffle`
to evaluate the full selected splits. For paper comparisons, also match the
benchmark prompts, judge settings, and scoring protocol in Appendix D. The paper
reports Qwen3-VL results averaged over three runs; a small subset checks the
installation rather than reproducing those scores.

### Datasets and scoring

Dataset names passed to `--data` select loaders in
[mini_vlm_toolkit/dataset](mini_vlm_toolkit/dataset). Examples across the four
domains are:

| Domain | Example dataset names |
| --- | --- |
| Counting and grounding | `FSC147_val`, `CountQA`, `PixmoCount_val`, `FSCD_147_bbox_val`, `FSCD_LVIS_bbox_val`, `ODinW13_bbox_val_qwen3` |
| Multimodal search and deep research | `MMSearch`, `MMSearchPlus`, `RealXBench`, `BrowseCompVL` |
| General VQA and hallucination | `RealWorldQA`, `HallusionBench`, `VStarBench`, `FREAK_MCQ`, `FREAK_VQA` |
| Spatial understanding | `MMSIBench_wo_circular`, `ERQA`, `EmbSpatialBench`, `SparBench_tiny`, `MindCubeBench_tiny_raw_qa` |

Prepared counting, grounding, VQA, and spatial splits are available in
[OpenVLHarness-Evaluation-Datasets](https://huggingface.co/datasets/uclanlp/OpenVLHarness-Evaluation-Datasets)
and download into `LMUData` on first use. Other loaders obtain data from their
original sources. ODinW also downloads evaluation annotations into `LMUData/GLIP`
unless `GLIP_ROOT` is set. The
[preparation scripts](mini_vlm_toolkit/dataset/scripts) document how the released
splits were built.

For benchmarks requiring an LLM judge or answer extractor, start from
[configs/judge.yaml](configs/judge.yaml) and configure the judge model, endpoint,
and key. The paper uses GPT-4o for search and deep-research judging and
Qwen3-VL-8B-Instruct for other LLM-judged tasks and answer extraction. The shipped
judge configuration points to Qwen; use the appropriate judge for each benchmark.

<details>
<summary>Grounding confidence scores</summary>

[configs/judge_no_judge.yaml](configs/judge_no_judge.yaml) selects
`score_mode: constant`, assigning every predicted box a confidence of 1.0.
For ODinW, FSCD, and LVIS grounding, set `score_mode: original` to use matching
tool detection scores. The configured value takes precedence over `SCORE_MODE`.
With neither override, evaluators run both modes when detection sidecars exist
(original is primary), or constant alone otherwise. The primary mode owns the
unsuffixed metrics files; alternate metrics include the mode in their filenames.

</details>

### Inspect and resume runs

Results are saved under `<work-dir>/<model>/<agent>/<run_id>/<dataset>/`, including
predictions, metrics, `trajectories.jsonl`, and full trajectory JSON. Use
`--save-html all` to browse the tool evidence in HTML reports. Images are written
beside trajectories by default; `--save-images base64` embeds them instead.

The run directory also records redacted configurations, resolved model profiles,
and service metadata. Add `--resume` to continue a compatible partial run; changes
to the effective configuration require a new run. `--incremental-json` appends
results as examples finish.

For **Qwen-only token-level export**, add `--save-raw-trace`. This writes per-sample
JSON and images under `raw_traces/`, with explicit error records for export
failures. It requires a matching tokenizer and a server that returns token IDs;
see [raw trace export](docs/models.md#raw-trace-export-qwen-only). Ordinary
trajectory JSON and HTML reports work with both presets.

## Configure and extend

Each agent YAML separates `agent` policy, `memory`, `capabilities`, and `services`.
Remove entries from `capabilities.tools` to run a tool ablation. The optional
[GroundingDINO backend](src/openvlharness/services/tools/grounding_dino/README.md)
can replace SAM 3 behind the same visual-grounding interface.

To add a model, register its served ID, API protocol, prompt, and supported options
in [model_registry.yaml](src/openvlharness/services/model_registry.yaml). Auxiliary
models can also use a registered compatible profile; multimodal Qwen remains the
official search default. See [runtime design](docs/runtime.md) for capability,
memory, and tool-service contracts.

| Guide | Contents |
| --- | --- |
| [Model services](docs/models.md) | Qwen serving, GPT connections, per-role settings, raw trace export |
| [Tool services](src/openvlharness/services/tools/README.md) | GPU workers, router deployment, search credentials, backend guides |
| [Runtime design](docs/runtime.md) | Agent loop, configuration, memory, capability interfaces |
| [Release validation](docs/validation.md) | Regression coverage and live validation evidence |

To run the package tests:

```bash
python -m pip install -e '.[test,server]'
python -m pytest tests -q
```

## Citation

If you use OpenVLHarness in your research, please cite:

```bibtex
@misc{openvlharness2026,
  title  = {OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory},
  author = {Zhou, Yu and Yang, Cheng-Fu and Sun, Rui and Wu, Di and Peng, Sibo and Zhang, Mingyang and Yang, Zhen and Zhang, Ruiyi and Peng, Nanyun and Gan, Zhe and Chang, Kai-Wei},
  year   = {2026},
  url    = {https://tool-vision.github.io/openvlharness.github.io/}
}
```
