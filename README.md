# README


## Tool Infra
`mini_vlm_toolkit/tool_infra` contains the lightweight tool server/client stack used for tool execution. It keeps the tool server, client, tools, tests, and docs self-contained so they stay stable alongside the eval code. To get started, see `mini_vlm_toolkit/tool_infra/TUTORIAL_NEW.md`.

### Implemented Tools

| Tool | Description |
| --- | --- |
| `Crop_Tool` | Crop an image using normalized or pixel coordinates. |
| `Depth_Tool` | Estimate depth from an image using a Depth-Anything server. |
| `DrawHorizontalLineByY_Tool` | Draw a horizontal line at the provided y coordinate. |
| `DrawLine_Tool` | Draw a dashed line between two corners of a bounding box. |
| `DrawVerticalLineByX_Tool` | Draw a vertical line at the provided x coordinate. |
| `ClipSearch_Tool` | Search a video-clip vector database for events matching a textual description. |
| `FrameInspection_Tool` | Inspect frames from specified time ranges and answer detailed video questions. |
| `GlobalBrowse_Tool` | Retrieve relevant video clips from a vector database and synthesize a final answer. |
| `Gemini_Google_Search_Tool` | Gemini-backed Google search tool with optional citations. |
| `GroundingDINO_Tool` | Detect objects in an image using a GroundingDINO server and return bounding boxes. |
| `OCR_Tool` | Extract text strings from an image using an OCR server. |
| `OverlayImages_Tool` | Overlay an image onto another with alpha blending. |
| `Point_Tool` | Extract point coordinates from text or provided points and return normalized coordinates. |
| `Python_Code_Generator_Tool` | Generate and execute simple Python code snippets for basic arithmetic calculations and math-related problems. |
| `LLM_Direct_Generator_Tool` | Send OpenAI-compatible chat or embeddings requests to a configured server and return the full payload. |
| `Python_Executor_Tool` | Execute a provided Python code snippet and return its output and variables. |
| `SAM3_Segment_Tool` | Call an external SAM3 segmentation server with a prompt and image. |
| `SAM3_VisualizeAllMasks_Tool` | Overlay all SAM3 masks on an image (grounding-agent or SAM3 server format). |
| `SAM3_VisualizeSingleMask_Tool` | Overlay a single SAM3 mask and return a zoomed-in view. |
| `SegmentAndMark_Tool` | Segment the full image and annotate each mask with a numeric label. |
| `SegmentRegionAroundPoint_Tool` | Segment a region around one or more points using SAM3. |
| `Serper_Google_Search_Tool` | Serper-backed Google search tool that returns URLs. |
| `Single_URL_QA_Tool` | Answer a query using content retrieved from a single URL. |
| `SlidingWindowDetection_Tool` | Run detection on overlapping crops to find small objects. |
| `ZoomInImageByBBox_Tool` | Crop an image around a normalized bbox with optional padding. |
| `ZoomInSubfigure_Tool` | Select and return subplots that match a textual description. |

## VLM Evaluation Usage
Use `run_benchmark.py` to run VLM benchmarks with different agents and datasets.

1. Install benchmark dependencies:
```bash
pip install -r requirements-benchmark.txt
```
2. Define your agent config in `configs/`. You can implement a new agent by inheriting from `mini_vlm_toolkit.agent.base.BaseAgent` and implementing `generate_inner`.
3. For native tool-calling runs, use `agent_name: BaseAgentWithTools` (for example `configs/base_agent_with_tools.yaml`) and set `agent_kwargs.prompt_version` to `"v1"` or `"v2"` as needed (`base_agent_with_tools.yaml` currently defaults to `"v2"`). Runtime environment hooks default to image updates (`runtime_environment_hooks: ["image"]`) with per-turn emission (`runtime_environment_emit_policy: "always"`). For OpenAI GPT-5 models, you can also set `agent_kwargs.reasoning_effort` (for example `"medium"`).
4. (Optional) Configure LLM-as-judge in `configs/judge.yaml` and pass `--judge-config`. If the judge config file is missing at the resolved path, evaluation falls back to exact matching.
5. Run the benchmark with desired config/data. Example:
```
python run_benchmark.py --config configs/base_agent_with_tools.yaml --judge-config configs/judge.yaml --data MMStar --num-proc 4 
```
6. Outputs are written under:
```text
<work-dir>/<model_name>/<agent_class_lower>/<model_name>_<dataset>.<format>
```
Default `work-dir` is `./outputs`.

Note: define your data root with `LMUData` in `.env` (capital D).
