# README


## Tool Infra
`mini_vlm_toolkit/tool_infra` contains the lightweight tool server/client stack used for tool execution. It keeps the tool server, client, tools, tests, and docs self-contained so they stay stable alongside the eval code. To get started, see `mini_vlm_toolkit/tool_infra/TUTORIAL_NEW.md`.

### Implemented Tools

| Tool | Description |
| --- | --- |
| `Camera_Trajectory_Tool` | Estimate per-image camera motion for two or more images of the same scene via the Depth-Anything pose endpoint; returns a bird's-eye-view plot plus movement and rotation summaries. |
| `Crop_Tool` | Crop an image using 0-1000 normalized or pixel coordinates. |
| `Depth_Estimation_Tool` | Estimate depth from an image using a Depth-Anything server. |
| `DrawHorizontalLineByY_Tool` | Draw a horizontal line at the provided y coordinate. |
| `DrawLine_Tool` | Draw a dashed line between two corners of a bounding box. |
| `DrawVerticalLineByX_Tool` | Draw a vertical line at the provided x coordinate. |
| `Gemini_Google_Search_Tool` | Gemini-backed Google search tool with optional citations. |
| `Get_Tool_Schema` | Return the full parameter schema and usage guide for a named tool; used by deferred-schema agents such as `BaseAgentWithToolsV3`. |
| `GroundingDINO_Tool` | Detect objects in an image using a GroundingDINO server and return bounding boxes. |
| `Image_Edit_Tool` | Edit or compose one or more runtime images from a text instruction using a Qwen-Image-Edit server. |
| `Image_Generation_Tool` | Generate a new image from a text prompt using a Qwen-Image server; the output is registered as a runtime image. |
| `Image_Search_Tool` | Reverse image search on a runtime image via Serper Google Lens; returns visual matches with titles, source URLs, and thumbnails. |
| `OCR_Tool` | Extract text strings from an image using an OCR server. |
| `OverlayImages_Tool` | Overlay an image onto another with alpha blending. |
| `Point_Tool` | Extract point coordinates from text or provided points and return 0-1000 normalized coordinates. |
| `Python_Code_Generator_Tool` | Generate and execute simple Python code snippets for basic arithmetic calculations and math-related problems. |
| `Python_Coding_Agent_Tool` | Ask an LLM to write Python over data bindings accumulated from earlier tool results (boxes, depth maps, OCR text); the agent executes the code locally and returns the printed output. |
| `LLM_Direct_Generator_Tool` | Send OpenAI-compatible chat or embeddings requests to a configured server and return the full payload. |
| `Python_Executor_Tool` | Execute a provided Python code snippet and return its output and variables. |
| `Visual_Grounding_Tool` | Call the SAM3 segmentation server and return RLE masks with a rendered overlay. |
| `SerpApi_Image_Search_Tool` | Search similar images with SerpApi Google Reverse Image Search from a public image URL. |
| `SerpApi_Google_Lens_Tool` | Identify objects, landmarks, artwork, text, and more in an image using Google Lens via SerpApi. |
| `Serper_Google_Search_Tool` | Serper-backed Google search tool that returns URLs. |
| `Single_URL_QA_Tool` | Answer a query using content retrieved from a single URL. |
| `Text_Search_Tool` | Serper-backed web search across web, image, news, scholar, maps, places, shopping, and patents endpoints, with optional LLM re-ranking and summarization; returned URLs become `webpage_url_N` refs. |
| `Tracking_Tool` | Track a described target across an ordered sequence of images using SAM3 detections and Depth-Anything depth; returns per-frame tracks, 2D movement direction, and depth change. |
| `Webpage_Visit_Tool` | Visit up to five URLs via Serper's scraping API and use an LLM to extract the evidence and a summary relevant to a stated goal. |
| `Zoom_In_Tool` | Crop an image around a 0-1000 normalized bbox with optional padding. |
| `ZoomInSubfigure_Tool` | Select and return subplots that match a textual description. |

Config-driven subagent tools (`mini_vlm_toolkit/tool_infra/src/tools/subagent/`, one tool per YAML file in `SUBAGENT_CONFIG_DIR`) are registered dynamically at server start and are not listed above.

## VLM Evaluation Usage
Use `run_benchmark.py` to run VLM benchmarks with different agents and datasets.

You can find the supported benchmark here: [link](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY)

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Define your agent config in `configs/`. You can implement a new agent by inheriting from `mini_vlm_toolkit.agent.base.BaseAgent` and implementing `generate_inner`. Agent config must include top-level `agent_config_version`.
3. For native tool-calling runs, use `agent_name: BaseAgentWithTools` (for example `configs/base_agent_with_tools.yaml`) and set `agent_kwargs.prompt_version` to `"v1"` or `"v2"` as needed (`base_agent_with_tools.yaml` currently defaults to `"v2"`). Persistent state memory defaults to image updates (`persistent_state_memory: {name: "image", emit_policy: "always"}`) and processes tool outputs before appending model-facing messages. For OpenAI GPT-5 models, you can also set `agent_kwargs.reasoning_effort` (for example `"medium"`).
4. Configure LLM-as-judge in `configs/judge.yaml` and pass `--judge-config`. Judge config must include top-level `judge_config_version` and `model_name`.
5. Run the benchmark with desired config/data. Example:
```
python run_benchmark.py --config configs/base_agent_with_tools.yaml --judge-config configs/judge.yaml --data MMStar --num-proc 4 --save-html 20
```
6. Outputs are written under:
```text
<work-dir>/<model_name>/<agent_class_lower>/<run_id>/<dataset>/
  <model_name>_<result/acc>.<format>                # predictions
  trajectories.jsonl                                # per-task trajectories
  htmls/                                            # optional debug HTML reports
<work-dir>/<model_name>/<agent_class_lower>/<run_id>/
  run_manifest.json                                 # resume identity + matching policy
  agent_config.redacted.yaml                        # copied once per run_id
  judge_config.redacted.yaml                        # copied once per run_id
```
Where `run_id` is derived automatically as
`agent_<agent_config_version>_judge_<judge_config_version>_<YYYYMMDD>_<HHMMSS>_<tag>`.
Each run creates a unique timestamped directory, preventing accidental cache collisions.

#### Optional:
`--run-tag` sets a custom tag appended to the run directory name (default: `"default"`):
```
python run_benchmark.py --config configs/base_agent_with_tools.yaml --judge-config configs/judge.yaml --data MMStar --run-tag my_experiment
```

`--resume` finds and resumes the latest existing run that matches the agent/judge version prefix, instead of creating a new timestamped directory:
```
python run_benchmark.py --config configs/base_agent_with_tools.yaml --judge-config configs/judge.yaml --data MMStar --resume
```

`--num-examples` limits inference to the first N examples in the dataset (useful for quick testing):
```
python run_benchmark.py --config configs/base_agent_with_tools.yaml --judge-config configs/judge.yaml --data MMStar --num-examples 10
```

`--shuffle` changes `--num-examples` from "take the first N" to "randomly sample N examples" from the dataset:
```
python run_benchmark.py --config configs/base_agent_with_tools.yaml --judge-config configs/judge.yaml --data MMStar --num-examples 10 --shuffle
```

`--seed` sets the random seed used by `--shuffle` (default: `42`) so the sampled subset is reproducible:
```
python run_benchmark.py --config configs/base_agent_with_tools.yaml --judge-config configs/judge.yaml --data MMStar --num-examples 10 --shuffle --seed 123
```

`--save-html` controls HTML report generation:
- `--save-html 0` (default): disable HTML output.
- `--save-html N`: generate top `N` tasks ranked by trajectory/chat-history length.
- `--save-html all`: generate HTML for all tasks.

`--save-images` controls how images are stored inside the per-task
`full_trajectories/<task_id>.json` files:
- `--save-images files` (default): every user/tool image is decoded and
  written to a sibling `full_trajectories/images/` folder as
  `<task_id>_<idx>.<ext>`, and the corresponding `image_url` entries in the
  saved messages are rewritten to relative paths like `images/51_0.jpg`.
  Keeps the JSON files small and gives downstream tooling (e.g. SFT data
  conversion) real on-disk paths.
- `--save-images base64`: keep inline `data:image/...;base64,...` URIs in
  the messages (legacy behavior). No `images/` folder is created.

`--incremental-json` enables real-time JSONL output during inference:
- When enabled, writes `{model_name}_incremental.jsonl` in the dataset work directory.
- Each completed task is appended as a JSON object per line as results arrive.
- Useful for debugging and monitoring progress without waiting for the full run to complete.
- You can monitor results in real-time using `tail -f <work-dir>/.../<model_name>_incremental.jsonl`.

Note: define your data root with `LMUData` in `.env` (capital D).

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

## Local Visualization Server
To visualize trajectories, download the full_trajectories directory and run this command to start a local visualization server:
```
streamlit run configs/cf-testing/visualize_server.py --trajectory-dir <path-to-trajectories>
```
Then open the provided localhost URL in your browser to explore the trajectories with an interactive interface.
