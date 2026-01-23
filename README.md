# README

## VLM Evaluation Usage
Use `run_benchmark.py` to run VLM benchmarks with different agents and datasets.
1. Define your agent config in `configs/`. You can implement a new agent by inheriting from `mini_vlm_toolkit.agent.base.BaseAgent`. You need to write a new `generate_inner` method to call your desired LLM model.
2. Run the benchmark with the desired config and dataset. For example, to run the `BaseAgent` on the `MMStar LongVideoBench_8frame_subs` dataset with 4 parallel processes, use the following command:
```
python run_benchmark.py --config configs/base_agent.yaml --data MMStar LongVideoBench_8frame_subs --num_proc 4
```
Note: Define your data root in `LMUDATA` in `.env` or set it in the command line.

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
