# Benchmark loaders

Each `--data` name selects a loader class. `build_dataset(name)` in
[`__init__.py`](__init__.py) instantiates the first class in `DATASET_CLASSES` whose
`supported_datasets()` contains the name. The loader reads the benchmark TSV from
`$LMUData` into a DataFrame; `openvlharness-eval` calls its `build_prompt(row)` for every
row and, after inference, `evaluate(prediction_file, **judge_kwargs)`.

## Benchmarks

The 23 benchmarks of the paper's Table 1, followed by the five extended search benchmarks
of its appendix. Files are relative to this directory; rows are those evaluated by default.

| `--data` | Class | File | Rows | Metric | Judge or extractor (default prompt) |
| --- | --- | --- | ---: | --- | --- |
| *Counting and grounding* | | | | | |
| `FSC147_val` | `FSC147Dataset` | `image_vqa.py` | 1,286 | exact-match accuracy | optional extractor |
| `CountQA` | `CountQA` | `countqa.py` | 1,528 | exact-match accuracy | optional extractor |
| `PixmoCount_val` | `PixmoCount` | `pixmo_count.py` | 534 | exact-match accuracy | optional extractor |
| `FSCD_147_bbox_val` | `FSCD147BBoxDataset` | `fscd.py` | 1,286 | box AP | none |
| `FSCD_LVIS_bbox_val` | `FSCDLVISBBoxDataset` | `fscd.py` | 1,181 | box AP | none |
| `ODinW13_bbox_val_qwen3` | `ODinWBBoxDataset` | `odinw.py` | 4,938 | box AP, mean of 13 domains | none |
| *Multimodal search and deep research* | | | | | |
| `MMSearch` | `MMSearchEnd2End` | `mmsearch.py` | 171 | judged accuracy | judge (`hle`) |
| `MMSearchPlus` | `MMSearchPlus` | `mmsearch_plus.py` | 311 | judged accuracy | judge (`hle`) |
| `RealXBench` | `RealXBench` | `realxbench.py` | 194 | judged accuracy | judge (`realxbench`) |
| `FVQA_test` | `ImageShortQADataset` | `image_shortqa.py` | 300 of 1,800 | judged accuracy | judge (`judge_rs`) |
| `hle` | `ImageShortQADataset` | `image_shortqa.py` | 330 | judged accuracy | judge (`hle`) |
| `BrowseCompVL` | `BrowseCompVLDataset` | `browsecomp_vl.py` | 399 | judged accuracy | judge (`hle`) |
| *General VQA and hallucination* | | | | | |
| `RealWorldQA` | `ImageMCQDataset` | `image_mcq.py` | 765 | option accuracy | fallback extractor |
| `SimpleVQA` | `SimpleVQA` | `simplevqa.py` | 2,025 | share graded correct | judge (`COMPARE_ANSWER_PROMPT`) |
| `HallusionBench` | `ImageYORNDataset` | `image_yorn.py` | 951 | mean of aAcc, fAcc, qAcc | fallback extractor |
| `FREAK_VQA` | `FREAKVQADataset` | `freak.py` | 799 | judged accuracy | judge (`QUESTION_QUALITY_PROMPT_EN_NO_COT`) |
| `FREAK_MCQ` | `FREAKMCQDataset` | `freak.py` | 994 | option accuracy | fallback extractor |
| `VStarBench` | `ImageMCQDataset` | `image_mcq.py` | 191 | option accuracy | fallback extractor |
| *Spatial understanding* | | | | | |
| `MMSIBench_wo_circular` | `MMSIBench` | `mmsibench.py` | 1,000 | option accuracy | optional grader |
| `ERQA` | `ERQABench` | `erqabench.py` | 400 | option accuracy | optional grader |
| `EmbSpatialBench` | `EmbSpatialBench` | `embspatialbench.py` | 3,640 | option accuracy | optional grader |
| `SparBench_tiny` | `SparBench` | `sparbench.py` | 1,000 | mean of 20 task scores (accuracy or MRA) | optional grader |
| `MindCubeBench_tiny_raw_qa` | `MindCubeBench` | `mindcubebench.py` | 1,040 | option accuracy | optional grader |
| *Extended search (appendix)* | | | | | |
| `MMBrowseComp` | `MMBrowseCompDataset` | `mmbrowsecomp.py` | 243 | judged accuracy | judge (`mmbrowsecomp`) |
| `VDRBench` | `VDRBench` | `vdrbench.py` | 500 | judged accuracy | judge (`hle`) |
| `VisBrowseBench` | `VisBrowseBenchDataset` | `visbrowse_bench.py` | 169 | judged accuracy | judge (`vbb`) |
| `Infoseek_GDM_test` | `ImageShortQADataset` | `image_shortqa.py` | 300 of 2,000 | judged accuracy | judge (`judge_rs`) |
| `EncyclopedicVQA_test` | `ImageShortQADataset` | `image_shortqa.py` | 300 of 1,960 | judged accuracy | judge (`judge_rs`) |

- Every TSV except CountQA and PixmoCount downloads with its images on first use from
  [OpenVLHarness-Evaluation-Datasets](https://huggingface.co/datasets/uclanlp/OpenVLHarness-Evaluation-Datasets).
  CountQA is built from `Jayant-Sravan/CountQA`; PixmoCount is built from `allenai/pixmo-count`
  by fetching each image from its source URL, dropping rows whose image is gone.
- HallusionBench: the TSV has 1,129 rows; the 178 without an image are skipped.
- FVQA_test, Infoseek_GDM_test and EncyclopedicVQA_test evaluate a fixed 300-row sample
  (`random_state=42`) unless `FVQA_TEST_ALL`, `INFOSEEK_GDM_TEST_ALL` or
  `ENCYCLOPEDICVQA_TEST_ALL` is `true`.
- SparBench_tiny averages 11 multiple-choice accuracies, 8 mean relative accuracies (MRA)
  for numeric tasks and one view-change score.

Other registered names, such as MMBench, TextVQA, DocVQA, ChartQA, MME, POPE and AMBER, are
inherited VLMEvalKit loaders; they remain registered but are not part of the paper and are
untested here.

## Class tree

```
ImageBaseDataset (image_base.py)
├── ImageMCQDataset (image_mcq.py)
│   └── FREAKMCQDataset, MMSIBench, EmbSpatialBench, MindCubeBench, CustomMCQDataset
├── ImageVQADataset (image_vqa.py)
│   └── FSC147Dataset, CountQA, PixmoCount, FREAKVQADataset, ERQABench
├── ImageYORNDataset (image_yorn.py)
├── ImageShortQADataset (image_shortqa.py)
├── _FSCDBBoxBase (fscd.py)
│   └── FSCD147BBoxDataset, FSCDLVISBBoxDataset
└── SimpleVQA, SparBench, MMSearchEnd2End, MMSearchPlus, RealXBench, BrowseCompVLDataset,
    MMBrowseCompDataset, VDRBench, VisBrowseBenchDataset, ODinWBBoxDataset, LVISBBoxDataset,
    CustomVQADataset
```

`SimpleVQA_EN`, the MMSearch sub-task loaders and `LVISMaskDataset` are also registered. An
unknown name falls back to `$LMUData/<name>.tsv`, loaded as `CustomMCQDataset` if it has `A`
and `B` option columns and otherwise as `CustomVQADataset`, which has no evaluator.

## Grounding metrics

Grounding needs no judge. Every grounding loader reports COCO box AP over IoU 0.50:0.95
(`AP`), plus AP50, AP75 and size breakdowns. Boxes are parsed from the response (JSON, or a
fallback for lists of four numbers) and mapped from the 0-1000 grid to pixels.

- **ODinW-13** (`odinw.py`): each `_qwen3` row is one image whose prompt lists every category
  of its domain, and predicted labels are mapped to category ids. `pycocotools` COCOeval runs
  per domain against the GLIP annotations (downloaded to `$LMUData/GLIP` unless `GLIP_ROOT`
  is set); `ODinW13_average` is the mean over the 13 domains.
- **FSCD-147 and FSCD-LVIS** (`fscd.py`): each row queries one category in one image. The
  ground truth is a synthetic COCO file built from the rows' `answer` boxes, and COCOeval runs
  once with default parameters (`maxDets` 100). Errors of the predicted box counts (MAE,
  RMSE, NAE, exact match, within 5% and 10%) are reported alongside.
- **LVIS** (appendix; `LVIS_bbox_val_federated`, `lvis.py`): the official `LVISEval` with
  `max_dets=300` on the queried (image, category) pairs, including verified-absent categories.
  Build the TSV with [`scripts/prepare_lvis_federated.py`](scripts/prepare_lvis_federated.py);
  the annotations are read from `LVIS_GT_PATH`, `LVIS_DATA_ROOT` or `$LMUData/lvis-api/data/`.

AP ranks boxes by confidence, which the judge-config key `score_mode` sets. `constant` gives
every box 1.0. `original` matches each box (IoU >= 0.5) to the tool detections saved in
`tool_detections/<index>.json` during inference and takes the matched detection's score and
box; unmatched boxes score 0.5. [configs/judge_no_judge.yaml](../../configs/judge_no_judge.yaml)
sets `constant`. The configured value takes precedence over the `SCORE_MODE` environment
variable; with neither, evaluators run both modes when detection sidecars exist (`original`
primary) and `constant` otherwise. The primary mode owns the unsuffixed `_metrics.json` and
`_acc` files; the other mode's files carry its name (`_metrics_constant.json`).

Each grounding run also writes `_per_row.csv` with true and false positives, misses at IoU
0.5 and mean matched IoU per (image, category), sorted by F1@0.5. These are diagnostics for
inspecting runs, not reported metrics.

## Judges and answer extractors

`--judge-config` names the judge YAML; inference defaults to
[configs/judge.yaml](../../configs/judge.yaml) and `--mode eval` reuses the configuration
saved with the run. `load_judge_config` ([`../utils/judge_utils.py`](../utils/judge_utils.py))
maps `model_name`, `api_base`, `key`, `nproc` and `retry` to `evaluate()` keyword arguments
and passes any other key through; `build_judge` ([`utils/judge_util.py`](utils/judge_util.py))
wraps the model in an OpenAI-compatible client. Evaluators read three extra keys:

| Key | Values | Read by |
| --- | --- | --- |
| `judge_prompt` | `hle`, `judge_rs`, `judge_vdr`; RealXBench, MMBrowseComp and VisBrowseBench take `realxbench`, `mmbrowsecomp` and `vbb` instead of `hle` | the search loaders |
| `judge_max_tokens` | output-token cap per judge call, default 2048 | `build_judge`, `resolve_judge_max_tokens` |
| `score_mode` | `constant` or `original` | the grounding loaders |

The prompts live in `utils/hle_judge.py`, `utils/judge_rs.py`, `utils/judge_vdr.py`,
`utils/realxbench_judge.py`, `utils/mmbrowsecomp_judge.py` and `utils/visbrowse_bench_judge.py`.
The benchmark table gives the code defaults; the judge model and prompt behind each paper table
are stated in the paper's appendix. The table's last column means:

- **judge**: the model decides correctness. The eleven search benchmarks, SimpleVQA and
  FREAK_VQA need one; the last two use fixed prompts and ignore `judge_prompt`.
- **optional extractor**: with a judge model configured, `build_vqa_extract_prompt`
  (`utils/vqa_judge.py`) first reduces every counting response to a short answer, showing the
  ground truth so that the answer matches its format. `extract_integer_from_prediction`
  (`utils/fsc147.py`) then reads the integer, and exact match decides correctness.
- **fallback extractor**: a rule-based parser reads the option letter (`can_infer`,
  `utils/multiple_choice.py`) or yes/no (`YOrN_Extraction`, `utils/yorn.py`); the judge model
  is queried only when it fails.
- **optional grader**: with a judge model configured, `GENERIC_EXTRACT_JUDGE_PROMPT`
  (`utils/spatial_bench/llm_extract.py`) grades every response, and SparBench's numeric tasks
  score the extracted number by MRA; otherwise a rule-based parser matches the option or number.

With `model_name: null`, as in `judge_no_judge.yaml`, only the rule-based parsers run, so use
it only for benchmarks that need no judge.

## Adding a dataset

1. Write a TSV with `index`, `question` and `answer` columns and either `image` (base64, or
   the `index` of a row whose image to reuse) or `image_path` (absolute, or relative to
   `$LMUData/images/<folder>` when resolved by `dump_image`). Multiple-choice TSVs add option
   columns `A`, `B`, ...
2. Subclass `ImageMCQDataset`, `ImageVQADataset` or `ImageYORNDataset` to reuse their prompt
   and scoring, or `ImageBaseDataset` to write both.
3. Set `DATASET_URL` and `DATASET_MD5`, keyed by the `--data` name (`supported_datasets()`
   returns the `DATASET_URL` keys). Use `hf_tsv_url(name)` for a split hosted in the evaluation
   repository; with an empty URL the loader reads `$LMUData/<name>.tsv`.
4. Implement `build_prompt(line)`, which returns `[{'type': 'image', 'value': path}, ...,
   {'type': 'text', 'value': prompt}]`, and `evaluate(eval_file, **judge_kwargs)`, which
   scores the prediction file, writes score files next to it and returns the metrics.
   `ImageBaseDataset.evaluate` does nothing, so direct subclasses must implement it.
5. Import the class in [`__init__.py`](__init__.py) and add it to `IMAGE_DATASET`; when two
   classes serve the same name, the earlier one wins.
