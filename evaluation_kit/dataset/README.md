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
| `FSC147_val` | `FSC147Dataset` | `image_vqa.py` | 1,286 | exact-match accuracy | strict count parse |
| `CountQA` | `CountQA` | `countqa.py` | 1,528 | exact-match accuracy | strict count parse |
| `PixmoCount_val` | `PixmoCount` | `pixmo_count.py` | 534 | exact-match accuracy | strict count parse |
| `FSCD_147_bbox_val` | `FSCD147BBoxDataset` | `fscd.py` | 1,286 | box AP | none |
| `FSCD_LVIS_bbox_val` | `FSCDLVISBBoxDataset` | `fscd.py` | 1,181 | box AP | none |
| `ODinW13_bbox_val_qwen3` | `ODinWBBoxDataset` | `odinw.py` | 4,938 | box AP, mean of 13 domains | none |
| *Multimodal search and deep research* | | | | | |
| `MMSearch` | `MMSearchEnd2End` | `mmsearch.py` | 171 | judged accuracy | judge (`judge_vdr`) |
| `MMSearchPlus` | `MMSearchPlus` | `mmsearch_plus.py` | 311 | judged accuracy | judge (`judge_vdr`) |
| `RealXBench` | `RealXBench` | `realxbench.py` | 194 | judged accuracy | judge (`judge_vdr`) |
| `FVQA_test` | `ImageShortQADataset` | `image_shortqa.py` | 1,800 | judged accuracy | judge (`judge_vdr`) |
| `hle` | `ImageShortQADataset` | `image_shortqa.py` | 330 | judged accuracy | judge (`judge_vdr`) |
| `BrowseCompVL` | `BrowseCompVLDataset` | `browsecomp_vl.py` | 399 | judged accuracy | judge (`judge_vdr`) |
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
| `MMBrowseComp` | `MMBrowseCompDataset` | `mmbrowsecomp.py` | 243 | judged accuracy | judge (`judge_vdr`) |
| `VDRBench` | `VDRBench` | `vdrbench.py` | 500 | judged accuracy | judge (`judge_vdr`) |
| `VisBrowseBench` | `VisBrowseBenchDataset` | `visbrowse_bench.py` | 169 | judged accuracy | judge (`judge_vdr`) |
| `Infoseek_GDM_test` | `ImageShortQADataset` | `image_shortqa.py` | 2,000 | judged accuracy | judge (`judge_vdr`) |
| `EncyclopedicVQA_test` | `ImageShortQADataset` | `image_shortqa.py` | 1,960 | judged accuracy | judge (`judge_vdr`) |

- Every TSV except CountQA and PixmoCount downloads with its images on first use from
  [OpenVLHarness-Evaluation-Datasets](https://huggingface.co/datasets/uclanlp/OpenVLHarness-Evaluation-Datasets).
  CountQA is built from `Jayant-Sravan/CountQA`; PixmoCount is built from `allenai/pixmo-count`
  by fetching each image from its source URL, dropping rows whose image is gone, so its row
  count can fall below the 534 listed as source URLs disappear.
- HallusionBench: the TSV has 1,129 rows; the 178 without an image are skipped.
- FVQA_test, Infoseek_GDM_test and EncyclopedicVQA_test evaluate their full splits, as in
  the paper. `FVQA_TEST_SUBSET=300`, `INFOSEEK_GDM_TEST_SUBSET=300` or
  `ENCYCLOPEDICVQA_TEST_SUBSET=300` evaluates a fixed random subset (seed 42) instead, a
  quick estimate of the full-split score.
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

`SimpleVQA_EN`, the MMSearch sub-task loaders and `LVISMaskDataset` (masks, scored by IoU
matching rather than AP; not covered here) are also registered. An unknown name falls back to
`$LMUData/<name>.tsv`, which needs a `question` column. It is loaded as `CustomMCQDataset` if
it has `A` and `B` option columns and an `image` or `image_path` column (an A/B TSV without
either is rejected), and otherwise as `CustomVQADataset`, which has no evaluator.

## Grounding metrics

Grounding needs no judge. Every box-grounding loader reports box AP over IoU 0.50:0.95
(`AP`), plus AP50, AP75 and size breakdowns, computed by `pycocotools` COCOeval for ODinW-13
and FSCD and by `LVISEval` for LVIS. Boxes are parsed from the response (JSON, or a fallback
for lists of four numbers) and mapped from the 0-1000 grid to pixels.

- **ODinW-13** (`odinw.py`): each `_qwen3` row is one image whose prompt lists every category
  of its domain, and predicted labels are mapped to category ids. `pycocotools` COCOeval runs
  per domain against the GLIP annotations (downloaded to `$LMUData/GLIP` unless `GLIP_ROOT`
  is set); `ODinW13_average` is the mean over the 13 domains.
- **FSCD-147 and FSCD-LVIS** (`fscd.py`): each row queries one category in one image. The
  ground truth is a synthetic COCO file built from the rows' `answer` boxes, and COCOeval runs
  once with default parameters (`maxDets` 100). Errors of the predicted box counts (MAE,
  RMSE, NAE, exact match, within 5% and 10%) are reported alongside.
- **LVIS** (appendix; `LVIS_bbox_val_federated`, `lvis.py`): the official `LVISEval` with
  `max_dets=300` on the queried (image, category) pairs, including verified-absent categories;
  it also reports APr, APc and APf. Build the TSV by running
  [`scripts/prepare_lvis_positive_only.py`](scripts/prepare_lvis_positive_only.py), which moves
  the images into `$LMUData/images/LVIS`, and then
  [`scripts/prepare_lvis_federated.py`](scripts/prepare_lvis_federated.py). The annotations are
  read from `LVIS_GT_PATH`, `LVIS_DATA_ROOT` or `$LMUData/lvis-api/data/`.

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
wraps the model in an OpenAI-compatible client (`gpt-4o` resolves to `gpt-4o-2024-05-13`; other
short names in `build_judge` map likewise). `api_base` is the full chat-completions URL, for
example `https://api.openai.com/v1/chat/completions`. With `key` omitted or set to `empty`, a
judge on `api.openai.com` reads `OPENAI_API_KEY`; other endpoints receive the value as
written. `${NAME}` placeholders resolve from the environment, as in the agent presets:
`configs/judge.yaml` reads `JUDGE_MODEL`, `JUDGE_API_BASE` and `JUDGE_API_KEY`. The extra keys
that matter for the
paper benchmarks are below; a few loaders read other flags (`use_verifier`, `enable_vdr_aux`,
`use_exact_match`), all off by default.

| Key | Values | Read by |
| --- | --- | --- |
| `judge_prompt` | `judge_vdr` (default) and `judge_rs` on every search loader; `hle` on the others, while RealXBench, MMBrowseComp and VisBrowseBench take their original judges `realxbench`, `mmbrowsecomp` and `vbb` instead | the search loaders; the `MMDR_JUDGE_PROMPT` environment variable overrides it |
| `judge_max_tokens` | output-token cap per judge call, default 2048; SimpleVQA's judge call sets no cap | `build_judge`, `resolve_judge_max_tokens` |
| `score_mode` | `constant` or `original` | the grounding loaders |

The `judge_prompt` prompts live in `utils/hle_judge.py`, `utils/judge_rs.py`,
`utils/judge_vdr.py`, `utils/realxbench_judge.py`, `utils/mmbrowsecomp_judge.py` and
`utils/visbrowse_bench_judge.py`; the fixed SimpleVQA and FREAK_VQA prompts live in
`simplevqa.py` and `utils/verifier.py`.
The benchmark table gives the code defaults, which are the paper's; the paper's search and
deep-research tables use GPT-4o as the judge model, and the judge behind each paper table is
stated in the paper's appendix. The table's last column means:

- **judge**: the model decides correctness. The eleven search benchmarks, SimpleVQA and
  FREAK_VQA need one; the last two use fixed prompts and ignore `judge_prompt`.
- **strict count parse**: `parse_count` (`utils/fsc147.py`) reads the last number in the
  response, after keeping only the last `\boxed{...}` content and removing thousands
  separators. A minus sign counts only when it does not follow a word character, so `3-5`
  reads as 5. Empty responses, agent failure messages, number words, non-integer numbers and
  negative numbers are unparseable and scored as wrong (they enter MAE and RMSE as 0). No judge model is
  called, whatever the judge config says. This is the protocol of the paper's OctoTools and
  Visual Sketchpad counting cells.
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
   the `index` of a row whose image to reuse) or `image_path` (absolute: the default
   `build_prompt` passes it through unchanged, and only a `build_prompt` that calls
   `dump_image` resolves relative paths against `$LMUData/images/<name>`). Multiple-choice
   TSVs add option columns `A`, `B`, ...
2. Subclass `ImageMCQDataset`, `ImageVQADataset` or `ImageYORNDataset` to reuse their prompt
   and scoring, or `ImageBaseDataset`, whose default prompt is the images followed by the
   question and whose `evaluate` does nothing.
3. Set `DATASET_URL` and `DATASET_MD5`, keyed by the `--data` name (`supported_datasets()`
   returns the `DATASET_URL` keys). Use `hf_tsv_url(name)` for a split hosted in the evaluation
   repository; with an empty URL the loader reads `$LMUData/<name>.tsv`.
4. Override `build_prompt(line)` if needed; it returns `[{'type': 'image', 'value': path},
   ..., {'type': 'text', 'value': prompt}]`. Implement `evaluate(eval_file, **judge_kwargs)`
   when the base class's does not fit, which is always the case for a direct `ImageBaseDataset`
   subclass; it scores the prediction file, writes score files next to it and returns the
   metrics.
5. Import the class in [`__init__.py`](__init__.py) and add it to `IMAGE_DATASET`; when two
   classes serve the same name, the earlier one wins.
