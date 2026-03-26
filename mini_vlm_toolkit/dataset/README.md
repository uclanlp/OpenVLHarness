# Dataset Module

This module implements the dataset loading, prompt construction, and evaluation logic for all supported benchmarks. Every dataset follows a three-phase lifecycle:

1. **Data Loading** — download/cache a TSV file and parse it into a pandas DataFrame
2. **Prompt Building** — construct a multi-modal message list (`[{type, value}, ...]`) for the model
3. **Evaluation** — parse model predictions and compute task-specific metrics

---

## Architecture

### Class Hierarchy

```
ImageBaseDataset  (image_base.py)      MODALITY = 'IMAGE'
├── ImageMCQDataset      (image_mcq.py)      TYPE = 'MCQ'
├── ImageVQADataset      (image_vqa.py)      TYPE = 'VQA'
├── ImageYORNDataset     (image_yorn.py)     TYPE = 'Y/N'
├── ImageShortQADataset  (image_shortqa.py)  TYPE = 'Short'
├── ImageCaptionDataset  (image_caption.py)  TYPE = 'Caption'
├── RefCOCODataset       (refcoco.py)        TYPE = 'GROUNDING'
├── LVISBBoxDataset      (lvis.py)           TYPE = 'GROUNDING'
├── ODinWBBoxDataset     (odinw.py)          TYPE = 'GROUNDING'
├── RefSpatialBench      (refspatialbench.py)TYPE = 'VQA'
└── (many specialized subclasses ...)

TextBaseDataset   (text_base.py)       MODALITY = 'TEXT'
└── TextMCQDataset       (text_mcq.py)       TYPE = 'MCQ'
```

### Dataset Registration

All dataset classes are imported and registered in `__init__.py`. The two master lists are:

- `IMAGE_DATASET` — all image-based dataset classes
- `TEXT_DATASET` — text-only dataset classes

The `build_dataset(dataset_name)` function iterates over these lists, matches the name against each class's `supported_datasets()`, and instantiates the first match. If no match is found, it falls back to loading a local TSV and auto-detecting whether it's MCQ (has columns `A`, `B`) or VQA.

Each class exposes two key class-level attributes:
- **`MODALITY`**: `'IMAGE'` or `'TEXT'` — determines the inference pipeline
- **`TYPE`**: `'MCQ'`, `'VQA'`, `'Y/N'`, `'Short'`, `'Caption'`, `'GROUNDING'` — determines the prompt suffix and evaluation strategy

---

## Base Classes

### `ImageBaseDataset` (`image_base.py`)

The foundation for all image benchmarks.

| Method | Description |
|--------|-------------|
| `__init__(dataset, skip_noimg)` | Downloads/loads TSV via `prepare_tsv()`, decodes base64 images, builds image map |
| `load_data(dataset)` | Downloads TSV from `DATASET_URL` with MD5 validation |
| `prepare_tsv(url, file_md5)` | Handles download, caching, and localization of large (>1 GB) files |
| `dump_image(line)` | Decodes base64 image data to JPEG on disk, returns file path(s) |
| `build_prompt(line)` | Returns `[{type:'image', value:path}, {type:'text', value:question}]` |
| `evaluate(eval_file, **judge_kwargs)` | **Abstract** — must be implemented by subclasses |

**Data format**: TSV with required columns `index`, `question`, and either `image` (base64) or `image_path`. The `image` field can reference another row's index (for deduplication when multiple questions share one image).

### `TextBaseDataset` (`text_base.py`)

Minimal base for text-only benchmarks. `build_prompt()` returns `[{type:'text', value:question}]` with no image.

---

## Task Type Evaluation

Each task type has a distinct approach to prompt construction, answer parsing, and metric computation. The table below summarizes the key differences:

| Type | Prompt Suffix | Answer Parsing | Primary Metric | LLM Judge |
|------|---------------|----------------|----------------|-----------|
| **MCQ** | `"Please select the correct answer from the options above."` | Extract letter A-Z | Accuracy (hit 0/1) | Fallback if heuristic fails |
| **VQA** | `"Answer the question using a single word or phrase."` | Normalize text | vqa_score / anls / relaxed_accuracy / accuracy | Optional answer extraction |
| **Y/N** | (none, or `"Please answer yes or no."` for AMBER) | Regex extract Yes/No | Dataset-specific rating | Fallback for ambiguous cases |
| **Short** | `"Please directly provide a short answer to the question."` | LLM correctness check | Accuracy (hit 0/1) | Always used |
| **Caption** | (none) | Direct comparison | BLEU, ROUGE-L, CIDEr | Not used |
| **Grounding** | (none) | Parse bbox [x1,y1,x2,y2] | Precision@IoU0.5 | Not used |

### MCQ — `ImageMCQDataset` (`image_mcq.py`)

**Prompt construction:**
```
Hint: {hint}                        ← optional
Question: {question}
Options:
A. {option_A}
B. {option_B}
...
Please select the correct answer from the options above.
```

**Answer parsing** (two-stage):
1. **Heuristic** (`can_infer(prediction, choices)` in `utils/multiple_choice.py`): regex-based extraction of a single letter from the prediction string.
2. **LLM fallback** (`extract_answer_from_item(model, item)`): sends prediction + options to judge model with the prompt *"Find which option is most similar to the answer"*, extracts a letter. Returns `Z` if no match.

**Evaluation modes:**
- **Vanilla eval** (`mcq_vanilla_eval`): each prediction is scored independently. `hit = 1` if extracted letter matches the gold answer.
- **Circular eval** (`mcq_circular_eval`): for datasets like MMBench that present each question with multiple option orderings. A question is `hit = 1` only if all circular variants are correct.

**Result aggregation:**
- `report_acc(df)`: groups by `split`, `l2-category`, `category`. Uses MMBench abbreviations (CP, FP-S, FP-C, LR, RR, AR).
- `report_acc_MMT(df)`: for MMT-Bench with 21+ task categories (VR, Loc, OCR, Count, etc.).

**Key files:** `image_mcq.py`, `utils/multiple_choice.py`

---

### VQA — `ImageVQADataset` (`image_vqa.py`)

**Prompt construction:** appends `"\nAnswer the question using a single word or phrase."` to the base prompt.

**Evaluation methods** — selected automatically by `_get_eval_method()` based on dataset name:

| Method | Datasets | Logic |
|--------|----------|-------|
| `vqa_score` | TextVQA (default) | Compare prediction against multiple reference answers. Score = `min(1, matching_refs / 3)` averaged over all references. Both prediction and references are normalized (lowercase, remove articles/punctuation, expand contractions, map number words → digits). |
| `anls` | DocVQA, InfoVQA | ANLS metric: `1 - (levenshtein_distance / max_length)`. Threshold at 0.5. |
| `relaxed_accuracy` | ChartQA | Numeric answers tolerate ±5% relative error. Non-numeric requires exact match. |
| `accuracy` | OCRVQA, GQA | Case-insensitive exact string match. |

**Answer normalization** (`process_line()` in `utils/vqa_eval.py`):
- Replace newlines/tabs with spaces
- Remove punctuation (commas, periods, etc.)
- Expand contractions ("don't" → "do not")
- Map number words to digits ("three" → "3")
- Remove articles ("a", "an", "the")

**Optional LLM judge**: When `model` is specified in judge_kwargs, the system uses `vqa_vanilla_eval()` (in `utils/vqa_judge.py`) to first extract a clean short answer from the model's response, then applies the metric. The extraction prompt provides the question, ground-truth format, and model response, and asks the judge to return just the final answer.

**Scoring** (`hit_calculate()` in `utils/vqa_eval.py`):
- TextVQA: `mean(match_scores)` per sample
- DocVQA/InfoVQA: `1 - min(match)` if above ANLS threshold, else 0
- ChartQA/OCRVQA: `max(match_scores)` per sample

**Key files:** `image_vqa.py`, `utils/vqa_eval.py`, `utils/vqa_judge.py`

---

### Y/N — `ImageYORNDataset` (`image_yorn.py`)

**Answer extraction** (two-stage):
1. **Regex** (`YOrN_Extraction(prediction)` in `utils/yorn.py`): case-insensitive regex for "Yes" or "No". Returns "Unknown" if ambiguous.
2. **LLM fallback** (`YOrN_auxeval(model, line)`): for "Unknown" predictions, asks judge to extract a Yes/No answer.

**Dataset-specific rating functions** (in `utils/yorn.py`):

| Function | Dataset | Metrics |
|----------|---------|---------|
| `MME_rating()` | MME | Per-category accuracy across 10 perception + 4 reasoning categories. Scores are doubled (sum of accuracy + per-image-pair accuracy). |
| `Hallusion_rating()` | HallusionBench | Three metrics: aAcc (answer accuracy), fAcc (figure accuracy), qAcc (question accuracy). |
| `POPE_rating()` | POPE | F1 score + precision + recall per category (random, popular, adversarial). |
| `AMBER_rating()` | AMBER | Per-category accuracy for attribute/existence/relation. Case-insensitive matching. |
| `VSR_rating()` | VSR-zeroshot | Precision, recall, F1. |
| `default_rating()` | (others) | Simple accuracy per category. |

**Key files:** `image_yorn.py`, `utils/yorn.py`

---

### Short QA — `ImageShortQADataset` (`image_shortqa.py`)

**Prompt construction:** appends `"\nPlease directly provide a short answer to the question."` to the base prompt.

**Evaluation**: always uses an LLM judge. Two sub-paths:

1. **If the item has MCQ options (columns A, B, ...):**
   - Single item → `eval_vanilla()` (MCQ letter extraction)
   - Grouped items (circular eval) → `eval_circular_group()`

2. **If free-form short answer** → `ShortQA_auxeval(model, line)`:
   - Sends to judge with structured format:
     ```
     [Begin Correctness] yes/no [End Correctness]
     [Begin Reason] ... [End Reason]
     ```
   - Retries up to 3 times with increasing temperature (0, 0.5, 1.0)
   - Returns `{hit: 1/0, log: reason}`

**Key files:** `image_shortqa.py`, `utils/shortqa.py`

---

### Caption — `ImageCaptionDataset` (`image_caption.py`)

**Prompt**: auto-generates `"Please describe this image in general..."` if no question column exists.

**Evaluation**: uses `COCO_Caption_Scorer` (requires `pycocoevalcap`).

**Metrics:**
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: n-gram precision
- **ROUGE-L**: longest common subsequence F-measure
- **CIDEr**: consensus-based image description evaluation

No LLM judge is used. Returns a dict: `{'Bleu': [b1,b2,b3,b4], 'ROUGE_L': score, 'CIDEr': score}`.

**Key files:** `image_caption.py`

---

### Grounding — `RefCOCODataset` (`refcoco.py`)

**Data loading**: parses bounding box coordinates from the `answer` column. Auto-detects whether coordinates are normalized (0-1 range) or pixel-absolute, and converts accordingly.

**Answer parsing** (`_parse_prediction(pred_text)`): extracts `[x1, y1, x2, y2]` from the model output. Handles JSON arrays, tuples, and various text formats.

**Evaluation**:
1. Parse predicted bbox from model output
2. Convert to absolute coordinates using image width/height
3. Compute IoU between predicted and ground-truth bboxes
4. `hit = 1` if `IoU >= 0.5`

**Metrics**: Precision@1 (accuracy at IoU ≥ 0.5) and Average IoU, reported per split (RefCOCO_val, RefCOCO_testA, RefCOCO_testB, RefCOCO+_*, RefCOCOg_*) and overall.

**Key files:** `refcoco.py`

---

### Object Detection — `LVISBBoxDataset` (`lvis.py`), `ODinWBBoxDataset` (`odinw.py`)

These datasets evaluate open-vocabulary object detection. The model receives an image and a category name, and must output a list of bounding boxes.

**Answer parsing**: robust extraction of bbox arrays from model text. Supports JSON arrays, markdown code blocks, and regex fallback for 4-number tuples. Auto-normalizes coordinates (detects absolute vs. normalized).

**Evaluation**: greedy IoU matching between predicted and ground-truth boxes at IoU threshold 0.5.

**Metrics**: Precision@IoU50, Recall@IoU50, F1@IoU50, Mean IoU, Count Accuracy. LVIS breaks down by frequency band; ODinW breaks down by domain.

**Key files:** `lvis.py`, `odinw.py`

---

### Spatial Referring — `RefSpatialBench` (`refspatialbench.py`)

**Evaluation**: point-in-mask evaluation. The model outputs point coordinates and the evaluation checks whether those points fall within the ground-truth segmentation mask.

Uses `Point2DParser` to extract `(x, y)` coordinates from JSON or tuple formats.

**Metrics**: success rate (percentage of predicted points inside the mask), reported per split (Location, Placement, Unseen) and overall.

**Key files:** `refspatialbench.py`, `refspatial.py`

---

## LLM-as-Judge System

Many task types optionally (or always) use an LLM to help parse or evaluate model outputs. The judge is configured via `configs/judge.yaml`:

```yaml
judge_config_version: v1_0215
model_name: Qwen/Qwen3-VL-8B-Instruct
api_base: http://localhost:8222/v1/chat/completions
key: empty
retry: 3
nproc: 4
```

### How the Judge is Built

`build_judge(**kwargs)` in `utils/judge_util.py`:
- Maps short model names to full identifiers (e.g., `'gpt-4o-mini'` → `'gpt-4o-mini-2024-07-18'`)
- Wraps with `OpenAIWrapper` (for API models) or `HFChatModel` (for local HuggingFace models)
- Respects `LOCAL_LLM` env var to override model selection

### When the Judge is Used

| Task Type | Judge Role | Trigger |
|-----------|-----------|---------|
| MCQ | Extract choice letter from verbose responses | When `can_infer()` heuristic fails |
| VQA | Extract short answer from long responses | When `model` is specified in judge config |
| Y/N | Disambiguate unclear Yes/No responses | When regex extraction returns "Unknown" |
| Short QA | Judge correctness of free-form answers | Always (required for scoring) |
| Caption | — | Not used |
| Grounding | — | Not used |

If `model` is set to `'exact_matching'` (or not specified), no LLM judge is used and the system relies purely on heuristic parsing.

---

## Utilities Reference

### `utils/multiple_choice.py`
- `build_choices(item)` — extract options A-Z from a DataFrame row
- `can_infer(prediction, choices)` — heuristic letter extraction via regex
- `extract_answer_from_item(model, item)` — LLM-based option matching
- `report_acc(df)` — aggregate accuracy by split/category with MMBench abbreviations
- `report_acc_MMT(df)` — aggregate for MMT-Bench categories
- `mcq_vanilla_eval()` / `mcq_circular_eval()` — batch evaluation with caching

### `utils/vqa_eval.py`
- `process_line(line, method)` — normalize and match prediction against ground truth
- `process_answer(answer)` — text normalization pipeline
- `hit_calculate(result, dataset_name)` — per-dataset scoring logic
- `relaxed_correctness(target, prediction)` — numeric ±5% tolerance check
- `anls_compute(groundtruth, prediction)` — ANLS via Levenshtein distance
- `levenshtein_distance(s1, s2)` — character-level edit distance

### `utils/vqa_judge.py`
- `build_vqa_extract_prompt()` — construct the answer extraction prompt for VQA
- `extract_answer_from_vqa_item(model, item)` — per-item LLM answer extraction
- `vqa_vanilla_eval(model, data, nproc, ...)` — batch VQA evaluation with judge

### `utils/yorn.py`
- `YOrN_Extraction(prediction)` — regex-based Yes/No extraction
- `YOrN_auxeval(model, line)` — LLM-based Yes/No extraction
- `MME_rating()`, `POPE_rating()`, `Hallusion_rating()`, `AMBER_rating()`, `VSR_rating()`, `default_rating()` — dataset-specific metric computations

### `utils/judge_util.py`
- `build_judge(**kwargs)` — instantiate judge model with proper API configuration

---

## Dataset Collections

### `ConcatDataset` (`__init__.py`)

Aggregates multiple image datasets into a single evaluation unit. Used for multi-lingual benchmarks:

```python
DATASET_SETS = {
    'MMMB': ['MMMB_ar', 'MMMB_cn', 'MMMB_en', 'MMMB_pt', 'MMMB_ru', 'MMMB_tr'],
    'MTL_MMBench_DEV': ['MMBench_dev_ar', 'MMBench_dev_cn', ...],
    'ScreenSpot': ['ScreenSpot_Mobile', 'ScreenSpot_Desktop', 'ScreenSpot_Web'],
    ...
}
```

Mechanics:
- Adds `SUB_DATASET` column to track origin
- Remaps indices via `original_index` to avoid collisions
- `evaluate()` splits results back by sub-dataset, evaluates each independently, then aggregates

---

## Adding a New Dataset

### Step 1: Choose a Base Class

| Your task type | Base class | File to subclass |
|---------------|------------|-----------------|
| Image + multiple choice | `ImageMCQDataset` | `image_mcq.py` |
| Image + free-form answer | `ImageVQADataset` | `image_vqa.py` |
| Image + yes/no | `ImageYORNDataset` | `image_yorn.py` |
| Image + short answer | `ImageShortQADataset` | `image_shortqa.py` |
| Image + captioning | `ImageCaptionDataset` | `image_caption.py` |
| Image + bounding box | `RefCOCODataset` | `refcoco.py` |
| Text-only MCQ | `TextMCQDataset` | `text_mcq.py` |
| Something unique | `ImageBaseDataset` | `image_base.py` |

### Step 2: Implement Required Methods

At minimum:

```python
class MyDataset(ImageMCQDataset):   # or whichever base
    TYPE = 'MCQ'  # inherited if subclassing a typed class

    DATASET_URL = {
        'MyDataset': 'https://example.com/MyDataset.tsv',
    }
    DATASET_MD5 = {
        'MyDataset': 'abc123...',
    }

    # Override if you need custom data loading
    def load_data(self, dataset):
        ...

    # Override if you need custom prompt formatting
    def build_prompt(self, line):
        ...

    # Override if you need custom evaluation
    def evaluate(self, eval_file, **judge_kwargs):
        ...
```

If subclassing `ImageMCQDataset`, `ImageVQADataset`, `ImageYORNDataset`, etc., you inherit a working `evaluate()` — only override if you need custom metrics.

If subclassing `ImageBaseDataset` directly, you **must** implement `evaluate()`.

### Step 3: Register

In `__init__.py`:

```python
from .my_dataset import MyDataset

# Add to the appropriate list:
IMAGE_DATASET = [
    ...,
    MyDataset,
]
```

### TSV Data Format

The standard TSV should contain at minimum:

| Column | Required | Description |
|--------|----------|-------------|
| `index` | Yes | Unique identifier for each example |
| `question` | Yes | The question text |
| `image` | One of these | Base64-encoded image data |
| `image_path` | One of these | Path to image file |
| `answer` | For eval | Ground-truth answer |
| `A`, `B`, `C`, `D` | MCQ only | Answer options |
| `hint` | Optional | Additional context shown before the question |
| `category` | Optional | For per-category metric breakdown |
| `split` | Optional | For per-split metric breakdown |
