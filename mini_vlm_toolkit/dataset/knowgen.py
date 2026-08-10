"""KnowGen-Bench dataset (Gen-Searcher paper).

Knowledge-grounded text-to-image benchmark introduced in
"Reinforcing Agentic Search for Image Generation" (Gen-Searcher, arXiv
2603.28767). 630 expert-crafted prompts across 20 categories, each paired
with a ground-truth reference image produced by the paper's own agent.

The task: given a text prompt (no input image), the model must produce an
image that satisfies knowledge-grounded constraints (correct year, official
name, specific venue, in-image text, etc.). Evaluation compares the
generated image against the GT reference image using GPT-4.1 as a
multimodal judge on four dimensions (faithfulness / visual_correctness /
text_accuracy / aesthetics) — each in {0, 0.5, 1}. See
``mini_vlm_toolkit/dataset/utils/judge_knowgen.py`` for the byte-exact
paper prompt.

Source: HuggingFace dataset ``GenSearcher/KnowGen-Bench``. First load
downloads the snapshot (~200MB of PNG GT images + a ~600KB JSON) and
caches a plaintext TSV under ``$LMUData/KnowGen.tsv``.

Record schema (cached TSV):
  - ``index``     : str (dataset row id, e.g. ``"7012"``)
  - ``prompt``    : str (task prompt — image generation instruction)
  - ``gt_image``  : str (absolute path to the ground-truth reference PNG)
  - ``category``  : str (top-level meta.category, e.g. ``"multi-subject-Game"``)
  - ``l2-category`` : str (meta.difficulty, e.g. ``"easy"``, ``"multiple subject"``)

Prediction resolution: the framework's prediction column stores the
agent's ``<answer>`` text. To recover the actual GENERATED IMAGE we walk
the per-run images directory (``<pred_root>/full_trajectories/images/``)
and take the LAST PNG associated with each task_id — that's the model's
final produced image (see ``_resolve_generated_image``).

Judge: only ``judge_knowgen`` is supported (Gen-Searcher paper protocol).
Consider WISE / T2I-FactualBench (planned) for cross-benchmark
triangulation on the same generation model output.
"""

import glob
import json
import os
import os.path as osp
# ⛔ Must be explicit: `from ..smp import *` does NOT re-export `re`, and the
# provenance regex at module level made the whole mini_vlm_toolkit.dataset package
# fail to import (NameError raised through dataset/__init__.py, killing EVERY
# benchmark, not just KnowGen). pyflakes did flag it, as "'re' may be undefined,
# or defined from star imports" -- a warning that is easy to dismiss as star-import
# noise but was real here.
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich


# ---------------------------------------------------------------------------
# Source data — HuggingFace snapshot
# ---------------------------------------------------------------------------
_HF_REPO_ID = 'GenSearcher/KnowGen-Bench'
_HF_METADATA_PATH = 'KnowGen-Bench.json'  # 630 rows dict keyed by id


def _snapshot_download_knowgen() -> str:
    """Download the KnowGen-Bench HF snapshot and return its local root.

    First call is slow (~200MB of gt_image/*.png); subsequent calls are
    near-instant thanks to HF cache.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=_HF_REPO_ID, repo_type='dataset')


# ---------------------------------------------------------------------------
# Per-sample grader
# ---------------------------------------------------------------------------

def _grade_item_judge_knowgen(model, item):
    """Invoke the KnowGen paper judge on one sample.

    Requires:
      - ``item['prompt']``          : original image-generation prompt
      - ``item['gt_image']``        : GT reference image path (absolute)
      - ``item['generated_image']`` : path to the model's generated image
      - ``item['index']``           : sample id (used for logging)

    Returns a dict with ``correct`` (mean-of-four >= 0.5) plus the raw
    four dimension scores and rationale for downstream analysis.
    """
    from .utils.judge_knowgen import (
        build_judge_knowgen_messages,
        parse_judge_knowgen_judgement,
    )

    prompt = str(item.get('prompt', ''))
    gt_path = str(item.get('gt_image', ''))
    gen_path = str(item.get('generated_image', ''))
    sample_id = str(item.get('index', ''))

    if not gen_path or not osp.isfile(gen_path):
        return dict(
            correct=0,
            mean_score=0.0,
            faithfulness=0.0,
            visual_correctness=0.0,
            text_accuracy=0.0,
            aesthetics=0.0,
            text_accuracy_na=False,
            rationale='',
            reasoning=f'generated image missing: {gen_path!r}',
            judge_raw='',
        )
    if not gt_path or not osp.isfile(gt_path):
        return dict(
            correct=0,
            mean_score=0.0,
            faithfulness=0.0,
            visual_correctness=0.0,
            text_accuracy=0.0,
            aesthetics=0.0,
            text_accuracy_na=False,
            rationale='',
            reasoning=f'GT image missing: {gt_path!r}',
            judge_raw='',
        )

    messages = build_judge_knowgen_messages(
        sample_id=sample_id,
        prompt=prompt,
        generated_image_path=gen_path,
        gt_image_path=gt_path,
    )

    retry = 3
    last = None
    while retry:
        try:
            out = model.generate(messages)
        except Exception as e:
            out = f'Failed to obtain answer via API: {e}'
        if 'Failed to obtain answer via API' in str(out):
            retry -= 1
            continue
        last = out
        parsed = parse_judge_knowgen_judgement(out)
        if parsed['verdict']:
            return dict(
                correct=parsed['correct'],
                mean_score=parsed['mean_score'],
                faithfulness=parsed['faithfulness'],
                visual_correctness=parsed['visual_correctness'],
                text_accuracy=parsed['text_accuracy'],
                aesthetics=parsed['aesthetics'],
                text_accuracy_na=parsed['text_accuracy_na'],
                rationale=parsed['rationale'],
                reasoning=parsed['reasoning'],
                judge_raw=str(out),   # full raw judge output (for viz + offline debug)
            )
        retry -= 1

    return dict(
        correct=0,
        mean_score=0.0,
        faithfulness=0.0,
        visual_correctness=0.0,
        text_accuracy=0.0,
        aesthetics=0.0,
        text_accuracy_na=False,
        rationale='',
        reasoning=f'judge_knowgen unparseable: {str(last)[:200]}' if last else 'judge unreachable',
        judge_raw=str(last) if last else '',   # keep full raw so truncation/parse
        # failures are debuggable offline instead of lost to a 200-char preview.
    )


# ---------------------------------------------------------------------------
# Generated-image resolution from the run directory
# ---------------------------------------------------------------------------

_GEN_TOOLS = ("Image_Generation_Tool", "Image_Edit_Tool")
_REF_SOURCE_LINE = re.compile(r"^-\s+(tool_generated_image_(\d+)|input_image_\d+):\s*(\S+)", re.M)


def _iter_message_text(messages):
    """Yield every text string in a persisted trajectory's messages."""
    for msg in messages or []:
        content = (msg or {}).get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    v = part.get("text") or part.get("value")
                    if isinstance(v, str):
                        yield v


def _image_provenance(pred_root: str, task_id: str):
    """Read ref_id -> source for one task from its persisted trajectory.

    The image memory module emits an environment-update block listing every
    registered image and where it came from::

        Currently available image IDs for tool calls:
        - input_image_1: user_input
        - tool_generated_image_1: tool_output:Text_Search_Tool
        - tool_generated_image_2: tool_output:Image_Generation_Tool

    That text is part of the saved trajectory, so provenance is recoverable for
    old runs too -- no re-inference needed. Returns ``None`` when there is no
    trajectory or no such block (i.e. provenance is genuinely unknown, which is
    different from "known to have no generated image").
    """
    traj = osp.join(pred_root, 'full_trajectories', f'{task_id}.json')
    if not osp.isfile(traj):
        return None
    try:
        with open(traj) as f:
            payload = json.load(f)
    except Exception:
        return None
    prov = {}
    for text in _iter_message_text(payload.get('messages')):
        for ref, num, source in _REF_SOURCE_LINE.findall(text):
            prov[ref] = source
    return prov or None


def _resolve_generated_image(pred_root: str, task_id: str) -> Optional[str]:
    """Find the image the agent actually GENERATED for ``task_id``.

    ⛔ Why this is not just "the last saved image": with ``--save-images files``
    the trajectory writer decodes EVERY data-URI image in message order,
    including the images returned by search/retrieval tools
    (``reporting_utils._replace_data_uris_with_files`` walks all messages
    "regardless of role"). Picking the highest numeric suffix therefore scored
    the last RETRIEVED image as the model's generation -- so an agent that only
    searched, and never generated anything, was credited with a real score. That
    is a fabricated number, which is worse than a zero.

    So the image is attributed through provenance (see ``_image_provenance``):
      * no image-generation/edit tool output at all -> None, i.e. the agent did
        not generate an image and the sample legitimately scores 0;
      * the most recent tool output IS a generation/edit -> the last saved file
        is that output;
      * a generation exists but something else was produced afterwards -> the
        file cannot be attributed, so return None rather than guess.

    Set ``MVTK_GEN_IMAGE_LEGACY_LASTFILE=1`` to force the old last-file
    heuristic. It exists only to REPRODUCE pre-fix numbers on purpose; it will
    score retrieved images as generations.
    """
    img_dir = osp.join(pred_root, 'full_trajectories', 'images')
    if not osp.isdir(img_dir):
        return None

    matches = []
    for ext in ('png', 'jpg', 'jpeg', 'webp'):
        matches.extend(glob.glob(osp.join(img_dir, f'{task_id}_*.{ext}')))
    if not matches:
        return None

    def _sort_key(p: str):
        stem = osp.basename(p).rsplit('.', 1)[0]  # e.g., "123_5"
        try:
            return int(stem.rsplit('_', 1)[-1])
        except Exception:
            return 0

    matches.sort(key=_sort_key)
    last_file = matches[-1]

    if os.environ.get('MVTK_GEN_IMAGE_LEGACY_LASTFILE') == '1':
        return last_file

    prov = _image_provenance(pred_root, task_id)
    if prov is None:
        # Provenance unknown (no trajectory json, or one written without the image
        # memory module). Refuse rather than silently reintroducing the heuristic that
        # scored retrieved images.
        return None

    # tool_generated_image_N is numbered in registration order, i.e. chronologically,
    # and _replace_data_uris_with_files numbers <task_id>_<idx> in that same
    # top-to-bottom message order. So the k-th tool-generated ref corresponds to the
    # k-th saved file, and we can pick the file belonging to the LAST generation/edit
    # ref rather than assuming it is the last file overall.
    tool_refs = sorted(
        ((int(m.group(1)), ref) for ref in prov
         for m in [re.match(r'^tool_generated_image_(\d+)$', ref)] if m),
        key=lambda t: t[0],
    )
    gen_positions = [i for i, (_n, ref) in enumerate(tool_refs)
                     if any(t in prov.get(ref, '') for t in _GEN_TOOLS)]
    if not gen_positions:
        return None                      # the agent never generated an image

    # ⛔ Do NOT fall back to "last file overall" once a generation exists. Two verified
    # failure modes of that shortcut:
    #   1) generate -> verify-by-search trajectories (refs [GEN, SEARCH]) scored 0 even
    #      though a real generation existed -- and our prompts actively encourage
    #      post-generation verification, so it punished the better-behaved runs;
    #   2) files left in the images dir by an EARLIER run kept a higher numeric suffix,
    #      so a fresh run that wrote T1_0..T1_1 was scored on a stale T1_4.
    # Indexing by position fixes both, and the count check below catches the stale-file
    # case that positional indexing alone cannot see.
    n_tool = len(tool_refs)
    n_input = sum(1 for ref in prov if ref.startswith('input_image_'))
    expected = n_input + n_tool
    if len(matches) != expected:
        # The file set does not match this run's provenance -- almost always stale files
        # from a previous run in the same directory. Refuse rather than guess.
        return None
    idx = n_input + gen_positions[-1]     # input images are written first, in order
    return matches[idx] if idx < len(matches) else None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class KnowGenDataset(ImageBaseDataset):
    """KnowGen-Bench (Gen-Searcher paper, arXiv 2603.28767).

    Knowledge-grounded text-to-image benchmark. 630 prompts, GT
    reference images bundled. Text-only input; the agent must produce an
    image satisfying grounded constraints (year, official name, in-image
    text, etc.). Judge: GPT-4.1 4-dim rubric per Gen-Searcher paper.

    Since the task input is text-only (no query image), ``build_prompt``
    returns a single text message. Predictions are stored as strings but
    the LLM-generated image is resolved from
    ``<pred_root>/full_trajectories/images/<task_id>_*.png`` at
    evaluate() time.
    """

    TYPE = 'VQA'  # kept 'VQA' for orchestrator compat; actual output is image

    DATASET_URL = {
        'KnowGen': '',
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # -- Data loading: auto-download from HF + cache TSV ------------------

    def load_data(self, dataset):
        data_root = LMUDataRoot()
        tsv_path = osp.join(data_root, f'{dataset}.tsv')

        if not osp.exists(tsv_path):
            df = self._convert_benchmark()
            os.makedirs(data_root, exist_ok=True)
            df.to_csv(tsv_path, sep='\t', index=False)

        self.data_path = tsv_path
        return load(tsv_path)

    @staticmethod
    def _convert_benchmark() -> pd.DataFrame:
        """Download HF snapshot, decode each row, return DataFrame.

        The KnowGen-Bench.json file is a dict keyed by task id; values are
        dicts with ``id, prompt, meta{category, difficulty}, gt_image``
        (``gt_image`` is a repo-relative path like ``./gt_image/answer_
        7012.png``).
        """
        snap_root = _snapshot_download_knowgen()
        meta_path = osp.join(snap_root, _HF_METADATA_PATH)
        if not osp.isfile(meta_path):
            raise FileNotFoundError(
                f'KnowGen-Bench.json not found in HF snapshot: {meta_path}'
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            rows_iter = data.items()
        elif isinstance(data, list):
            rows_iter = enumerate(data)
        else:
            raise ValueError(f'Unexpected KnowGen-Bench.json shape: {type(data)}')

        rows: List[Dict[str, Any]] = []
        for _, obj in rows_iter:
            sample_id = str(obj.get('id', obj.get('index', '')))
            prompt = str(obj.get('prompt', ''))

            meta = obj.get('meta', {}) if isinstance(obj.get('meta'), dict) else {}
            category = str(meta.get('category', 'unknown') or 'unknown')
            difficulty = str(meta.get('difficulty', 'unknown') or 'unknown')

            gt_rel = str(obj.get('gt_image', ''))
            # Strip leading "./" if present, then resolve absolute
            gt_rel = gt_rel.lstrip('./').lstrip('/')
            gt_abs = osp.join(snap_root, gt_rel)
            if not osp.isfile(gt_abs) and gt_rel:
                # Snapshot layouts vary (some nest gt_image/ one level deeper).
                # Fall back to searching the snapshot for the GT basename.
                cands = glob.glob(
                    osp.join(snap_root, '**', osp.basename(gt_rel)), recursive=True
                )
                if cands:
                    gt_abs = cands[0]

            rows.append({
                'index': sample_id,
                'prompt': prompt,
                # keep 'question' as alias so the run harness's default
                # prompt-building path (which reads ``question``) still works
                'question': prompt,
                # placeholder answer for compat with generic evaluate paths
                'answer': '',
                'gt_image': gt_abs,
                'category': category,
                'l2-category': difficulty,
            })

        return pd.DataFrame(rows)

    # -- Prompt: text-only input for image generation ---------------------

    def build_prompt(self, line):
        """Return a single-text message. No input image."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        prompt = str(line.get('prompt', line.get('question', '')))
        # KnowGen is a knowledge-grounded text-to-image GENERATION task. Frame the
        # prompt explicitly as generation so the agent produces an image rather than
        # answering in text (see memory gen_eval_framework_gaps). The judge still
        # scores against meta['prompt'] + the GT reference image.
        instruction = (
            'Text-to-image generation task. The description below is an implicit, '
            'knowledge-grounded prompt: infer what must appear using world knowledge '
            '(search for facts or visual references when helpful), then GENERATE an '
            'image that realizes it. You MUST produce an image using the image '
            'generation tool as your final result; do not answer with text only.\n\n'
            f'Prompt to realize: {prompt}'
        )
        return [dict(type='text', value=instruction)]

    # -- Evaluation: KnowGen judge only -----------------------------------

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate KnowGen predictions with the Gen-Searcher paper judge.

        For each row, resolves the model's generated image from
        ``<pred_root>/full_trajectories/images/<task_id>_*.png`` and
        invokes the 4-dim GPT-4.1 judge on (prompt, gen_image, GT_image).

        Requires the run to have been produced with
        ``--save-images files`` (default for our v5 wrappers).
        """
        logger = get_logger('Evaluation')
        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge prompt / gt_image from dataset metadata
        meta = self.data
        idx_str = lambda v: str(v)
        prompt_map = {idx_str(idx): p for idx, p in zip(meta['index'], meta['prompt'])}
        gt_map = {idx_str(idx): g for idx, g in zip(meta['index'], meta['gt_image'])}

        data['prompt'] = [prompt_map.get(idx_str(i), '') for i in data['index']]
        data['gt_image'] = [gt_map.get(idx_str(i), '') for i in data['index']]

        # Resolve generated images from the run directory
        pred_root = osp.dirname(osp.abspath(eval_file))
        gen_paths = []
        missing_count = 0
        for i in range(len(data)):
            tid = idx_str(data.iloc[i]['index'])
            p = _resolve_generated_image(pred_root, tid)
            if not p:
                missing_count += 1
            gen_paths.append(p or '')
        data['generated_image'] = gen_paths
        if missing_count:
            logger.warning(
                f'{missing_count}/{len(data)} tasks have no resolvable generated image under '
                f'{pred_root}/full_trajectories/images/. Judge will score them as failures.'
            )

        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)
        # Env / kwargs judge selector: KnowGen only supports one judge
        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'judge_knowgen'
        ).strip().lower()
        # Unconditional pop: the or-chain above short-circuits when the env
        # var is set, which would leave judge_prompt in judge_kwargs and leak
        # it into the API payload (cf. livevqa.py).
        judge_kwargs.pop('judge_prompt', None)
        if judge_name != 'judge_knowgen':
            raise ValueError(
                f"KnowGen only supports 'judge_knowgen'; got {judge_name!r}. "
                "Set MMDR_JUDGE_PROMPT=judge_knowgen or omit."
            )

        # Full-detail image input for the judge (framework default is 'low')
        judge_kwargs.setdefault('img_detail', 'high')
        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'KnowGen evaluation requires a working multimodal judge (default gpt-4.1). '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        grade_fn = _grade_item_judge_knowgen
        stem_base = '_knowgen_judge_knowgen'
        grade_stem = f'_grade{stem_base}'
        detail_stem = f'_detail{stem_base}'
        acc_stem = f'_acc{stem_base}'

        # Cache + resume support
        result_file = get_intermediate_file_path(eval_file, grade_stem, 'pkl')
        result_cache = {}
        if osp.exists(result_file):
            result_cache = load(result_file)

        items_to_grade = []
        for i in range(len(data)):
            item = data.iloc[i]
            idx = str(item['index'])
            if idx not in result_cache:
                items_to_grade.append(item)

        if items_to_grade:
            tups = [dict(model=model, item=x) for x in items_to_grade]
            keys = [str(x['index']) for x in items_to_grade]
            res = track_progress_rich(
                grade_fn, tups, nproc=nproc, chunksize=nproc,
                save=result_file, keys=keys,
            )
            result_cache = load(result_file)
            for k, v in zip(keys, res):
                if k not in result_cache:
                    result_cache[k] = v

        # Populate score columns
        for col, default in [
            ('correct', 0),
            ('mean_score', 0.0),
            ('faithfulness', 0.0),
            ('visual_correctness', 0.0),
            ('text_accuracy', 0.0),
            ('aesthetics', 0.0),
            ('text_accuracy_na', False),
            ('rationale', ''),   # full judge rationale (was computed but not persisted)
            ('reasoning', ''),
            ('judge_raw', ''),   # full raw judge output — enables viz + offline debug
        ]:
            data[col] = [
                result_cache.get(str(idx), {}).get(col, default)
                for idx in data['index']
            ]

        # Detailed per-row dump
        detail_file = get_intermediate_file_path(eval_file, detail_stem)
        dump(data, detail_file)

        # Accuracy breakdown by category / difficulty
        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, acc_stem, 'csv')
        dump(acc, score_file)
        logger.info(f'KnowGen [{judge_name}] accuracy saved to {score_file}')

        return acc

    @staticmethod
    def _compute_accuracy(data) -> pd.DataFrame:
        """Aggregate scores overall + per category + per difficulty."""
        rows = []

        def _bloc(df: pd.DataFrame, name: str):
            if len(df) == 0:
                return None
            row = {'group': name, 'n': len(df)}
            for k in ('correct', 'mean_score', 'faithfulness',
                      'visual_correctness', 'text_accuracy', 'aesthetics'):
                if k in df.columns:
                    vals = pd.to_numeric(df[k], errors='coerce').fillna(0.0)
                    row[k] = float(vals.mean())
            return row

        # Overall
        blk = _bloc(data, 'Overall')
        if blk is not None:
            rows.append(blk)

        # Per top-level category
        if 'category' in data.columns:
            for cat, sub in data.groupby('category'):
                blk = _bloc(sub, f'category:{cat}')
                if blk is not None:
                    rows.append(blk)

        # Per difficulty
        if 'l2-category' in data.columns:
            for diff, sub in data.groupby('l2-category'):
                blk = _bloc(sub, f'difficulty:{diff}')
                if blk is not None:
                    rows.append(blk)

        return pd.DataFrame(rows)
