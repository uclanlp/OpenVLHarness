"""SearchGen-Bench dataset (SearchGen paper).

Knowledge-intensive image-generation benchmark introduced in "Search Beyond
What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual
Generation" (arXiv 2607.05382; GitHub HaozheH3/SearchGen). 751 test
prompts (551 en / 200 zh; 571 i2i / 179 t2i by upstream's task-type tag)
across 22 domains and 12 failure-mode categories. Each sample carries a
verification checklist (3-10 items), a weighted task-specific rubric, 0-6
visual reference images and 0-4 textual knowledge-gap slots — all
JUDGE-SIDE ONLY. The agent receives just the user prompt and must search
for whatever knowledge the prompt needs before generating the image
(that's the point of the benchmark).

Source: HF dataset ``JasperHaozhe/SearchGen-Bench``. First load downloads
``metadata/eval_metadata.jsonl`` (~1.8MB) and, if reference images are not
already staged, ``data/references/bench-000000.tar`` (~288MB; 1099 images)
which is extracted under ``$LMUData/images/SearchGenBench/`` (override
with ``SEARCHGEN_IMAGE_ROOT``). A plaintext TSV is cached at
``$LMUData/SearchGenBench.tsv``.

Record schema (cached TSV):
  - ``index``                  : str (upstream bench_id, e.g. ``bench_0001``)
  - ``prompt`` / ``question``  : str (user_prompt — the generation task)
  - ``answer``                 : '' (compat placeholder)
  - ``language``               : 'en' | 'zh'
  - ``category``               : upstream generation_task_type ('i2i'/'t2i')
  - ``domains`` / ``failure_modes``          : JSON lists (metadata)
  - ``verification_checklist``               : JSON list (judge-side)
  - ``evaluation_rubric``                    : JSON dict (judge-side)
  - ``visual_reference_slots``               : JSON list; ``relative_path``
    entries resolve against the image root at evaluate() time (judge-side)
  - ``text_knowledge_slots``                 : JSON list (judge-side)

Prediction resolution: same as KnowGen — the agent's final generated image
is recovered from ``<pred_root>/full_trajectories/images/<task_id>_*.png``
(requires ``--save-images files``); see
``knowgen._resolve_generated_image``.

Judge: only ``judge_searchgen`` (verbatim upstream protocol; see
``dataset/utils/judge_searchgen.py``). One multimodal LLM call per sample
scoring 10 components on 0-3 (half points). Aggregation follows upstream
``scoring.py`` exactly: per-sample component means -> cross-sample
component means (samples where a component is absent are SKIPPED for that
component) -> overall_mean = mean of present component means. We
additionally report the x100/3 scale used in the paper tables.

Known limitation vs the paper: the NoSearch(100)/Search-Intensive(651)
strata and the Easy/Medium/Hard tiers are MANUALLY CURATED by the authors
and not released in any metadata file (confirmed against the HF repo and
paper), so per-stratum numbers cannot be reproduced; we report full-751
overall + per-component + per-language/task-type. Full-751 baselines can
be derived from the paper's per-stratum tables as 100/651-weighted means.

Deviation (ours only; documented): a sample whose agent produced NO final
image is scored 0 on every component that is applicable a-priori
(checklist, adaptive_rubric, the 5 always-on generics, plus
visual_reference/text_reference when the sample has such slots;
text_rendering is left absent since its applicability is judge-decided).
Upstream has no missing-image case — its generators always emit an image —
but silently skipping such samples would inflate agent scores.
"""

import json
import os
import os.path as osp
import tarfile
from typing import Any, Dict, List, Optional

import pandas as pd

from .image_base import ImageBaseDataset
from .knowgen import _resolve_generated_image
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..smp.file import LMUDataRoot, get_intermediate_file_path
from ..utils import track_progress_rich


# ---------------------------------------------------------------------------
# Source data — HuggingFace
# ---------------------------------------------------------------------------
_HF_REPO_ID = 'JasperHaozhe/SearchGen-Bench'
_HF_METADATA_FILE = 'metadata/eval_metadata.jsonl'
_HF_REFS_TAR = 'data/references/bench-000000.tar'
_N_REF_IMAGES = 1099   # per upstream benchmark_manifest.json


def searchgen_image_root() -> str:
    """Reference-image root: SEARCHGEN_IMAGE_ROOT or $LMUData/images/SearchGenBench."""
    env = os.environ.get('SEARCHGEN_IMAGE_ROOT', '').strip()
    if env:
        return env
    return osp.join(LMUDataRoot(), 'images', 'SearchGenBench')


def _ensure_reference_images(needed_rel_paths: List[str]) -> str:
    """Make sure all reference images exist under the image root.

    If any are missing, download + stream-extract the upstream tar
    (members are ``reference_images/bench_XXXX/ref_XX.<ext>``). Returns the
    image root.
    """
    root = searchgen_image_root()
    missing = [p for p in needed_rel_paths if not osp.isfile(osp.join(root, p))]
    if not missing:
        return root

    from huggingface_hub import hf_hub_download
    tar_path = hf_hub_download(_HF_REPO_ID, _HF_REFS_TAR, repo_type='dataset')
    os.makedirs(root, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, 'r') as tf:
        for m in tf:
            if not m.isfile():
                continue
            # keep the tar's own relative layout; refuse path escapes
            rel = m.name.lstrip('./')
            dst = osp.join(root, rel)
            if not osp.abspath(dst).startswith(osp.abspath(root) + osp.sep):
                raise ValueError(f'tar member escapes image root: {m.name}')
            if osp.isfile(dst):
                continue
            os.makedirs(osp.dirname(dst), exist_ok=True)
            # extractfile() returns None for non-regular members (dirs, links);
            # check before entering the context manager, and close it so we
            # don't leak a file descriptor per extracted image.
            src = tf.extractfile(m)
            if src is None:
                raise ValueError(f'tar member is not a regular file: {m.name}')
            with src, open(dst, 'wb') as f:
                f.write(src.read())
            n += 1
    still = [p for p in needed_rel_paths if not osp.isfile(osp.join(root, p))]
    if still:
        raise FileNotFoundError(
            f'{len(still)} SearchGen-Bench reference images still missing under {root} '
            f'after extracting {tar_path} (first: {still[:3]}). '
            'Check SEARCHGEN_IMAGE_ROOT / re-download the HF tar.'
        )
    return root


# ---------------------------------------------------------------------------
# Per-sample grader
# ---------------------------------------------------------------------------

def _zeroed_components(item) -> Dict[str, Optional[float]]:
    """Missing-image score dict (see module docstring: our documented policy)."""
    from .utils.judge_searchgen import COMPONENTS

    vis_slots = json.loads(item.get('visual_reference_slots') or '[]')
    txt_slots = json.loads(item.get('text_knowledge_slots') or '[]')
    out: Dict[str, Optional[float]] = {c: 0.0 for c in COMPONENTS}
    out['text_rendering'] = None            # applicability is judge-decided
    if not vis_slots:
        out['visual_reference'] = None
    if not txt_slots:
        out['text_reference'] = None
    return out


def _grade_item_judge_searchgen(model, item):
    """Invoke the SearchGen judge on one sample (single multimodal call).

    Requires (from the merged eval frame):
      ``index / prompt / verification_checklist / evaluation_rubric /
      visual_reference_slots / text_knowledge_slots / generated_image /
      _judge_cache_dir / _image_root``

    Returns dict with ``judged`` (bool), per-component values (None ->
    absent), ``scores_json`` (raw labeled 0-3 scores) and ``reasoning``.
    """
    from .utils.judge_searchgen import (
        COMPONENTS,
        build_judge_searchgen_messages,
        component_means_from_scores,
        parsed_judge_labeled_scores_03,
        prepare_judge_image,
        try_parse_judge_output,
    )

    def _result(judged, comps, scores_json='', reasoning=''):
        out = dict(judged=int(judged), scores_json=scores_json, reasoning=reasoning)
        for c in COMPONENTS:
            v = comps.get(c) if comps else None
            out[c] = float(v) if v is not None else None
        return out

    sample_id = str(item.get('index', ''))
    gen_path = str(item.get('generated_image', ''))
    if not gen_path or not osp.isfile(gen_path):
        return _result(True, _zeroed_components(item),
                       reasoning=f'generated image missing: {gen_path!r} -> zeroed (see policy)')

    cache_dir = str(item.get('_judge_cache_dir', ''))
    image_root = str(item.get('_image_root', ''))
    vis_slots = json.loads(item.get('visual_reference_slots') or '[]')
    txt_slots = json.loads(item.get('text_knowledge_slots') or '[]')
    checklist = json.loads(item.get('verification_checklist') or '[]')
    rubric = json.loads(item.get('evaluation_rubric') or '{}')

    try:
        ref_paths = [
            prepare_judge_image(osp.join(image_root, str(s['relative_path'])), cache_dir)
            for s in vis_slots
        ]
        assess_path = prepare_judge_image(gen_path, cache_dir)
    except Exception as e:
        return _result(False, None, reasoning=f'image preprocessing failed: {e}')

    messages = build_judge_searchgen_messages(
        sample_id=sample_id,
        user_prompt=str(item.get('prompt', '')),
        verification_checklist=checklist,
        evaluation_rubric=rubric,
        reference_slots=vis_slots,
        ref_image_paths=ref_paths,
        text_knowledge_slots=txt_slots,
        assess_image_path=assess_path,
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
        try:
            parsed = try_parse_judge_output(str(out))
            if parsed is None:
                raise ValueError('judge response could not be parsed')
            scores = dict(parsed_judge_labeled_scores_03(parsed))
            comps = component_means_from_scores(scores)
            return _result(
                True, comps,
                scores_json=json.dumps(scores, ensure_ascii=False),
                reasoning=' '.join(
                    f'{c}={comps[c]:.2f}' for c in COMPONENTS if comps[c] is not None
                ),
            )
        except Exception:
            retry -= 1

    return _result(False, None,
                   reasoning=f'judge_searchgen unparseable: {str(last)[:200]}' if last else 'judge unreachable')


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SearchGenBenchDataset(ImageBaseDataset):
    """SearchGen-Bench (arXiv 2607.05382). 751 knowledge-intensive
    image-generation prompts; judge-side checklists/rubrics/reference
    images; verbatim upstream 10-component judge. See module docstring.
    """

    TYPE = 'VQA'  # kept 'VQA' for orchestrator compat; actual output is image

    DATASET_URL = {
        'SearchGenBench': '',
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
        """Download eval_metadata.jsonl from HF, return the TSV frame."""
        from huggingface_hub import hf_hub_download

        meta_path = hf_hub_download(_HF_REPO_ID, _HF_METADATA_FILE, repo_type='dataset')

        rows: List[Dict[str, Any]] = []
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                bench_id = str(obj.get('bench_id', ''))
                prompt = str(obj.get('user_prompt', ''))
                if not bench_id or not prompt:
                    raise ValueError(f'malformed eval_metadata row: {line[:200]}')
                vis_slots = obj.get('visual_reference_slots') or []
                for s in vis_slots:
                    rel = str(s.get('relative_path', ''))
                    if not rel or osp.isabs(rel) or '..' in rel.split('/'):
                        raise ValueError(f'{bench_id}: bad reference relative_path {rel!r}')
                rows.append({
                    'index': bench_id,
                    'prompt': prompt,
                    # keep 'question' as alias so the run harness's default
                    # prompt-building path (which reads ``question``) works
                    'question': prompt,
                    'answer': '',
                    'language': str(obj.get('language', '') or 'unknown'),
                    # upstream generation_task_type ('' for 1 row -> unknown)
                    'category': str(obj.get('generation_task_type', '') or 'unknown'),
                    'domains': json.dumps(obj.get('domains') or [], ensure_ascii=False),
                    'failure_modes': json.dumps(obj.get('failure_modes') or [], ensure_ascii=False),
                    'verification_checklist': json.dumps(
                        obj.get('verification_checklist') or [], ensure_ascii=False),
                    'evaluation_rubric': json.dumps(
                        obj.get('evaluation_rubric') or {}, ensure_ascii=False),
                    'visual_reference_slots': json.dumps(vis_slots, ensure_ascii=False),
                    'text_knowledge_slots': json.dumps(
                        obj.get('text_knowledge_slots') or [], ensure_ascii=False),
                })
        if len(rows) != 751:
            get_logger('SearchGenBench').warning(
                f'expected 751 eval rows, got {len(rows)} — upstream metadata may have changed')
        return pd.DataFrame(rows)

    # -- Prompt: text-only input for image generation ---------------------

    def build_prompt(self, line):
        """Single text message: the raw user prompt. No input image, no
        judge-side material (checklist/rubric/references stay hidden —
        the agent must search on its own)."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        prompt = str(line.get('prompt', line.get('question', '')))
        return [dict(type='text', value=prompt)]

    # -- Evaluation: SearchGen judge only ----------------------------------

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate with the verbatim SearchGen-Bench judge protocol.

        Resolves generated images from ``<pred_root>/full_trajectories/
        images/`` (requires ``--save-images files``), stages the reference
        images from HF if needed, runs one judge call per sample, and
        aggregates per upstream scoring.py (skip policy).
        """
        from .utils.judge_searchgen import COMPONENTS

        logger = get_logger('Evaluation')
        data = load(eval_file)
        assert 'prediction' in data, 'prediction column missing from eval file'
        data['prediction'] = [str(x) for x in data['prediction']]

        # Merge judge-side metadata from the dataset TSV
        meta = self.data
        idx_str = lambda v: str(v)
        meta_map = {idx_str(r['index']): r for _, r in meta.iterrows()}
        for col in ('prompt', 'language', 'category', 'verification_checklist',
                    'evaluation_rubric', 'visual_reference_slots', 'text_knowledge_slots'):
            data[col] = [
                str(meta_map[idx_str(i)][col]) if idx_str(i) in meta_map else ''
                for i in data['index']
            ]
        unknown = [idx_str(i) for i in data['index'] if idx_str(i) not in meta_map]
        if unknown:
            raise ValueError(f'{len(unknown)} eval rows not in SearchGenBench metadata: {unknown[:5]}')

        # Stage reference images (only the ones this eval subset needs)
        needed_rels: List[str] = []
        for i in range(len(data)):
            for s in json.loads(data.iloc[i]['visual_reference_slots'] or '[]'):
                needed_rels.append(str(s['relative_path']))
        image_root = _ensure_reference_images(sorted(set(needed_rels)))

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
                f'{pred_root}/full_trajectories/images/. They are scored 0 on applicable components.'
            )

        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)
        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or 'judge_searchgen'
        ).strip().lower()
        # Unconditional pop: the or-chain short-circuits when the env var is
        # set, which would leak judge_prompt into the API payload.
        judge_kwargs.pop('judge_prompt', None)
        if judge_name != 'judge_searchgen':
            raise ValueError(
                f"SearchGenBench only supports 'judge_searchgen'; got {judge_name!r}. "
                "Set MMDR_JUDGE_PROMPT=judge_searchgen or omit."
            )

        # Full-detail image input for the judge (framework default is 'low')
        judge_kwargs.setdefault('img_detail', 'high')
        model = build_judge(**judge_kwargs)
        if model is None or not model.working():
            logger.error(
                'SearchGenBench evaluation requires a working multimodal judge '
                '(vision + long output; upstream used an OpenAI-compatible endpoint). '
                'Please configure judge_kwargs properly.'
            )
            logger.error(DEBUG_MESSAGE)
            return pd.DataFrame()

        cache_dir = osp.join(pred_root, 'judge_searchgen_images')
        data['_judge_cache_dir'] = cache_dir
        data['_image_root'] = image_root

        grade_fn = _grade_item_judge_searchgen
        stem_base = '_searchgen_judge_searchgen'
        grade_stem = f'_grade{stem_base}'
        detail_stem = f'_detail{stem_base}'
        acc_stem = f'_acc{stem_base}'

        # Cache + resume support
        result_file = get_intermediate_file_path(eval_file, grade_stem, 'pkl')
        result_cache = {}
        if osp.exists(result_file):
            result_cache = load(result_file)
            # re-judge rows whose earlier attempt failed (judged=0)
            result_cache = {k: v for k, v in result_cache.items() if v.get('judged')}

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

        # Populate per-component columns (None -> NaN = absent, skip policy)
        for col in ['judged', 'scores_json', 'reasoning'] + COMPONENTS:
            default = 0 if col == 'judged' else ('' if col in ('scores_json', 'reasoning') else None)
            data[col] = [
                result_cache.get(str(idx), {}).get(col, default)
                for idx in data['index']
            ]
        data = data.drop(columns=['_judge_cache_dir', '_image_root'])

        # Detailed per-row dump
        detail_file = get_intermediate_file_path(eval_file, detail_stem)
        dump(data, detail_file)

        n_failed = int((pd.to_numeric(data['judged'], errors='coerce').fillna(0) == 0).sum())
        if n_failed:
            logger.warning(
                f'{n_failed}/{len(data)} samples have NO valid judge verdict after retries — '
                'they are EXCLUDED from aggregation (upstream skip policy). '
                'Re-run --mode eval to retry them.'
            )

        acc = self._compute_accuracy(data)
        score_file = get_intermediate_file_path(eval_file, acc_stem, 'csv')
        dump(acc, score_file)
        logger.info(f'SearchGenBench [{judge_name}] scores saved to {score_file}')

        return acc

    @staticmethod
    def _compute_accuracy(data) -> pd.DataFrame:
        """Upstream aggregation (skip policy) overall + per-group.

        ``overall_mean`` (0-3) = mean of present component means;
        ``overall_100`` = overall_mean * 100/3 (paper scale).
        """
        from .utils.judge_searchgen import COMPONENTS

        def _bloc(df: pd.DataFrame, name: str):
            ok = df[pd.to_numeric(df['judged'], errors='coerce').fillna(0) == 1]
            if len(ok) == 0:
                return {'group': name, 'n': len(df), 'n_judged': 0}
            row: Dict[str, Any] = {
                'group': name, 'n': len(df), 'n_judged': len(ok),
                'n_image_missing': int((ok['generated_image'] == '').sum()),
            }
            present_means = []
            for c in COMPONENTS:
                vals = pd.to_numeric(ok[c], errors='coerce').dropna()
                if len(vals):
                    m = float(vals.mean())
                    row[c] = m
                    present_means.append(m)
                else:
                    row[c] = None
            overall = sum(present_means) / len(present_means) if present_means else None
            row['overall_mean'] = overall
            row['overall_100'] = overall * 100.0 / 3.0 if overall is not None else None
            return row

        rows = [_bloc(data, 'Overall')]
        if 'language' in data.columns:
            for lang, sub in data.groupby('language'):
                rows.append(_bloc(sub, f'language:{lang}'))
        if 'category' in data.columns:
            for cat, sub in data.groupby('category'):
                rows.append(_bloc(sub, f'task_type:{cat}'))
        # Unofficial diagnostic split (NOT the paper's curated strata — those
        # are unreleased): samples with any judge-side slot vs none.
        has_slot = data.apply(
            lambda r: bool(json.loads(r.get('visual_reference_slots') or '[]'))
            or bool(json.loads(r.get('text_knowledge_slots') or '[]')), axis=1)
        rows.append(_bloc(data[has_slot], 'unofficial:has_ref_slots'))
        rows.append(_bloc(data[~has_slot], 'unofficial:no_ref_slots'))
        return pd.DataFrame(rows)
