# Adapted from VLMEvalKit (https://github.com/open-compass/VLMEvalKit), Apache License 2.0; modified by the OpenVLHarness authors.
import ast

from evaluation_kit import *
from ..smp import *
from .image_base import ImageBaseDataset
from .hf_release import HF_SPLITS, load_hf_split
from .utils import build_judge, DEBUG_MESSAGE
from .utils.judge_cache import guard_cache, require_judge
from .utils.hle_judge import HLE_JUDGE_PROMPT, parse_hle_judgement
from .utils.multiple_choice import report_acc, eval_vanilla, eval_circular_group
from .utils.shortqa import ShortQA_prompt
from ..utils import track_progress_rich
from ..smp.file import get_intermediate_file_path


def ShortQA_auxeval(model, line):
    def proc_str(s):
        chs = set(s)
        chs = [x for x in chs if x not in string.ascii_letters + ': ']
        for ch in chs:
            s = s.replace(ch, ' ')
        return s

    def extraction(resp):
        correct, reason = None, None
        correct_st, correct_ed = '[Begin Correctness]', '[End Correctness]'
        reason_st, reason_ed = '[Begin Reason]', '[End Reason]'
        if correct_st in resp and correct_ed in resp:
            correct = resp.split(correct_st)[1].split(correct_ed)[0].strip().lower()
            if ('yes' in correct) ^ ('no' in correct):
                correct = 1 if 'yes' in correct else 0
                if reason_st in resp and reason_ed in resp:
                    reason = resp.split(reason_st)[1].split(reason_ed)[0].strip()
                return correct, reason
            else:
                return None, None
        else:
            return None, None

    prompt = ShortQA_prompt(line)
    retry = 3
    for i in range(retry):
        output = model.generate(prompt, temperature=0.5 * i)
        ans = extraction(output)
        # print(output, ans)
        if ans[0] in [0, 1]:
            return dict(hit=ans[0], log=ans[1])

    return dict(hit=0, log='Fail to Judge')


def Comprehensive_auxeval(model, data):
    def valid(record, key_name):
        return key_name in record and (not pd.isna(record[key_name])) and record[key_name] != ''

    if isinstance(data, pd.DataFrame) and len(data) > 1:
        # Should Adopt CircularEval
        assert valid(data.iloc[0], 'A')
        data['GT'] = data['answer']
        return eval_circular_group(model, data)
    else:
        item = data.iloc[0] if isinstance(data, pd.DataFrame) else data
        if valid(item, 'A') and len(item['answer']) == 1:
            item['GT'] = item['answer']
            return eval_vanilla(model, item)
        else:
            return ShortQA_auxeval(model, item)


def _gt_with_alternatives(item):
    """Build a list `[primary_gt, *alternatives]` from a row.

    Handles three TSV conventions:
      - ``answer`` only (HLE, MMSearch, BCVL, ...): single string OR
        already a "[a, b, c]" JSON list string.
      - ``answer`` + ``alternative_gt_answers`` (FVQA): primary + JSON
        list of synonyms / variants. Both fields merged into one
        candidate list so Judge_RS / Judge_VDR / HLE can match any.
      - ``answer`` + ``answer_eval`` (Infoseek_GDM): same idea, separate
        column name. Merge if present.

    Returns a non-empty list of strings.
    """
    answer = item.get('answer', '')
    if isinstance(answer, str) and answer.strip().startswith('[') and answer.strip().endswith(']'):
        try:
            answer = ast.literal_eval(answer)
        except Exception:
            answer = [answer]
    if not isinstance(answer, list):
        answer = [answer]
    # FVQA-style alternatives column
    alts_raw = item.get('alternative_gt_answers', '') or item.get('answer_eval', '')
    if isinstance(alts_raw, str) and alts_raw.strip():
        try:
            alt_list = json.loads(alts_raw) if alts_raw.strip().startswith('[') else []
            if isinstance(alt_list, list) and alt_list:
                answer = list(answer) + [str(a) for a in alt_list if str(a).strip()]
        except Exception:
            pass
    # Deduplicate while preserving order
    seen = set()
    out = []
    for a in answer:
        s = str(a).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out or ['']


def HLE_style_auxeval(model, item):
    """HLE-style binary judge for short-answer datasets (default for hle).

    Mirrors ShortQA_auxeval's interface (returns ``{hit: 0|1, log: str}``)
    but uses the shared HLE judge prompt + JSON-extracting parser, so
    the resulting accuracy is directly comparable to WebWatcher / BCVL /
    other HLE-protocol scorers.

    For datasets with alternative answers (FVQA, Infoseek_GDM), uses
    ``_gt_with_alternatives`` to merge candidates; HLE judge only sees
    the primary GT but the secondary candidates are appended into the
    prompt's correct_answer field as ``primary (also accepted: x, y)``.
    """
    item = item.iloc[0] if isinstance(item, pd.DataFrame) else item
    question = str(item.get('question', ''))
    answer = _gt_with_alternatives(item)
    if len(answer) > 1:
        correct_answer = f"{answer[0]} (also accepted: {', '.join(answer[1:])})"
    else:
        correct_answer = answer[0]
    response = str(item.get('prediction', ''))

    prompt = HLE_JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )
    retry = 3
    last = None
    for _ in range(retry):
        out = model.generate(prompt)
        if 'Failed to obtain answer via API' in (out or ''):
            continue
        last = out
        parsed = parse_hle_judgement(out)
        if parsed.get('reasoning') != 'judge failed to parse':
            log = f"verdict={parsed['correct']} extracted={parsed.get('extracted_answer','')[:120]} reasoning={parsed.get('reasoning','')[:200]}"
            return dict(hit=int(parsed['correct']), log=log)
    return dict(hit=0, log=f'judge unparseable: {str(last)[:200]}' if last else 'judge unreachable')


def Judge_RS_auxeval(model, item):
    """Judge_RS variant of ``HLE_style_auxeval``.

    Sends ``JUDGE_RS_SYSTEM_PROMPT`` + ``JUDGE_RS_USER_PROMPT`` as
    separate roles via ``grade_with_judge_rs`` helper. Returns the
    same ``{hit, log}`` shape so the downstream aggregation is
    drop-in compatible. Opt-in via env ``MMDR_JUDGE_PROMPT=judge_rs``.
    """
    from .utils.judge_rs import grade_with_judge_rs

    item = item.iloc[0] if isinstance(item, pd.DataFrame) else item
    question = str(item.get('question', ''))
    answer = _gt_with_alternatives(item)
    response = str(item.get('prediction', ''))

    try:
        parsed = grade_with_judge_rs(model, question, answer, response)
    except Exception as e:
        return dict(hit=0, log=f'judge_rs error: {e}')

    log = f"verdict={parsed['correct']} reasoning={parsed.get('reasoning','')[:200]}"
    return dict(hit=int(parsed['correct']), log=log)


def Judge_VDR_auxeval(model, item):
    """Judge_VDR variant of ``HLE_style_auxeval`` (VDR paper contains-based
    judge, plain-text output). Returns the same ``{hit, log}`` shape so
    downstream aggregation is drop-in compatible. Opt-in via env
    ``MMDR_JUDGE_PROMPT=judge_vdr``.
    """
    from .utils.judge_vdr import grade_with_judge_vdr

    item = item.iloc[0] if isinstance(item, pd.DataFrame) else item
    question = str(item.get('question', ''))
    answer = _gt_with_alternatives(item)
    response = str(item.get('prediction', ''))

    try:
        parsed = grade_with_judge_vdr(model, question, answer, response)
    except Exception as e:
        return dict(hit=0, log=f'judge_vdr error: {e}')

    log = f"verdict={parsed['correct']} reasoning={parsed.get('reasoning','')[:200]}"
    return dict(hit=int(parsed['correct']), log=log)


_INFOSEEK_TEST_PARQUET_URL_DEFAULT = (
    "https://github.com/EvolvingLMMs-Lab/multimodal-search-r1/"
    "raw/main/mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet"
)


# ===== EncyclopedicVQA helpers =====
# EVQA = Mensink et al. 2023 "Encyclopedic VQA: Visual Questions About
# Detailed Properties of Fine-grained Categories". Images reference
# iNaturalist 2021 (~600GB) or Google Landmarks v2 (~500GB) by id;
# CSV (small) lists question+answer+image_id+dataset_name. Our default
# eval = random 2000 from test (5750 rows) with seed=42 — mirrors paper
# 2510.12801 §EVQA processing.
#
# General-purpose data root via env MMDR_EVQA_DATA_ROOT, defaulting to
# the LMUData-adjacent layout (osp.dirname(LMUData)/train/). Subdirs:
#   - EncyclopedicVQA/test.csv           (small CSV, ~2MB)
#   - iNat2021/train.tar.gz + train.json (224GB tar.gz + metadata)
#   - iNat2021/val.tar.gz + val.json     (8.4GB tar.gz + metadata)
#   - GoogleLandmarksV2/train/images_NNN.tar (500 tars, ~498GB)


def _gldv2_tar_num(image_id: str) -> int:
    """GLDv2 tar index from image_id: int(image_id[:3], 16) * 500 // 4096.

    Verified by sampling images_000/050/100/200/300/400/499.tar — the
    first-2-of-3 hex chars determine which of 500 tars contains the image.
    """
    return (int(image_id[:3], 16) * 500) // 4096


def _evqa_data_roots(lmu_root: str):
    """Return (evqa_csv_dir, inat_root, gldv2_train_dir).

    Defaults to {LMUData}/../train/{EncyclopedicVQA,iNat2021,GoogleLandmarksV2}.
    Override via env MMDR_EVQA_DATA_ROOT (root containing those subdirs).
    """
    default_root = osp.join(osp.dirname(lmu_root.rstrip('/')), 'train')
    root = os.environ.get('MMDR_EVQA_DATA_ROOT', default_root)
    return (
        osp.join(root, 'EncyclopedicVQA'),
        osp.join(root, 'iNat2021'),
        osp.join(root, 'GoogleLandmarksV2', 'train'),
    )


def _restore_literal_na_answers(df, source, cols=('answer',)):
    """Restore answer strings that pandas' default NA parsing turned into NaN.

    Some ground-truth answers are literally 'None', 'NA' or 'null' (e.g. three
    InfoSeek test questions), and ``pd.read_csv`` / ``pd.read_excel`` read them
    as NaN by default. ``source`` is either the dataset TSV path or a DataFrame
    that already holds the literal strings; rows are matched on ``index`` and
    only cells that are NaN in ``df`` and non-empty in ``source`` are changed.
    """
    import pandas as pd
    if not any(col in df and df[col].isna().any() for col in cols):
        return df
    if isinstance(source, str):
        if not source or not osp.exists(source):
            return df
        source = pd.read_csv(source, sep='\t', keep_default_na=False, dtype=str)
    if source is None or 'index' not in df or 'index' not in source:
        return df
    for col in cols:
        if col not in df or col not in source:
            continue
        valid = source[col].notna() & (source[col].astype(str) != '')
        lookup = dict(zip(source.loc[valid, 'index'].astype(str),
                          source.loc[valid, col].astype(str)))
        missing = df[col].isna()
        if not missing.any():
            continue
        restored = df.loc[missing, 'index'].astype(str).map(lookup)
        restored = restored[restored.notna()]
        if len(restored):
            df[col] = df[col].astype(object)
            df.loc[restored.index, col] = restored
    return df


_DEFAULT_SUBSET_SPLITS = ('FVQA_test', 'Infoseek_GDM_test', 'EncyclopedicVQA_test')
_DEFAULT_SUBSET_N = 300


def _write_test_mini(lmu_root: str, full_name: str, mini_name: str,
                     n: int = 300, seed: int = 42):
    """Materialise an N-row random subset TSV from an existing full-eval TSV.

    Idempotent: skip if mini TSV already exists. Image files are shared
    with the full variant (no copy — image_path column in TSV unchanged).

    Used for the FVQA_train_mini / EncyclopedicVQA_val_mini variants.
    """
    full_tsv = osp.join(lmu_root, f'{full_name}.tsv')
    mini_tsv = osp.join(lmu_root, f'{mini_name}.tsv')
    if osp.exists(mini_tsv):
        return mini_tsv
    if not osp.exists(full_tsv):
        raise FileNotFoundError(
            f'Cannot build {mini_name}: underlying {full_tsv} missing. '
            f'Trigger the full variant first.')
    import pandas as pd
    df = pd.read_csv(full_tsv, sep='\t', keep_default_na=False, dtype=str)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    df.to_csv(mini_tsv, sep='\t', index=False, quoting=__import__('csv').QUOTE_ALL)
    print(f'[mini-subset] wrote {mini_name}.tsv ({len(df)} rows, '
          f'random_state={seed} from {full_name})')
    return mini_tsv


def _ensure_evqa_subset(lmu_root: str, dataset_name: str,
                        split: str = 'test', n_samples=2000, seed: int = 42):
    """Materialise an EVQA split subset.

    split='test' n_samples=2000 → paper-aligned random 2K from 5750
        test rows (mirrors 2510.12801 §EVQA processing).
    split='val'  n_samples=None → all 2950 val rows (for experience
        learning; test stays untouched as eval set).

    Same image-resolution logic for both splits (iNat val.tar.gz + GLDv2
    images_NNN.tar). On first call, may take 10-30 min to scan tars;
    subsequent loads find TSV + images cached.
    """
    import os, csv, json, tarfile
    tsv_path = osp.join(lmu_root, f'{dataset_name}.tsv')
    if osp.exists(tsv_path):
        return tsv_path

    if split not in {'train', 'val', 'test'}:
        raise ValueError(f'EVQA split must be train/val/test, got {split!r}')

    evqa_dir, inat_root, gldv2_train = _evqa_data_roots(lmu_root)
    src_csv = osp.join(evqa_dir, f'{split}.csv')
    if not osp.exists(src_csv):
        raise FileNotFoundError(
            f'EVQA {split}.csv missing at {src_csv}. Set MMDR_EVQA_DATA_ROOT '
            f'or place EncyclopedicVQA/{split}.csv there.')

    import pandas as pd
    df = pd.read_csv(src_csv)
    if n_samples is not None and len(df) > n_samples:
        print(f'[evqa/auto-prep] loaded {len(df)} {split} rows; sampling {n_samples} with seed={seed}')
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    else:
        print(f'[evqa/auto-prep] loaded {len(df)} {split} rows; using all')
        df = df.reset_index(drop=True)
    id_prefix = f'evqa_{split}'

    # Split image lookups by dataset_name; resolve iNat first (1 tar scan),
    # then GLDv2 (grouped by tar number for minimum tar scans).
    inat_rows = df[df['dataset_name'] == 'inaturalist']
    gldv2_rows = df[df['dataset_name'] == 'landmarks']
    print(f'[evqa/auto-prep] iNat: {len(inat_rows)}  landmarks: {len(gldv2_rows)}')

    out_img_dir = osp.join(lmu_root, 'images', dataset_name)
    os.makedirs(out_img_dir, exist_ok=True)

    # iNat: id → file_name. EVQA references iNat *val* images (val.json
    # id range 2,686,843-2,786,842) — NOT train (0-2,686,842). Verified
    # empirically: every sampled EVQA iNat id resolves in val.json, none
    # in train.json. Val tar.gz is only 8.4GB so this is faster anyway.
    print(f'[evqa/auto-prep] loading iNat val.json metadata...')
    inat_id_to_path = {}
    val_json = osp.join(inat_root, 'val.json')
    val_json_gz = osp.join(inat_root, 'val.json.tar.gz')
    if not osp.exists(val_json) and osp.exists(val_json_gz):
        with tarfile.open(val_json_gz, 'r:gz') as tar:
            m = tar.getmember('val.json')
            tar.extract(m, inat_root)
    with open(val_json) as f:
        meta = json.load(f)
    for img in meta.get('images', []):
        inat_id_to_path[int(img['id'])] = img['file_name']
    print(f'[evqa/auto-prep] iNat val metadata loaded ({len(inat_id_to_path)} ids)')

    # iNat extract: open val.tar.gz once, sequential scan.
    # Map source file → LIST of dsts (multiple EVQA questions may
    # reference the same iNat image, so 1 file may fan out to N dsts).
    inat_file_to_dsts = {}   # inat_file_name -> [dst_path, ...]
    for _, rec in inat_rows.iterrows():
        first_id = str(rec['dataset_image_ids']).split(',')[0].split('|')[0].strip()
        try: iid = int(first_id)
        except Exception: continue
        if iid not in inat_id_to_path: continue
        fp_in_tar = inat_id_to_path[iid]   # "val/<dir>/<uuid>.jpg"
        data_id = f'{id_prefix}_{rec.name}'
        dst = osp.join(out_img_dir, f'{data_id}.jpg')
        if osp.exists(dst): continue
        inat_file_to_dsts.setdefault(fp_in_tar, []).append(dst)

    if inat_file_to_dsts:
        n_unique = len(inat_file_to_dsts)
        n_total_dsts = sum(len(v) for v in inat_file_to_dsts.values())
        print(f'[evqa/auto-prep] scanning iNat val.tar.gz for {n_unique} unique images '
              f'({n_total_dsts} dsts after fan-out, ~5-10 min — 8.4GB compressed)...')
        n_ex = 0
        with tarfile.open(osp.join(inat_root, 'val.tar.gz'), 'r:gz') as tar:
            for m in tar:
                if not m.isreg(): continue
                dsts = inat_file_to_dsts.get(m.name)
                if not dsts: continue
                src = tar.extractfile(m)
                if src:
                    blob = src.read()
                    for dst in dsts:
                        with open(dst, 'wb') as f:
                            f.write(blob)
                        n_ex += 1
                del inat_file_to_dsts[m.name]
                if not inat_file_to_dsts: break
        print(f'[evqa/auto-prep] iNat extracted: {n_ex} dsts / {n_total_dsts} requested')

    # GLDv2: group by tar number, scan each tar once.
    # Same fan-out fix: one source image_id → multiple dsts possible.
    print(f'[evqa/auto-prep] resolving GLDv2 tars...')
    gldv2_by_tar = {}   # tar_num -> {image_id: [dst_path, ...]}
    for _, rec in gldv2_rows.iterrows():
        first_id = str(rec['dataset_image_ids']).split(',')[0].split('|')[0].strip()
        if len(first_id) < 3: continue
        tar_num = _gldv2_tar_num(first_id)
        data_id = f'{id_prefix}_{rec.name}'
        dst = osp.join(out_img_dir, f'{data_id}.jpg')
        if osp.exists(dst): continue
        gldv2_by_tar.setdefault(tar_num, {}).setdefault(first_id, []).append(dst)

    if gldv2_by_tar:
        n_tars = len(gldv2_by_tar)
        n_imgs = sum(len(dsts) for tar_map in gldv2_by_tar.values() for dsts in tar_map.values())
        n_unique = sum(len(tar_map) for tar_map in gldv2_by_tar.values())
        print(f'[evqa/auto-prep] {n_imgs} GLDv2 dsts ({n_unique} unique images) across {n_tars} tars; scanning...')
        n_ex = 0
        for tar_num in sorted(gldv2_by_tar):
            tar_path = osp.join(gldv2_train, f'images_{tar_num:03d}.tar')
            if not osp.exists(tar_path):
                tot = sum(len(d) for d in gldv2_by_tar[tar_num].values())
                print(f'  [warn] missing {tar_path}; skipping {tot} images')
                continue
            wanted = dict(gldv2_by_tar[tar_num])   # image_id -> [dsts]
            with tarfile.open(tar_path, 'r') as tar:
                for m in tar:
                    if not m.isreg(): continue
                    stem = m.name.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                    if stem in wanted:
                        src = tar.extractfile(m)
                        if src:
                            blob = src.read()
                            for dst in wanted[stem]:
                                with open(dst, 'wb') as f:
                                    f.write(blob)
                                n_ex += 1
                        del wanted[stem]
                        if not wanted: break
        print(f'[evqa/auto-prep] GLDv2 extracted: {n_ex}/{n_imgs}')

    # Write TSV. Skip rows whose image isn't on disk.
    print(f'[evqa/auto-prep] writing TSV → {tsv_path}')
    cols = ['index', 'question', 'answer', 'alternative_gt_answers',
            'area', 'subfield', 'image_path']
    n_written = 0
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_ALL)
        w.writerow(cols)
        for idx, rec in df.iterrows():
            data_id = f'{id_prefix}_{idx}'
            img_path = osp.join(out_img_dir, f'{data_id}.jpg')
            if not osp.exists(img_path):
                continue
            # EVQA answer is pipe-separated; first is primary, rest are alternatives
            ans_str = str(rec.get('answer') or '').strip()
            parts = [p.strip() for p in ans_str.split('|') if p.strip()]
            primary = parts[0] if parts else ''
            alts = parts[1:] if len(parts) > 1 else []
            alts_json = json.dumps(alts, ensure_ascii=False)
            w.writerow([
                data_id,
                str(rec.get('question') or ''),
                primary,
                alts_json,
                str(rec.get('dataset_name') or ''),       # area: inaturalist|landmarks
                str(rec.get('question_type') or ''),     # subfield: automatic|templated|multi_answer|2_hop
                img_path,
            ])
            n_written += 1
    n_requested = n_samples if n_samples is not None else len(df)
    print(f'[evqa/auto-prep] wrote {n_written} rows (of {n_requested} {split}; the rest lost images)')
    return tsv_path


# Backward-compat alias — test subset = paper-aligned 2K.
def _ensure_evqa_test_subset(lmu_root: str, dataset_name: str = 'EncyclopedicVQA_test'):
    return _ensure_evqa_subset(lmu_root, dataset_name, split='test', n_samples=2000)


def _ensure_fvqa_subset(lmu_root: str, split: str = 'test'):
    """Idempotently materialise the FVQA train/test split from HuggingFace.

    Source: ``lmms-lab/FVQA`` (HuggingFace), same data as MMSearch-R1's
    fvqa (``data_source: mmsearch_r1/fvqa_{split}``).

    First call:
    - downloads the HF dataset (~MB-scale text + image bytes per row)
    - writes ``{lmu_root}/FVQA_{split}.tsv``
    - extracts each row's first image to ``{lmu_root}/images/FVQA_{split}/<data_id>.jpg``

    Idempotent: skip if TSV already exists.

    Schema matches the existing TSV layout used by the framework:
      index, question, answer, alternative_gt_answers, category, subfield, image_path

    Override HF repo via env ``MMDR_FVQA_HF_REPO`` (default ``lmms-lab/FVQA``).
    """
    import os, csv, json
    if split not in {'train', 'test'}:
        raise ValueError(f'FVQA split must be train/test, got {split!r}')
    tsv_path = osp.join(lmu_root, f'FVQA_{split}.tsv')
    if osp.exists(tsv_path):
        return tsv_path

    from datasets import load_dataset
    repo_id = os.environ.get('MMDR_FVQA_HF_REPO', 'lmms-lab/FVQA')
    print(f'[fvqa/auto-download] loading HF dataset {repo_id} split={split} (one-time)')
    ds = load_dataset(repo_id, split=split)
    print(f'[fvqa/auto-download] loaded {len(ds)} rows; writing TSV + extracting images...')

    img_dir = osp.join(lmu_root, 'images', f'FVQA_{split}')
    os.makedirs(img_dir, exist_ok=True)

    cols = ['index', 'question', 'answer', 'alternative_gt_answers',
            'category', 'subfield', 'image_path']
    n_rows = n_imgs = 0
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_ALL)
        w.writerow(cols)
        for rec in ds:
            data_id = str(rec.get('data_id') or '')
            # Question lives in prompt[0].content; ground truth in
            # reward_model.ground_truth; alternatives in candidate_answers.
            prompt = rec.get('prompt') or []
            question = ''
            if hasattr(prompt, '__len__') and len(prompt) > 0 and isinstance(prompt[0], dict):
                question = str(prompt[0].get('content', ''))
            rm = rec.get('reward_model') or {}
            if not isinstance(rm, dict):
                rm = {}
            gt = str(rm.get('ground_truth', ''))
            alts_raw = rm.get('candidate_answers', '[]')
            try:
                alts_list = json.loads(alts_raw) if isinstance(alts_raw, str) else list(alts_raw or [])
            except (json.JSONDecodeError, TypeError):
                alts_list = list(alts_raw) if alts_raw else []
            alts_json = json.dumps([str(a) for a in alts_list], ensure_ascii=False)
            # Image: first entry of images list; HF gives PIL.Image or
            # {bytes: ...}. Save as JPG.
            imgs = rec.get('images') or []
            img_bytes = None
            if hasattr(imgs, '__len__') and len(imgs) > 0:
                first_img = imgs[0]
                if isinstance(first_img, dict):
                    img_bytes = first_img.get('bytes')
                else:
                    # PIL.Image instance
                    from io import BytesIO
                    buf = BytesIO()
                    first_img.save(buf, format='JPEG')
                    img_bytes = buf.getvalue()
            if not img_bytes:
                continue
            img_path = osp.join(img_dir, f'{data_id}.jpg')
            if not osp.exists(img_path):
                with open(img_path, 'wb') as imf:
                    imf.write(img_bytes)
                n_imgs += 1
            # FVQA train has 'category' (search_required/search_free); test is None.
            category = str(rec.get('category') or '') if rec.get('category') else ''
            w.writerow([data_id, question, gt, alts_json, category, '', img_path])
            n_rows += 1
    print(f'[fvqa/auto-download] wrote TSV: {tsv_path} ({n_rows} rows, {n_imgs} new images)')
    return tsv_path


def _ensure_infoseek_test_subset(lmu_root: str, dataset_name: str = 'Infoseek_GDM_test'):
    """Idempotently materialise the paper-aligned 2K InfoSeek test subset.

    Downloads `mmsearch_r1_infoseek_sub_2k.parquet` from the multimodal-search-r1
    GitHub LFS (MMSearch-R1 paper §4.1) on first use, converts to:
      - {lmu_root}/{dataset_name}.tsv                              (2000 rows)
      - {lmu_root}/images/{dataset_name}/<data_id>.JPEG            (2000 images)

    Skip-load if TSV already exists. Override URL via env
    MMDR_INFOSEEK_TEST_PARQUET_URL.
    """
    import os, csv, json
    tsv_path = osp.join(lmu_root, f'{dataset_name}.tsv')
    if osp.exists(tsv_path):
        return tsv_path
    # Need pandas / urllib only on first-time path.
    import urllib.request
    import pandas as pd
    img_dir = osp.join(lmu_root, 'images', dataset_name)
    os.makedirs(img_dir, exist_ok=True)
    parquet_path = osp.join(lmu_root, 'images', f'{dataset_name}.parquet')
    url = os.environ.get('MMDR_INFOSEEK_TEST_PARQUET_URL', _INFOSEEK_TEST_PARQUET_URL_DEFAULT)
    if not osp.exists(parquet_path):
        print(f'[infoseek/auto-download] fetching paper 2K subset from {url}')
        print(f'[infoseek/auto-download] → {parquet_path} (~277 MB, one-time)')
        urllib.request.urlretrieve(url, parquet_path)
    df = pd.read_parquet(parquet_path)
    print(f'[infoseek/auto-download] loaded {len(df)} rows; writing TSV + extracting images...')
    cols = ['index', 'question', 'answer', 'alternative_gt_answers',
            'area', 'subfield', 'image_path']
    n_rows = n_imgs = 0
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_ALL)
        w.writerow(cols)
        for _, rec in df.iterrows():
            data_id = str(rec.get('data_id') or '')
            prompt = rec.get('prompt') or []
            question = ''
            if hasattr(prompt, '__len__') and len(prompt) > 0 and isinstance(prompt[0], dict):
                question = str(prompt[0].get('content', ''))
            rm = rec.get('reward_model') or {}
            if not isinstance(rm, dict):
                rm = {}
            gt = str(rm.get('ground_truth', ''))
            alts_raw = rm.get('candidate_answers', '[]')
            try:
                alts_list = json.loads(alts_raw) if isinstance(alts_raw, str) else list(alts_raw or [])
            except (json.JSONDecodeError, TypeError):
                alts_list = list(alts_raw) if alts_raw else []
            alts_json = json.dumps([str(a) for a in alts_list], ensure_ascii=False)
            imgs = rec.get('images') or []
            img_bytes = None
            if hasattr(imgs, '__len__') and len(imgs) > 0 and isinstance(imgs[0], dict):
                img_bytes = imgs[0].get('bytes')
            if not img_bytes:
                continue
            img_path = osp.join(img_dir, f'{data_id}.JPEG')
            if not osp.exists(img_path):
                with open(img_path, 'wb') as imf:
                    imf.write(img_bytes)
                n_imgs += 1
            image_id = str(rec.get('image_id') or '')
            area = f'shard_{image_id[5:7]}' if image_id.startswith('oven_') else ''
            w.writerow([data_id, question, gt, alts_json, area, '', img_path])
            n_rows += 1
    print(f'[infoseek/auto-download] wrote TSV: {tsv_path} ({n_rows} rows, {n_imgs} new images)')
    return tsv_path


class ImageShortQADataset(ImageBaseDataset):
    TYPE = 'Short'

    DATASET_URL = {
        'LiveMMBench_Infographic': '',
        'LiveMMBench_Perception': '',
        'LiveMMBench_Reasoning': '',
        'LiveMMBench_Reasoning_circular': '',
        'hle':'https://opencompass.openxlab.space/utils/VLMEval/hle.tsv',
        # No download URL: the TSV must already be in LMUData.
        'Infoseek_GDM_train': '',
        'Infoseek_GDM_val': '',
        # FVQA_test, Infoseek_GDM_test, EncyclopedicVQA_test and hle are loaded from the
        # OpenVLHarness eval repo (hf_release.HF_SPLITS). The builders and URLs described
        # for them below are kept only for the case that a split is removed from HF_SPLITS.
        #
        # Infoseek_GDM_test = paper-aligned 2K subset (MMSearch-R1 paper §4.1).
        # First use auto-downloads the parquet from GitHub LFS (see
        # _INFOSEEK_TEST_PARQUET_URL below), converts to TSV + extracts
        # 2000 images into LMUData/images/Infoseek_GDM_test/. The bare
        # InfoSeek test split has no public GT (hidden leaderboard);
        # this 2K subset is what the paper actually evaluates on.
        # Override URL via env MMDR_INFOSEEK_TEST_PARQUET_URL if needed.
        'Infoseek_GDM_test': '',
        # FVQA splits (== MMSearch-R1 fvqa). Auto-download from
        # HuggingFace ``lmms-lab/FVQA`` on first use; writes TSV +
        # extracts images. Override repo via env MMDR_FVQA_HF_REPO.
        'FVQA_train': '',
        'FVQA_test': '',
        # Encyclopedic-VQA test = random 2000 from full 5750 test (seed=42).
        # Mirrors paper 2510.12801 §EVQA processing. First use builds TSV +
        # extracts images from iNat2021 / GoogleLandmarksV2 tars under the
        # MMDR_EVQA_DATA_ROOT directory (default {LMUData}/../train/).
        'EncyclopedicVQA_test': '',
        # EVQA val = all 2950 val rows (no sampling). Inference + future
        # experience-learning use. Test (2000 subset) stays the eval split.
        # Same image-resolution path as EncyclopedicVQA_test.
        'EncyclopedicVQA_val': '',
        # *_train_mini / *_val_mini = 300 random sub (seed=42) of the
        # corresponding train/val TSV; image files shared with the full
        # variant (no extra disk).
        'FVQA_train_mini':           '',
        'EncyclopedicVQA_val_mini':  '',
        # Hand-curated Wikidata seed set for multimodal multi-hop validation.
        # No download URL: the TSV must already be in LMUData.
        'Gen_Seed_V0': '',
    }

    DATASET_MD5 = {
        'hle': 'a83cbdbea89f27c2aa5b8f34a8894b72',
    }

    def load_data(self, dataset):
        # Hosted splits come straight from the eval repo. The builders below
        # (GitHub LFS parquet for Infoseek_GDM_test, lmms-lab/FVQA, the EVQA
        # tars) are used by FVQA_train / EncyclopedicVQA_val and would only be
        # used for the test splits if those were removed from HF_SPLITS.
        hosted = dataset in HF_SPLITS
        if dataset == 'Infoseek_GDM_test' and not hosted:
            from ..smp.file import LMUDataRoot
            _ensure_infoseek_test_subset(LMUDataRoot(), dataset_name='Infoseek_GDM_test')
        # EncyclopedicVQA test: paper-aligned 2K subset (seed=42 sample
        # of 5750 test rows). First-time builds TSV + extracts images
        # from iNat2021 / GoogleLandmarksV2 tars (MMDR_EVQA_DATA_ROOT).
        if dataset == 'EncyclopedicVQA_test' and not hosted:
            from ..smp.file import LMUDataRoot
            _ensure_evqa_test_subset(LMUDataRoot(), dataset_name='EncyclopedicVQA_test')
        # EncyclopedicVQA val: all 2950 rows (no sampling).
        if dataset == 'EncyclopedicVQA_val':
            from ..smp.file import LMUDataRoot
            _ensure_evqa_subset(LMUDataRoot(), 'EncyclopedicVQA_val',
                                split='val', n_samples=None)

        # FVQA train + test: HuggingFace ``lmms-lab/FVQA`` auto-download
        # (same data as MMSearch-R1 fvqa_{split}). First use grabs the
        # HF dataset, writes TSV + extracts images. ~MB-scale, fast.
        if dataset in ('FVQA_test', 'FVQA_train') and not hosted:
            from ..smp.file import LMUDataRoot
            split = 'test' if dataset == 'FVQA_test' else 'train'
            _ensure_fvqa_subset(LMUDataRoot(), split=split)

        # train / val mini variants (300 random seed=42 of full).
        # Ensure full TSV exists, then materialise mini TSV that shares images.
        if dataset == 'FVQA_train_mini':
            from ..smp.file import LMUDataRoot
            _write_test_mini(LMUDataRoot(), 'FVQA_train', 'FVQA_train_mini')
        if dataset == 'EncyclopedicVQA_val_mini':
            from ..smp.file import LMUDataRoot
            _ensure_evqa_subset(LMUDataRoot(), 'EncyclopedicVQA_val',
                                split='val', n_samples=None)
            _write_test_mini(LMUDataRoot(), 'EncyclopedicVQA_val', 'EncyclopedicVQA_val_mini')

        df = load_hf_split(self, dataset) if hosted else super().load_data(dataset)
        df = _restore_literal_na_answers(df, getattr(self, 'data_path', ''))

        # FVQA_train-only: optional category subset filter.
        # FVQA train rows carry a ``category`` column with values
        #   'search_required' (~68%) — questions that need search to answer
        #   'search_free'     (~32%) — answerable from image+knowledge alone
        # Set FVQA_CATEGORY={search_required,search_free,all} to narrow.
        # 'all' or unset → no filter. FVQA_test has no category labels
        # (all 'None'), so this filter is a no-op there.
        if dataset == 'FVQA_train':
            cat = os.environ.get('FVQA_CATEGORY', '').strip().lower()
            if cat and cat != 'all':
                if cat not in ('search_required', 'search_free'):
                    raise ValueError(
                        f"FVQA_CATEGORY must be 'search_required', 'search_free', "
                        f"or 'all' (unset); got {cat!r}"
                    )
                if 'category' not in df.columns:
                    raise ValueError(
                        "FVQA_train TSV missing 'category' column; "
                        "delete FVQA_train.tsv from LMUData to rebuild it from lmms-lab/FVQA"
                    )
                before = len(df)
                df = df[df['category'].astype(str).str.lower() == cat].reset_index(drop=True)
                print(f"[FVQA_train] FVQA_CATEGORY={cat}: filtered {before} → {len(df)} rows")

        # FVQA_test / Infoseek_GDM_test / EncyclopedicVQA_test evaluate on a
        # fixed 300-row subset (random_state=42) by default; set
        # <DATASET>_ALL=true (FVQA_TEST_ALL, INFOSEEK_GDM_TEST_ALL,
        # ENCYCLOPEDICVQA_TEST_ALL) for the full split.
        if dataset in _DEFAULT_SUBSET_SPLITS:
            env = f'{dataset.upper()}_ALL'
            raw = os.environ.get(env, '').strip().lower()
            if raw not in ('', 'true', 'false'):
                raise ValueError(f"{env} must be 'true' or 'false' (unset = false); got {raw!r}")
            if raw == 'true':
                print(f"[{dataset}] {env}=true: keeping full {len(df)} rows")
            elif len(df) > _DEFAULT_SUBSET_N:
                before = len(df)
                df = df.sample(n=_DEFAULT_SUBSET_N, random_state=42).reset_index(drop=True)
                print(f"[{dataset}] default subset: {before} → {len(df)} rows (seed=42); {env}=true for all")
        return df

    # It returns a DataFrame
    # Needs an LLM judge (there is no exact-matching scorer); checked before inference.
    REQUIRES_LLM_JUDGE = 'strict'

    def evaluate(self, eval_file, **judge_kwargs):
        require_judge(self.dataset_name, judge_kwargs, allow_exact_matching=False)
        data = load(eval_file)
        _ = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data = _restore_literal_na_answers(data, getattr(self, 'data', None))
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # Pop MMSearch-specific keys that would otherwise be forwarded to
        # OpenAIWrapper.generate_inner and into the request body, where OpenAI
        # returns 400 on unrecognized fields. Same bug as browsecomp_vl.py
        # and livevqa.py had; fix mirrors those.
        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 4)
        judge_kwargs.pop('enable_llm_judge', None)
        # ``judge_max_tokens`` is consumed centrally in build_judge() (yaml is
        # the single source of truth; no per-evaluator default).

        # Per-dataset default judge (when neither env var nor judge_kwargs
        # specifies). Datasets in HLE_FAMILY_DATASETS support the
        # {hle, judge_rs, judge_vdr} switch; everything else falls back
        # to Comprehensive_auxeval (ShortQA prompt) regardless.
        _DATASET_DEFAULT_JUDGE = {
            'hle': 'hle',
            'FVQA_train': 'judge_rs',  # FVQA uses judge_rs (MMSearch-R1 paper protocol) by default
            'FVQA_test':  'judge_rs',
            # InfoSeek_GDM: short-answer factoid (entity names, place names, numbers etc.)
            # — same family as FVQA. Default to judge_rs (semantic short-answer match)
            # since strict contains-based judge_vdr would miss many paraphrases.
            'Infoseek_GDM_val':            'judge_rs',
            'Infoseek_GDM_test':           'judge_rs',
            'EncyclopedicVQA_test':        'judge_rs',
            'EncyclopedicVQA_val':         'judge_rs',
            # 300-sample mini variants inherit underlying default
            'FVQA_train_mini':             'judge_rs',
            'EncyclopedicVQA_val_mini':    'judge_rs',
        }
        _HLE_FAMILY_DATASETS = {'hle', 'FVQA_train', 'FVQA_test',
                                'Infoseek_GDM_val',
                                'Infoseek_GDM_test',
                                'EncyclopedicVQA_test', 'EncyclopedicVQA_val',
                                'FVQA_train_mini',
                                'EncyclopedicVQA_val_mini'}
        _default_judge = _DATASET_DEFAULT_JUDGE.get(self.dataset_name, 'hle')

        # Judge selection: env var > judge_kwargs > per-dataset default
        judge_name = (
            os.environ.get('MMDR_JUDGE_PROMPT')
            or judge_kwargs.pop('judge_prompt', None)
            or _default_judge
        ).strip().lower()
        if judge_name not in ('hle', 'judge_rs', 'judge_vdr'):
            raise ValueError(
                f"Unknown judge_prompt {judge_name!r}; expected 'hle', 'judge_rs', or 'judge_vdr'"
            )
        judge_kwargs.pop('judge_prompt', None)

        if judge_name == 'judge_rs':
            storage_stem, tmp_stem, acc_stem = '_judge_judge_rs', '_tmp_judge_rs', '_acc_judge_rs'
        elif judge_name == 'judge_vdr':
            storage_stem, tmp_stem, acc_stem = '_judge_judge_vdr', '_tmp_judge_vdr', '_acc_judge_vdr'
        else:
            storage_stem, tmp_stem, acc_stem = '_judge', '_tmp', '_acc'
        storage = get_intermediate_file_path(eval_file, storage_stem)
        tmp_file = get_intermediate_file_path(eval_file, tmp_stem, 'pkl')
        judge_model = judge_kwargs.get('model', 'gpt-4o-mini')
        guard_cache(storage, data, judge_model, prompt=judge_name, keyed=False)
        # With g_index the tmp file is keyed by question group, not by row.
        guard_cache(tmp_file, data, judge_model, prompt=judge_name, keyed='g_index' not in data)

        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            model_name = judge_kwargs.pop('model', 'gpt-4o-mini')
            model = build_judge(model=model_name, **judge_kwargs)
            if not model.working():
                raise RuntimeError(
                    f'The LLM judge {model_name} at {judge_kwargs.get("api_base") or "the default endpoint"} '
                    f'did not answer a test request; {self.dataset_name} needs a working judge. '
                    'Check the judge key and api_base.'
                )

            if model is not None:
                if 'g_index' not in data:
                    lines = [data.iloc[i] for i in range(len(data))]
                    indices = [x['index'] for x in lines if x['index'] not in ans_map]
                    lines = [x for x in lines if x['index'] not in ans_map]
                    tups = [(model, line) for line in lines]
                else:
                    main_data = data[[x == y for x, y in zip(data['index'], data['g_index'])]]
                    lines = [data[data['g_index'] == x] for x in main_data['index']]
                    indices = [x.iloc[0]['g_index'] for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    lines = [x for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    tups = [(model, x) for x in lines]
                    data = main_data

                if len(lines):
                    # For datasets in _HLE_FAMILY_DATASETS (hle, FVQA_train,
                    # FVQA_test): support the 3-way judge switch
                    # {hle, judge_rs, judge_vdr}. Per-dataset default lives
                    # in _DATASET_DEFAULT_JUDGE above (FVQA → judge_rs,
                    # hle → hle). Env MMDR_JUDGE_PROMPT or
                    # judge_kwargs['judge_prompt'] overrides per call.
                    #
                    # All other ImageShortQA datasets (LiveMMBench_*,
                    # Infoseek_GDM_train, Gen_Seed_V0) keep the
                    # Comprehensive_auxeval path (ShortQA prompt with EN/CN
                    # ICE examples; MCQ vs ShortQA dispatch). To extend the
                    # 3-way switch to those, add them to
                    # _HLE_FAMILY_DATASETS + _DATASET_DEFAULT_JUDGE above.
                    if self.dataset_name in _HLE_FAMILY_DATASETS:
                        if judge_name == 'judge_rs':
                            auxeval_fn = Judge_RS_auxeval
                        elif judge_name == 'judge_vdr':
                            auxeval_fn = Judge_VDR_auxeval
                        else:
                            auxeval_fn = HLE_style_auxeval
                    else:
                        auxeval_fn = Comprehensive_auxeval
                    res = track_progress_rich(
                        auxeval_fn, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            judge_results = [ans_map[x] for x in data['index']]
            data['hit'] = [x['hit'] for x in judge_results]
            data['log'] = [x['log'] for x in judge_results]
            dump(data, storage)

        data = load(storage)
        acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, acc_stem, 'csv')
        dump(acc, score_file)
        return acc
