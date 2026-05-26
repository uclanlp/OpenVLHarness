from mini_vlm_toolkit import *
from ..smp import *
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.hle_judge import HLE_JUDGE_PROMPT, parse_hle_judgement
from .utils.multiple_choice import report_acc, eval_vanilla, eval_circular_group
from .utils.shortqa import ShortQA_prompt
from .utils.webwatcher_subset import apply_webwatcher_subset
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
            answer = eval(answer)
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
    "https://github.com/tool-vision/multimodal-search-r1/"
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


def _write_test_mini(lmu_root: str, full_name: str, mini_name: str,
                     n: int = 300, seed: int = 42):
    """Materialise an N-row random subset TSV from an existing full-eval TSV.

    Idempotent: skip if mini TSV already exists. Image files are shared
    with the full variant (no copy — image_path column in TSV unchanged).

    Used for *_test_mini variants of FVQA / Infoseek_GDM / EncyclopedicVQA,
    so expensive-backbone runs (gpt-5 etc.) can hit a fixed-N cheap subset
    while the full variant stays available for cheaper Qwen runs.
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
    df = pd.read_csv(full_tsv, sep='\t')
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
    import os, csv, json, tarfile, random
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
        except: continue
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


_INFOSEEK_OVEN_INDEX_CACHE = '_oven_to_shard.pkl'


def _load_or_build_infoseek_oven_index(images_root: str) -> dict:
    """Return {oven_id: shard_2digit_str} for all images in shard01-08 tars.

    First call scans all 8 tars (~3 min, 5.8M files) and caches the result
    as {images_root}/_oven_to_shard.pkl (~120 MB). Subsequent calls load
    the pickle (~2-3s). The 2-digit prefix in `oven_XX...` is NOT a
    reliable shard locator — many ids live in shards other than the one
    their prefix would suggest, and ~36% of "shard00" prefix ids actually
    live in shards 04..08. Always use this index, never the prefix.
    """
    import pickle, tarfile, time
    cache = osp.join(images_root, _INFOSEEK_OVEN_INDEX_CACHE)
    if osp.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    print(f'[infoseek/oven-index] building from 8 shard tars (~3 min one-time)...')
    t0 = time.time()
    oven2shard = {}
    for s in ['01', '02', '03', '04', '05', '06', '07', '08']:
        p = osp.join(images_root, f'shard{s}.tar')
        if not osp.exists(p):
            continue
        with tarfile.open(p, 'r') as tar:
            for m in tar:
                if not m.isreg():
                    continue
                stem = m.name.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                if stem.startswith('oven_'):
                    oven2shard[stem] = s
    with open(cache, 'wb') as f:
        pickle.dump(oven2shard, f, protocol=4)
    print(f'[infoseek/oven-index] indexed {len(oven2shard):,} oven_ids in {time.time()-t0:.0f}s; '
          f'cached to {cache}')
    return oven2shard


def _ensure_infoseek_val_subset(lmu_root: str, n_samples: int = 3000, seed: int = 42,
                                dataset_name: str = 'Infoseek_GDM_val_3k'):
    """Random-N InfoSeek val subset for inference + experience learning.

    Source: /local3/.../InfoSeek_GDM/annotations/infoseek_val.jsonl (73,620 rows).
    Random sample N (seed=42), look up each oven_id in the cross-shard
    index, extract from the right shard tar.
    """
    import os, csv, json, random
    from pathlib import Path
    tsv_path = osp.join(lmu_root, f'{dataset_name}.tsv')
    if osp.exists(tsv_path):
        return tsv_path

    # Default infoseek_root next to the EVQA / iNat tree.
    default_root = osp.join(osp.dirname(lmu_root.rstrip('/')), 'train', 'InfoSeek_GDM')
    infoseek_root = os.environ.get('MMDR_INFOSEEK_GDM_ROOT', default_root)
    jsonl_path = osp.join(infoseek_root, 'annotations', 'infoseek_val.jsonl')
    if not osp.exists(jsonl_path):
        raise FileNotFoundError(
            f'InfoSeek val annotations missing at {jsonl_path}. '
            f'Set MMDR_INFOSEEK_GDM_ROOT to the InfoSeek_GDM root containing '
            f'annotations/ and images/.')

    from sft.data.download_train_datasets import (
        _infoseek_extract_from_shard, _infoseek_join_answers,
    )

    images_root = osp.join(infoseek_root, 'images')
    oven2shard = _load_or_build_infoseek_oven_index(images_root)

    print(f'[infoseek_val/auto-prep] loading {jsonl_path}')
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
    # Filter out any record whose image_id has no local shard (should be 0
    # for val on this server but stays defensive).
    n_raw = len(records)
    records = [r for r in records if r.get('image_id', '') in oven2shard]
    print(f'[infoseek_val/auto-prep] total val: {n_raw:,}  with local image: '
          f'{len(records):,}  sampling {n_samples} (seed={seed})')
    random.Random(seed).shuffle(records)
    selected = records[:n_samples]

    img_dir = osp.join(lmu_root, 'images', dataset_name)
    os.makedirs(img_dir, exist_ok=True)

    # Group missing-on-disk images by REAL shard (from index, not prefix).
    from collections import defaultdict
    cached = set()
    missing_by_shard = defaultdict(set)
    for rec in selected:
        oven = rec.get('image_id', '')
        if not oven.startswith('oven_'):
            continue
        if osp.exists(osp.join(img_dir, f'{oven}.JPEG')):
            cached.add(oven)
        else:
            shard = oven2shard.get(oven)
            if shard is not None:
                missing_by_shard[shard].add(oven)

    print(f'[infoseek_val/auto-prep] cached: {len(cached)}  to extract: '
          f'{sum(len(s) for s in missing_by_shard.values())}')

    for shard in sorted(missing_by_shard):
        wanted = missing_by_shard[shard]
        tar_path = Path(images_root) / f'shard{shard}.tar'
        if not tar_path.is_file():
            print(f'[infoseek_val/auto-prep] shard {shard}: no local tar; '
                  f'skipping {len(wanted)} ids')
            continue
        print(f'[infoseek_val/auto-prep] shard {shard}: extracting {len(wanted)} from {tar_path.name}')
        n_ex, n_miss = _infoseek_extract_from_shard(tar_path, wanted, Path(img_dir))
        print(f'  → extracted {n_ex}, still missing {n_miss}')

    # Write TSV (mirror Infoseek_GDM_test column layout).
    cols = ['index', 'question', 'answer', 'alternative_gt_answers',
            'area', 'subfield', 'image_path']
    n_written = 0
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_ALL)
        w.writerow(cols)
        for rec in selected:
            oven = rec.get('image_id', '')
            data_id = str(rec.get('data_id') or oven)
            img_path = osp.join(img_dir, f'{oven}.JPEG')
            if not osp.exists(img_path):
                continue
            # answer_eval is the official evaluation field but for numerical
            # questions it carries dict-with-range, not a plain string. Use
            # answer_eval only when its entries are plain strings; otherwise
            # fall back to the canonical `answer` field (plain string list).
            ans_eval = rec.get('answer_eval') or []
            if ans_eval and all(isinstance(a, str) for a in ans_eval):
                ans_field = ans_eval
            else:
                ans_field = rec.get('answer') or []
            primary, alts = _infoseek_join_answers(ans_field)
            area = f'shard_{oven2shard.get(oven, "?")}' if oven.startswith('oven_') else ''
            w.writerow([
                data_id,
                str(rec.get('question', '')),
                primary,
                alts,
                area,
                str(rec.get('question_type', '')),
                img_path,
            ])
            n_written += 1
    print(f'[infoseek_val/auto-prep] wrote {n_written} rows '
          f'(of {len(selected)} sampled; rest lost images)')
    return tsv_path


class ImageShortQADataset(ImageBaseDataset):
    TYPE = 'Short'

    DATASET_URL = {
        'LiveMMBench_Infographic': '',
        'LiveMMBench_Perception': '',
        'LiveMMBench_Reasoning': '',
        'LiveMMBench_Reasoning_circular': '',
        'hle':'https://opencompass.openxlab.space/utils/VLMEval/hle.tsv',
        # SFT distillation training source — TSV produced by
        # `sft/data/download_train_datasets.py --action to_tsv --dataset infoseek --split {train|val|test}`.
        # Empty URL = no auto-download; local TSV is authoritative.
        'Infoseek_GDM_train': '',
        'Infoseek_GDM_val': '',
        # Infoseek_GDM_test = paper-aligned 2K subset (MMSearch-R1 paper §4.1).
        # First use auto-downloads the parquet from GitHub LFS (see
        # _INFOSEEK_TEST_PARQUET_URL below), converts to TSV + extracts
        # 2000 images into LMUData/images/Infoseek_GDM_test/. The bare
        # InfoSeek test split has no public GT (hidden leaderboard);
        # this 2K subset is what the paper actually evaluates on.
        # Override URL via env MMDR_INFOSEEK_TEST_PARQUET_URL if needed.
        'Infoseek_GDM_test': '',
        # FVQA splits (== MMSearch-R1 fvqa). TSV produced by
        # `sft/data/download_train_datasets.py --action to_tsv --dataset fvqa --split {train|test}`.
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
        # InfoSeek val (random 3000 seed=42 from 73,620 val rows).
        # Inference + future experience-learning. Test (paper 2K) stays eval.
        # Source: $MMDR_INFOSEEK_GDM_ROOT/annotations/infoseek_val.jsonl
        # (default {LMUData}/../train/InfoSeek_GDM/).
        'Infoseek_GDM_val_3k': '',
        # *_test_mini = 300 random sub of the full eval set (seed=42).
        # Use for expensive backbones (gpt-5 etc.) where running full
        # 1800-2000 samples is cost-prohibitive. Same seed → consistent
        # 300 across runs. Subset materialised at first load_data; image
        # files shared with the full variant (no extra disk).
        'FVQA_test_mini':            '',
        'Infoseek_GDM_test_mini':    '',
        'EncyclopedicVQA_test_mini': '',
        # *_train_mini / *_val_mini = 300 random sub (seed=42) of the
        # corresponding train/val TSV. Same _write_test_mini helper as
        # the test_mini variants; image files shared with the full
        # variant (no extra disk).
        'FVQA_train_mini':           '',
        'Infoseek_GDM_val_3k_mini':  '',
        'EncyclopedicVQA_val_mini':  '',
        # Phase 0 hand-curated Wikidata seed for multimodal multi-hop validation.
        # TSV produced by /tmp/build_gen_seed_v0.py from /tmp/sparql_*.json + keepers.json.
        'Gen_Seed_V0': '',
    }

    DATASET_MD5 = {
        'hle': 'a83cbdbea89f27c2aa5b8f34a8894b72',
    }

    def load_data(self, dataset):
        # Special-case: Infoseek_GDM_test is the paper-aligned 2K subset
        # (MMSearch-R1 paper §4.1). On first use, fetch from public GitHub
        # LFS and convert parquet → TSV + images. After this, parent
        # load_data finds the TSV at the expected LMUData path.
        if dataset == 'Infoseek_GDM_test':
            from ..smp.file import LMUDataRoot
            _ensure_infoseek_test_subset(LMUDataRoot(), dataset_name='Infoseek_GDM_test')
        # EncyclopedicVQA test: paper-aligned 2K subset (seed=42 sample
        # of 5750 test rows). First-time builds TSV + extracts images
        # from iNat2021 / GoogleLandmarksV2 tars (MMDR_EVQA_DATA_ROOT).
        if dataset == 'EncyclopedicVQA_test':
            from ..smp.file import LMUDataRoot
            _ensure_evqa_test_subset(LMUDataRoot(), dataset_name='EncyclopedicVQA_test')
        # EncyclopedicVQA val: all 2950 rows (no sampling).
        if dataset == 'EncyclopedicVQA_val':
            from ..smp.file import LMUDataRoot
            _ensure_evqa_subset(LMUDataRoot(), 'EncyclopedicVQA_val',
                                split='val', n_samples=None)
        # InfoSeek val random 3000 (seed=42).
        if dataset == 'Infoseek_GDM_val_3k':
            from ..smp.file import LMUDataRoot
            _ensure_infoseek_val_subset(LMUDataRoot(),
                                        n_samples=3000, seed=42,
                                        dataset_name='Infoseek_GDM_val_3k')

        # *_test_mini = 300 random sub of underlying full TSV (seed=42).
        # Ensure underlying full exists, then materialise the mini TSV.
        if dataset == 'Infoseek_GDM_test_mini':
            from ..smp.file import LMUDataRoot
            _ensure_infoseek_test_subset(LMUDataRoot(), 'Infoseek_GDM_test')
            _write_test_mini(LMUDataRoot(), 'Infoseek_GDM_test', 'Infoseek_GDM_test_mini')
        if dataset == 'EncyclopedicVQA_test_mini':
            from ..smp.file import LMUDataRoot
            _ensure_evqa_test_subset(LMUDataRoot(), 'EncyclopedicVQA_test')
            _write_test_mini(LMUDataRoot(), 'EncyclopedicVQA_test', 'EncyclopedicVQA_test_mini')
        if dataset == 'FVQA_test_mini':
            from ..smp.file import LMUDataRoot
            _write_test_mini(LMUDataRoot(), 'FVQA_test', 'FVQA_test_mini')

        # train / val mini variants (300 random seed=42 of full).
        # Ensure full TSV exists, then materialise mini TSV that shares images.
        if dataset == 'FVQA_train_mini':
            from ..smp.file import LMUDataRoot
            _write_test_mini(LMUDataRoot(), 'FVQA_train', 'FVQA_train_mini')
        if dataset == 'Infoseek_GDM_val_3k_mini':
            from ..smp.file import LMUDataRoot
            _ensure_infoseek_val_subset(LMUDataRoot(),
                                        n_samples=3000, seed=42,
                                        dataset_name='Infoseek_GDM_val_3k')
            _write_test_mini(LMUDataRoot(), 'Infoseek_GDM_val_3k', 'Infoseek_GDM_val_3k_mini')
        if dataset == 'EncyclopedicVQA_val_mini':
            from ..smp.file import LMUDataRoot
            _ensure_evqa_subset(LMUDataRoot(), 'EncyclopedicVQA_val',
                                split='val', n_samples=None)
            _write_test_mini(LMUDataRoot(), 'EncyclopedicVQA_val', 'EncyclopedicVQA_val_mini')

        # Default ImageBaseDataset loader; then auto-filter to the
        # WebWatcher subset where applicable (currently only ``hle``).
        # No-op for datasets without a subset entry.
        # Set MVT_DISABLE_WEBWATCHER_SUBSET=1 to bypass.
        df = super().load_data(dataset)
        df = apply_webwatcher_subset(df, dataset)

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
                        "regenerate via sft/data/download_train_datasets.py --action to_tsv --dataset fvqa"
                    )
                before = len(df)
                df = df[df['category'].astype(str).str.lower() == cat].reset_index(drop=True)
                print(f"[FVQA_train] FVQA_CATEGORY={cat}: filtered {before} → {len(df)} rows")

        # FVQA_test default subsample: 1800 → 300 random rows (seed=42).
        # Full 1800 is too expensive for routine baselines; 300 matches the
        # WebWatcher-style budget used for SimpleVQA_EN / LIVEVQA_v2 so the
        # numbers are directly comparable.
        #   FVQA_TEST_ALL=false (default, unset) → 300 subset, seed=42
        #   FVQA_TEST_ALL=true                  → full 1800
        if dataset == 'FVQA_test':
            raw = os.environ.get('FVQA_TEST_ALL', '').strip().lower()
            if raw not in ('', 'true', 'false'):
                raise ValueError(
                    f"FVQA_TEST_ALL must be 'true' or 'false' (unset = false); got {raw!r}"
                )
            test_all = (raw == 'true')
            if not test_all and len(df) > 300:
                before = len(df)
                df = df.sample(n=300, random_state=42).reset_index(drop=True)
                print(f"[FVQA_test] FVQA_TEST_ALL=false (default): subsampled {before} → {len(df)} rows (seed=42)")
            elif test_all:
                print(f"[FVQA_test] FVQA_TEST_ALL=true: keeping full {len(df)} rows")
        return df

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        _ = self.dataset_name
        # Subset-aware re-eval: if the dataset has a WebWatcher subset
        # entry, filter the predictions to ONLY those indices before
        # judging. Without this, `--mode eval` on a full prediction xlsx
        # would re-judge every row (waste). Applies to ``hle``; no-op for
        # datasets without a subset entry.
        data = apply_webwatcher_subset(data, self.dataset_name)
        assert 'answer' in data and 'prediction' in data
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
            'Infoseek_GDM_val_3k':         'judge_rs',
            'Infoseek_GDM_test':           'judge_rs',
            'EncyclopedicVQA_test':        'judge_rs',
            'EncyclopedicVQA_val':         'judge_rs',
            # 300-sample mini variants inherit underlying default
            'FVQA_test_mini':              'judge_rs',
            'Infoseek_GDM_test_mini':      'judge_rs',
            'EncyclopedicVQA_test_mini':   'judge_rs',
            'FVQA_train_mini':             'judge_rs',
            'Infoseek_GDM_val_3k_mini':    'judge_rs',
            'EncyclopedicVQA_val_mini':    'judge_rs',
        }
        _HLE_FAMILY_DATASETS = {'hle', 'FVQA_train', 'FVQA_test',
                                'Infoseek_GDM_val', 'Infoseek_GDM_val_3k',
                                'Infoseek_GDM_test',
                                'EncyclopedicVQA_test', 'EncyclopedicVQA_val',
                                'FVQA_test_mini', 'Infoseek_GDM_test_mini',
                                'EncyclopedicVQA_test_mini',
                                'FVQA_train_mini', 'Infoseek_GDM_val_3k_mini',
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

        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            model = judge_kwargs.pop('model', 'gpt-4o-mini')
            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(model=model, **judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None

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
                    # PathVQA, Infoseek_GDM_*) keep the legacy
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


class PathVQA_VAL(ImageShortQADataset):
    DATASET_URL = {
        'PathVQA_VAL': 'https://huggingface.co/datasets/Pfei111/PathVQA/resolve/main/PathVQA_VAL.tsv',
    }

    DATASET_MD5 = {
        'PathVQA_VAL': None,
    }


class PathVQA_TEST(ImageShortQADataset):
    DATASET_URL = {
        'PathVQA_TEST': 'https://huggingface.co/datasets/Pfei111/PathVQA/resolve/main/PathVQA_TEST.tsv',
    }

    DATASET_MD5 = {
        'PathVQA_TEST': None,
    }
