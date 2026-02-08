from huggingface_hub import snapshot_download, hf_hub_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from glob import glob
import os
import os.path as osp
import re
import json
import shutil
import zipfile
import pandas as pd

FAIL_MSG = 'Failed to obtain answer via API.'

# ── HuggingFace repos ────────────────────────────────────────────────
META_REPO = 'zai-org/LVBench'        # video_info.meta.jsonl + QA
VIDEO_REPO = 'AIWinter/LVBench'      # all_videos_split.zip.001 … .014
NUM_ZIP_PARTS = 14
# ─────────────────────────────────────────────────────────────────────


class LVBench(VideoBaseDataset):

    TYPE = 'Video-MCQ'
    DATASET_NAME = 'LVBench'

    CATEGORIES = {
        "key information retrieval": "KIR",
        "event understanding": "EU",
        "summarization": "Sum",
        "entity recognition": "ER",
        "reasoning": "Rea",
        "temporal grounding": "TG",
    }

    SYS = (
        "These are the frames of a video. "
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
    )

    def __init__(self, dataset='LVBench', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['LVBench']

    # ── prepare_dataset ──────────────────────────────────────────────
    def prepare_dataset(self, dataset_name='LVBench', repo_id=META_REPO):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False
            video_dir = osp.join(pth, 'videos')
            if not osp.isdir(video_dir):
                return False
            n_videos = len(glob(osp.join(video_dir, '*.mp4')))
            if n_videos < 50:
                return False
            return True

        # 1. Resolve cache path
        cache_path = get_cache_path(repo_id)
        if cache_path is None:
            cache_path = osp.expanduser('~/.cache/huggingface/hub/datasets--zai-org--LVBench')
            os.makedirs(cache_path, exist_ok=True)

        if check_integrity(cache_path):
            dataset_path = cache_path
        else:
            # 2. Download metadata from zai-org/LVBench
            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            meta_file = osp.join(dataset_path, 'video_info.meta.jsonl')
            if not osp.exists(meta_file):
                raise FileNotFoundError(f'video_info.meta.jsonl not found in {dataset_path}')

            # 3. Download & extract videos (split zip from AIWinter/LVBench)
            video_dir = osp.join(dataset_path, 'videos')
            if not osp.isdir(video_dir) or len(glob(osp.join(video_dir, '*.mp4'))) < 50:
                self._download_and_extract_videos(dataset_path)

            # 4. Generate TSV
            self._generate_tsv(meta_file, dataset_path, dataset_name)
            cache_path = dataset_path

        # 5. Filter to only questions whose videos actually exist
        data_file = self._filter_by_existing_videos(cache_path, dataset_name)
        return dict(data_file=data_file, root=cache_path)

    # ── Download split zips from AIWinter/LVBench & extract ──────────
    def _download_and_extract_videos(self, dataset_path):
        video_dir = osp.join(dataset_path, 'videos')
        os.makedirs(video_dir, exist_ok=True)

        print('\n' + '=' * 70)
        print('📥  Downloading LVBench video archives from HuggingFace …')
        print('    This is ~66 GB total and may take a while.')
        print('=' * 70 + '\n')

        # ── download each split part ──
        zip_parts = []
        for i in range(1, NUM_ZIP_PARTS + 1):
            fname = f'all_videos_split.zip.{i:03d}'
            local = osp.join(dataset_path, fname)

            if osp.exists(local):
                print(f'  ✓ {fname} already present')
            else:
                print(f'  ↓ Downloading {fname} …')
                if modelscope_flag_set():
                    raise NotImplementedError('ModelScope download not supported for AIWinter/LVBench')
                hf_hub_download(
                    repo_id=VIDEO_REPO,
                    repo_type='dataset',
                    filename=fname,
                    local_dir=dataset_path,
                )
            zip_parts.append(local)

        # ── concatenate into a single zip ──
        merged_zip = osp.join(dataset_path, 'all_videos_merged.zip')
        if not osp.exists(merged_zip):
            print('\n🔗  Concatenating split archives …')
            with open(merged_zip, 'wb') as wf:
                for part in sorted(zip_parts):
                    print(f'     merging {osp.basename(part)}')
                    with open(part, 'rb') as rf:
                        shutil.copyfileobj(rf, wf, length=1024 * 1024 * 64)
            print('     Done.')

        # ── extract videos ──
        print('\n📦  Extracting videos …')
        with zipfile.ZipFile(merged_zip, 'r') as zf:
            video_members = [
                m for m in zf.infolist()
                if not m.is_dir() and m.filename.lower().endswith('.mp4')
            ]
            print(f'     Found {len(video_members)} video files in archive')
            for member in video_members:
                # Flatten: extract all mp4s directly into videos/
                out_name = osp.basename(member.filename)
                out_path = osp.join(video_dir, out_name)
                if osp.exists(out_path):
                    continue
                with zf.open(member) as src, open(out_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

        n = len(glob(osp.join(video_dir, '*.mp4')))
        print(f'\n✅  Extracted {n} videos into {video_dir}')

    # ── Generate TSV from metadata ───────────────────────────────────
    @staticmethod
    def _generate_tsv(meta_file, output_dir, dataset_name='LVBench'):
        tsv_path = osp.join(output_dir, f'{dataset_name}.tsv')
        if osp.exists(tsv_path):
            return tsv_path

        import jsonlines
        print(f'📝  Generating {dataset_name}.tsv from metadata …')

        with jsonlines.open(meta_file) as reader:
            video_meta = list(reader)

        rows = []
        idx = 0
        for meta in video_meta:
            video_id = meta['key']
            for qa in meta.get('qa', []):
                # question_type can be a list like ["event understanding", "entity recognition"]
                q_type = qa.get('question_type', ['unknown'])
                if isinstance(q_type, list):
                    q_type = q_type[0] if q_type else 'unknown'

                rows.append({
                    'index': idx,
                    'video': video_id,
                    'video_path': f'videos/{video_id}.mp4',
                    # The question already contains the MCQ options inline,
                    # e.g. "What year ...?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633"
                    'question': qa['question'],
                    # Answer is already a letter like "A", "B", "C", "D"
                    'answer': qa['answer'],
                    'uid': qa['uid'],
                    'category': q_type,
                })
                idx += 1

        df = pd.DataFrame(rows)
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f'     Wrote {len(rows)} questions to {tsv_path}')
        return tsv_path

    # ── Filter TSV by videos that actually exist on disk ─────────────
    @staticmethod
    def _filter_by_existing_videos(dataset_path, dataset_name='LVBench'):
        tsv_path = osp.join(dataset_path, f'{dataset_name}.tsv')
        df = pd.read_csv(tsv_path, sep='\t')

        video_dir = osp.join(dataset_path, 'videos')
        mask = df['video'].astype(str).apply(
            lambda vid: osp.exists(osp.join(video_dir, f'{vid}.mp4'))
        )

        kept = df[mask].copy()
        out_path = osp.join(dataset_path, f'{dataset_name}.tsv')
        kept.to_csv(out_path, sep='\t', index=False)
        print(f'✅  LVBench: {len(kept)}/{len(df)} questions have available videos')
        return out_path

    # ── Frame extraction (override base to handle video_path) ────────
    def save_video_frames(self, video_path, video_llm=False):
        """
        Follows the same pattern as LongVideoBench:
        video_path is a relative path like 'videos/<id>.mp4'.
        """
        vid_path = osp.join(self.data_root, video_path)
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }

        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_path[:-4])  # strip .mp4
        elif self.fps > 0:
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_path[:-4], len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth) and not video_llm:
                            im.save(pth)

        return frame_paths, indices, video_info

    # ── build_prompt ─────────────────────────────────────────────────
    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video_path'], video_llm)

        message = [dict(type='text', value=self.SYS)]

        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, line['video_path'])))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        # The question already contains the MCQ options inline like:
        #   "What year appears ...?\n(A) 1636\n(B) 1366\n(C) 1363\n(D) 1633"
        # So we just append it directly with the answer instruction.
        prompt = line['question'] + "\nAnswer with the option's letter from the given choices directly."
        message.append(dict(type='text', value=prompt))
        return message

    # ── evaluate ─────────────────────────────────────────────────────
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv', 'pkl'], \
            'Eval file should be xlsx/json/tsv/pkl'

        score_file = get_intermediate_file_path(eval_file, '_score', 'json')

        if not osp.exists(score_file):
            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                row = data.loc[data['index'] == idx].iloc[0]
                gt = str(row['answer']).strip().upper()
                pred = str(row.get('prediction', ''))
                extracted = cls._extract_answer(pred)
                data.loc[data['index'] == idx, 'score'] = int(extracted == gt)

            # Per-category accuracy
            results = {}
            for cat_full, cat_short in cls.CATEGORIES.items():
                subset = data[data['category'] == cat_full]
                if len(subset) > 0:
                    valid = subset[subset['score'] >= 0]
                    results[cat_short] = valid['score'].mean() if len(valid) > 0 else 0.0

            # Overall
            valid_all = data[data['score'] >= 0]
            results['Overall'] = valid_all['score'].mean() if len(valid_all) > 0 else 0.0

            rejected = len(data) - len(data_un)
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {rejected} questions. '
                f'Those will be counted as incorrect.'
            )

            dump(data, get_intermediate_file_path(eval_file, '_scored'))
            dump(results, score_file)

        results = load(score_file)

        # Print results
        print('\n' + '=' * 50)
        print('📊 LVBench Results:')
        print('=' * 50)
        for k, v in results.items():
            print(f'  {k:15s}: {v * 100:5.2f}%')
        print('=' * 50 + '\n')

        return results

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _extract_answer(pred_text):
        """Extract A/B/C/D from model prediction."""
        if pred_text is None or (isinstance(pred_text, float) and pd.isna(pred_text)):
            return ''
        s = str(pred_text).strip().upper()

        # Direct single letter
        if s in ('A', 'B', 'C', 'D'):
            return s

        # Letter at start with punctuation
        m = re.match(r'^([ABCD])[\s\.\)\]:,;]', s)
        if m:
            return m.group(1)

        # "OPTION A" style
        m = re.search(r'OPTION\s*([ABCD])', s)
        if m:
            return m.group(1)

        # Parenthesized letter
        m = re.search(r'\(([ABCD])\)', s)
        if m:
            return m.group(1)

        # Fallback: first A/B/C/D found
        m = re.search(r'\b([ABCD])\b', s)
        if m:
            return m.group(1)

        return ''