from huggingface_hub import hf_hub_download, snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
import warnings
import os
import os.path as osp
import re
import json
import pandas as pd
import subprocess  # ← ADD
import sys  # ← ADD

FAIL_MSG = 'Failed to obtain answer via API.'

import glob

def _count_mp4_recursive(video_dir: str) -> int:
    return len(glob.glob(osp.join(video_dir, "**/*.mp4"), recursive=True))

def _count_id_symlinks(video_dir: str) -> int:
    # symlinks or real files directly under videos/ like GcRKREorGSc.mp4
    return len(glob.glob(osp.join(video_dir, "*.mp4")))

class LVBench(VideoBaseDataset):
    
    TYPE = 'Video-MCQ'
    DATASET_URL = 'zai-org/LVBench'
    DATASET_NAME = 'LVBench'
    
    CATEGORIES = {
        "key information retrieval": "KIR",
        "event understanding": "EU", 
        "summarization": "Sum",
        "entity recognition": "ER",
        "reasoning": "Rea",
        "temporal grounding": "TG"
    }
    
    FRAMES_TMPL = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""
    
    def __init__(self, dataset='LVBench', nframe=8, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset
    
    @classmethod
    def supported_datasets(cls):
        return ['LVBench']

    def _filter_tsv_by_existing_videos(self, dataset_path: str, dataset_name: str = "LVBench"):
        """Filter TSV to only include questions for videos that exist"""
        tsv_path = osp.join(dataset_path, f"{dataset_name}.tsv")
        df = pd.read_csv(tsv_path, sep="\t")

        video_dir = osp.join(dataset_path, "videos")
        exists = df["video"].astype(str).apply(
            lambda vid: osp.exists(osp.join(video_dir, f"{vid}.mp4"))
        )

        kept = df[exists].copy()
        out_path = osp.join(dataset_path, f"{dataset_name}.available.tsv")
        kept.to_csv(out_path, sep="\t", index=False)

        print(f"✅ LVBench: kept {len(kept)}/{len(df)} questions with available videos")
        return out_path
    
    def prepare_dataset(self, dataset_name='LVBench', repo_id='zai-org/LVBench'):
        """
        Prepare LVBench dataset.
        Downloads video_info.meta.jsonl and videos automatically.
        """
        
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not os.path.exists(data_file):
                return False
            
            video_dir = osp.join(pth, 'videos')
            if not os.path.exists(video_dir):
                return False
            
            # Check for ID-named videos (what we actually use)
            video_count = _count_id_symlinks(video_dir)
            if video_count < 50:
                return False
            
            return True
        
        # Try to find in cache
        cache_path = get_cache_path(repo_id)
        
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            # Download metadata from HuggingFace
            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            
            meta_file = osp.join(dataset_path, 'video_info.meta.jsonl')
            
            if not os.path.exists(meta_file):
                raise FileNotFoundError(f"video_info.meta.jsonl not found in {dataset_path}")
            
            # Convert to TSV
            self._convert_to_tsv(meta_file, dataset_path)
            
            # Download videos automatically
            self._download_videos(dataset_path)
        
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        
        # Filter to only available videos
        available_tsv = self._filter_tsv_by_existing_videos(dataset_path, dataset_name)
        
        return dict(data_file=available_tsv, root=dataset_path)  # ← FIX: root is dataset_path

    def _ensure_id_links(self, dataset_path: str):
        """
        video2dataset saves as videos/00000/00000000.mp4 ...
        We create symlinks: videos/<youtube_id>.mp4 -> that numeric file
        """
        import jsonlines

        video_dir = osp.join(dataset_path, "videos")
        meta_file = osp.join(dataset_path, "video_info.meta.jsonl")

        if not osp.exists(meta_file):
            raise FileNotFoundError(f"meta not found: {meta_file}")

        with jsonlines.open(meta_file) as r:
            video_meta = list(r)

        made = 0
        for idx, m in enumerate(video_meta):
            ytid = m["key"]
            link_path = osp.join(video_dir, f"{ytid}.mp4")
            if osp.exists(link_path):
                continue

            shard = idx // 10000
            candidate = osp.join(video_dir, f"{shard:05d}", f"{idx:08d}.mp4")

            target = None
            if osp.exists(candidate):
                target = candidate
            else:
                # fallback search
                hits = glob.glob(osp.join(video_dir, "**", f"{idx:08d}.mp4"), recursive=True)
                if hits:
                    target = hits[0]

            if target is None:
                continue

            # symlink relative for portability
            try:
                rel = osp.relpath(target, start=video_dir)
                os.symlink(rel, link_path)
                made += 1
            except (FileExistsError, OSError):
                pass

        if made > 0:
            print(f"🔗 Created {made} video ID symlinks")

    def _download_videos(self, dataset_path):
        """Download videos using video2dataset"""
        
        video_dir = osp.join(dataset_path, 'videos')
        
        # Check if already downloaded
        if osp.isdir(video_dir):
            n_recursive = _count_mp4_recursive(video_dir)
            n_flat = _count_id_symlinks(video_dir)

            if n_flat > 50:
                print(f"✅ Videos already downloaded ({n_flat} videos)")
                return

            if n_recursive > 50 and n_flat <= 50:
                print(f"✅ Videos downloaded ({n_recursive} found). Creating ID symlinks...")
                self._ensure_id_links(dataset_path)
                return
        
        print("\n" + "="*70)
        print("📥 Downloading LVBench videos from YouTube...")
        print("This may take 30-60 minutes depending on your connection.")
        print("="*70 + "\n")
        
        # Install video2dataset if needed
        try:
            import video2dataset
        except ImportError:
            print("Installing video2dataset...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "video2dataset", "-q"
            ])
            subprocess.call([
                sys.executable, "-m", "pip", "uninstall", "-y", "transformer-engine"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Create videos.txt
        meta_file = osp.join(dataset_path, 'video_info.meta.jsonl')
        videos_txt = osp.join(dataset_path, 'videos.txt')
        
        import jsonlines
        with jsonlines.open(meta_file) as reader:
            video_meta = list(reader)
        
        with open(videos_txt, 'w') as f:
            for meta in video_meta:
                f.write(f"https://www.youtube.com/watch?v={meta['key']}\n")
        
        total_videos = len(video_meta)
        print(f"Starting download of {total_videos} videos...")
        
        # Create directories
        tmp_dir = osp.join(dataset_path, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        
        # Run video2dataset
        cmd = [
            "video2dataset",
            f"--url_list={videos_txt}",
            f"--output_folder={video_dir}",
            "--input_format=txt",
            "--output_format=files",
            f"--tmp_dir={tmp_dir}"
        ]
        
        try:
            from tqdm import tqdm
            
            pbar = tqdm(total=total_videos, desc="Downloading", unit="video")
            
            process = subprocess.Popen(
                cmd,
                cwd=dataset_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            last_count = 0
            for line in process.stdout:
                tqdm.write(line.rstrip())
                
                try:
                    current_count = _count_mp4_recursive(video_dir)
                    if current_count > last_count:
                        pbar.update(current_count - last_count)
                        last_count = current_count
                except:
                    pass
            
            process.wait()
            
            # Create symlinks
            self._ensure_id_links(dataset_path)
            
            final_count = _count_id_symlinks(video_dir)
            pbar.update(final_count - last_count)
            pbar.close()
            
            # Clean up
            import shutil
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            
            print(f"\n✅ Downloaded {final_count}/{total_videos} videos successfully")
            
            if final_count < total_videos * 0.5:
                print("⚠️  Many videos unavailable on YouTube (normal for older datasets)")
            
        except Exception as e:
            print(f"⚠️  Download error: {e}")
            print("Continuing with any downloaded videos...")

    def _convert_to_tsv(self, meta_file, output_dir):
        """Convert video_info.meta.jsonl to TSV format"""
        import jsonlines
        
        print(f"Converting {meta_file} to TSV...")
        
        with jsonlines.open(meta_file) as reader:
            video_meta = list(reader)
        
        rows = []
        index = 0
        
        for meta_data in video_meta:
            video_id = meta_data['key']
            video_path = f'videos/{video_id}.mp4'
            
            for qa in meta_data.get('qa', []):
                q_type = qa.get('question_type', ['unknown'])
                if isinstance(q_type, list):
                    q_type = q_type[0] if q_type else 'unknown'
                
                rows.append({
                    'index': index,
                    'video': video_id,
                    'video_path': video_path,
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'uid': qa['uid'],
                    'category': q_type,
                    'time_reference': qa.get('time_reference', '')
                })
                index += 1
        
        df = pd.DataFrame(rows)
        tsv_path = osp.join(output_dir, f'{self.DATASET_NAME}.tsv')
        df.to_csv(tsv_path, sep='\t', index=False)
        print(f"Converted {len(rows)} questions to TSV")

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_id = str(line["video"]).strip()
        
        # Check if video exists (like LongVideoBench does)
        video_path = osp.join(self.data_root, "videos", f"{video_id}.mp4")  # ← FIX
        if not osp.exists(video_path):
            return None  # Skip missing videos
    
        message = []
        
        if video_llm:
            message.append(dict(type="video", value=video_path))
        else:
            # Parent class expects video ID for frame extraction
            frames = self.save_video_frames(video_id)
            for im in frames:
                message.append(dict(type="image", value=im))

        message.append(dict(type="text", value=self.FRAMES_TMPL))
        message.append(dict(type="text", value=f"{line['question']}\nAnswer: "))
        
        return message

    @staticmethod
    def compute_accuracy(answer_file: str, video_meta_file: str):
        """Compute accuracy using official LVBench metric"""
        import jsonlines
        from collections import defaultdict

        total_qa_num = 0
        right_num = 0
        category_right = defaultdict(int)
        category_total = defaultdict(int)

        with open(answer_file) as f:
            model_answers = json.load(f)

        with jsonlines.open(video_meta_file) as reader:
            for meta_data in reader:
                for qa in meta_data["qa"]:
                    uid = str(qa["uid"])
                    if uid not in model_answers:
                        continue

                    model_answer = str(model_answers[uid]).strip().upper()
                    gt = str(qa["answer"]).strip().upper()

                    for category in qa.get("question_type", []):
                        category_total[category] += 1
                        if model_answer == gt:
                            category_right[category] += 1

                    total_qa_num += 1
                    if model_answer == gt:
                        right_num += 1

        category_acc = {}
        for k in category_total:
            category_acc[k] = category_right[k] / max(1, category_total[k])

        acc = right_num / max(1, total_qa_num)
        category_acc["acc"] = acc
        return category_acc

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluate predictions using official LVBench metric"""
        
        # Handle different file formats
        if eval_file.endswith(".xlsx") and (not osp.exists(eval_file)):
            pkl_file = eval_file.replace(".xlsx", ".pkl")
            if osp.exists(pkl_file):
                eval_file = pkl_file
        
        data = load(eval_file)

        # Extract predictions
        predictions = {}
        for _, row in data.iterrows():
            uid = str(row.get("uid", "")).strip()
            if not uid:
                continue

            pred = row.get("prediction", "")
            letter = cls._extract_answer(pred)
            if letter:
                predictions[uid] = letter

        # Save in LVBench format
        pred_json = eval_file.replace(".tsv", "_lvbench.json").replace(".xlsx", "_lvbench.json")
        with open(pred_json, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"Saved predictions: {pred_json}")

        # Get metadata file
        cache_path = get_cache_path(cls.DATASET_URL)
        if cache_path is None:
            cache_path = snapshot_download(repo_id=cls.DATASET_URL, repo_type="dataset")

        meta_file = osp.join(cache_path, "video_info.meta.jsonl")
        if not osp.exists(meta_file):
            raise FileNotFoundError(f"Metadata not found: {meta_file}")

        results = cls.compute_accuracy(pred_json, meta_file)

        # Format results
        name_map = {
            "key information retrieval": "KIR",
            "event understanding": "EU",
            "summarization": "Sum",
            "entity recognition": "ER",
            "reasoning": "Rea",
            "temporal grounding": "TG",
            "acc": "Overall",
        }
        formatted = {name_map.get(k, k): v for k, v in results.items()}
        
        # Save results to JSON (like VideoMME/LongVideoBench)
        score_file = eval_file.replace('.tsv', '_score.json').replace('.xlsx', '_score.json')
        dump(formatted, score_file)
        print(f"Saved scores: {score_file}")
        
        # Print results
        print("\n" + "="*50)
        print("📊 LVBench Results:")
        print("="*50)
        for k, v in formatted.items():
            print(f"  {k:15s}: {v*100:5.2f}%")
        print("="*50 + "\n")
        
        return formatted
        
    @staticmethod
    def _extract_answer(pred_text: str) -> str:
        """Extract A/B/C/D from prediction"""
        if pred_text is None or (isinstance(pred_text, float) and pd.isna(pred_text)):
            return ""

        s = str(pred_text).strip().upper()

        # Match single letter
        m = re.search(r"(?:^|[\s\(\[\{])([ABCD])(?:$|[\s\)\]\}\.,:;])", s)
        if m:
            return m.group(1)

        # Match "OPTION A" etc
        m = re.search(r"OPTION\s*([ABCD])", s)
        if m:
            return m.group(1)

        return ""