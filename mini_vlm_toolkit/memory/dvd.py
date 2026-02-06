from nano_vectordb import NanoVectorDB
import os
import json
import multiprocessing
from tqdm import tqdm
import numpy as np

def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def preprocess_captions(caption_json_path):
    with open(caption_json_path, "r") as f:
        captions = json.load(f)
    scripts = []
    captions.pop('subject_registry', None)
    captions.pop('character_registry', None)
    for idx, (timestamp, cap_info) in enumerate(captions.items()):
        if cap_info.get('caption') is None or len(cap_info['caption']) == 0:
            print(f"Empty caption information for {timestamp} in {caption_json_path}")
            continue
        elif isinstance(cap_info['caption'], list):
            cap_info['caption'] = cap_info['caption'][0]
        elif not isinstance(cap_info['caption'], str):
            print(f"Invalid caption type for {cap_info['caption']}")
            cap_info['caption'] = str(cap_info['caption'])
        timestamp = list(map(float, timestamp.split("_")))
        scripts.append((timestamp, cap_info['caption'], cap_info))

    # batchify
    batch_size = 128
    batched_scripts = []
    print(f"Embedding {len(scripts)} captions...")
    for i in range(0, len(scripts), batch_size):
        batch = scripts[i:i+batch_size]
        batched_scripts.append(batch)
    cap2emb_list = []
    with multiprocessing.Pool(os.cpu_count() // 2) as pool:
        with tqdm(total=len(scripts), desc="Embedding captions...") as pbar:
            for result in pool.imap_unordered(
                single_batch_embedding_task,
                batched_scripts,
            ):
                cap2emb_list.extend(result)
                pbar.update(len(result))
    return cap2emb_list

# tbd
def single_batch_embedding_task(data):
    timestamps, captions, cap_infos = map(list, (zip(*data)))
    embs = VLLMEmbeddingService.get_embeddings(
        port=config.QWEN_EMBEDDING_MODEL_PORT,
        input_text=captions,
        is_query=False,
    )
    max_tries = 3
    while embs is None or len(embs) != len(captions):
        max_tries -= 1
        if max_tries < 0:
            raise ValueError(f"Failed to get embeddings for {timestamps} {captions}")
        print(f"Error in embedding {timestamps} {captions} retrying...")
        embs = VLLMEmbeddingService.get_embeddings(
            port=config.QWEN_EMBEDDING_MODEL_PORT,
            input_text=captions,
            is_query=False,
        )
    return list(zip(timestamps, cap_infos, [d['embedding'] for d in embs]))


class DVDMemory:
    """DVD memory wrapper that manages a NanoVectorDB and provides basic helpers."""

    def __init__(self, video_caption_path, video_db_path, embed_dim):
        self.database = self.init_single_video_db(
            video_caption_json_path=video_caption_path,
            output_video_db_path=video_db_path,
            emb_dim=embed_dim,
        )
        

    def attach_database(self, database: NanoVectorDB):
        if not isinstance(database, NanoVectorDB):
            raise TypeError("database must be a NanoVectorDB instance")
        self.database = database
        return self

    def init_single_video_db(self, video_caption_json_path, output_video_db_path, emb_dim):
        vdb = NanoVectorDB(emb_dim, storage_file=output_video_db_path)
        if os.path.exists(output_video_db_path):
            print(f"Database {output_video_db_path} already exists.")
        else:
            cap2emb_list = preprocess_captions(video_caption_json_path)
            data = []
            for idx, (timestamp, cap, emb) in enumerate(cap2emb_list):
                t1 = convert_seconds_to_hhmmss(timestamp[0])
                t2 = convert_seconds_to_hhmmss(timestamp[1])
                prefix = f"[From {t1} to {t2} seconds]\n"
                data.append(
                    {
                        "__vector__": np.array(emb),
                        "time_start_secs": timestamp[0],
                        "time_end_secs": timestamp[1],
                        "caption": prefix + cap['caption'],
                    }
                )
            _ = vdb.upsert(data)
            with open(video_caption_json_path, "r") as f:
                captions = json.load(f)
            subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))          
            video_length = max([float(k.split("_")[1]) for k in captions.keys()])
            video_length = convert_seconds_to_hhmmss(video_length)
            addtional_data = {
                'subject_registry': subject_registry,
                'video_length': video_length,
                'video_file_root': os.path.dirname(os.path.dirname(video_caption_json_path)),
                'fps': int(os.environ.get("VIDEO_FPS", 2)),
            }
            vdb.store_additional_data(**addtional_data)
            vdb.save()
        return vdb


    def query(self, embedding, top_k=16):
        if self.database is None:
            raise ValueError("database is not initialized")
        return self.database.query(embedding, top_k=top_k)
