"""Preprocess FSC147 counting dataset into verl-compatible parquet files."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd

SYSTEM_PROMPT = (
    "You are a visual counting expert. You are given an image and a question asking you to count "
    "objects. Think carefully and reason step by step. "
    "You have access to SAM3_Segment_Tool, which can help segment and identify objects — "
    "use it when it would help you count more accurately."
)

INSTRUCTION_SUFFIX = (
    " Think step by step. "
    "The final answer MUST BE a single integer enclosed in \\boxed{}."
)


def _load_image_bytes(image_dir: str, image_path: str) -> bytes:
    full_path = os.path.join(image_dir, image_path)
    with open(full_path, "rb") as f:
        return f.read()


def _build_record(
    row: pd.Series,
    image_dir: str,
    split: str,
) -> dict[str, Any]:
    index = int(row["index"])
    image_path = str(row["image_path"])
    question = str(row["question"])
    answer = str(row["answer"])

    prompt_text = f"{question}{INSTRUCTION_SUFFIX}"

    image_bytes = _load_image_bytes(image_dir, image_path)

    return {
        "data_source": "FSC147",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>\n{prompt_text}"},
        ],
        "images": [{"bytes": image_bytes}],
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "question": question,
            "answer": answer,
            "need_tools_kwargs": True,
            # Empty struct not supported by Parquet; use a placeholder that the
            # ToolAgentLoop ignores (it only reads entries for known tool names).
            "tools_kwargs": {"__placeholder__": {"create_kwargs": {"note": "no per-tool kwargs"}}},
        },
    }


def _df_to_parquet(records: list[dict[str, Any]], output_path: str) -> None:
    import datasets as hf_datasets

    ds = hf_datasets.Dataset.from_list(records)
    ds.to_parquet(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess FSC147 for GRPO training.")
    parser.add_argument(
        "--lmu-data-dir",
        default="/data/<user>/lmu_data",
        help="Root directory containing FSC147_{train,val}.tsv and images/FSC147/",
    )
    parser.add_argument("--output-train", required=True, help="Output train parquet path")
    parser.add_argument("--output-val", required=True, help="Output val parquet path")
    parser.add_argument(
        "--train-max-samples", type=int, default=-1, help="Cap for train split (-1 = all)"
    )
    parser.add_argument(
        "--val-max-samples", type=int, default=50, help="Number of val samples (default 50)"
    )
    parser.add_argument(
        "--val-seed", type=int, default=None, help="Random seed for val subset sampling"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_dir = os.path.join(args.lmu_data_dir, "images", "FSC147")
    train_tsv = os.path.join(args.lmu_data_dir, "FSC147_train.tsv")
    val_tsv = os.path.join(args.lmu_data_dir, "FSC147_val.tsv")

    train_df = pd.read_csv(train_tsv, sep="\t")
    val_df = pd.read_csv(val_tsv, sep="\t")

    if args.train_max_samples > 0:
        train_df = train_df.iloc[: args.train_max_samples]

    if args.val_seed is not None:
        val_df = val_df.sample(
            n=min(args.val_max_samples, len(val_df)), random_state=args.val_seed
        ).reset_index(drop=True)
    else:
        val_df = val_df.iloc[: args.val_max_samples]

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    train_records = []
    for _, row in train_df.iterrows():
        train_records.append(_build_record(row, image_dir=image_dir, split="train"))

    val_records = []
    for _, row in val_df.iterrows():
        val_records.append(_build_record(row, image_dir=image_dir, split="val"))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_train)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_val)), exist_ok=True)

    _df_to_parquet(train_records, args.output_train)
    _df_to_parquet(val_records, args.output_val)

    print(
        json.dumps(
            {
                "train_samples": len(train_records),
                "val_samples": len(val_records),
                "output_train": args.output_train,
                "output_val": args.output_val,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
