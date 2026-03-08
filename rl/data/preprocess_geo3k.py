"""Preprocess Geometry3K into verl-compatible parquet files."""

from __future__ import annotations

import argparse
import json
from typing import Any

import datasets

SYSTEM_PROMPT = (
    "You are a math expert. You are given a question and you need to solve it step by step. "
    "Reasoning step by step before any tool call. "
    "You should use the `calc_geo3k_reward` tool after step by step solving the question, "
    "before generate final answer at least once and refine your answer if necessary. "
)

INSTRUCTION_SUFFIX = (
    " You FIRST think about the reasoning process as an internal monologue and then provide "
    "the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)


def _build_record(example: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    problem = str(example.get("problem", ""))
    answer = str(example.get("answer", ""))
    images = example.get("images", [])
    prompt = f"{problem}{INSTRUCTION_SUFFIX}"

    return {
        "data_source": "hiyouga/geometry3k",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "images": images,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": problem,
            "answer": answer,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "calc_geo3k_reward": {
                    "create_kwargs": {"ground_truth": answer},
                },
            },
        },
    }


def _map_split(split_ds: datasets.Dataset, split: str) -> datasets.Dataset:
    def process_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        return _build_record(example, idx=idx, split=split)

    return split_ds.map(process_fn, with_indices=True, num_proc=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Geometry3K for GRPO training.")
    parser.add_argument("--output-train", required=True, help="Output train parquet path")
    parser.add_argument("--output-test", required=True, help="Output test parquet path")
    parser.add_argument(
        "--local-dataset-path",
        default=None,
        help="Optional local dataset path for datasets.load_dataset(...)",
    )
    parser.add_argument("--train-max-samples", type=int, default=-1, help="Optional cap for train split")
    parser.add_argument("--test-max-samples", type=int, default=-1, help="Optional cap for test split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_source = args.local_dataset_path or "hiyouga/geometry3k"
    dataset = datasets.load_dataset(dataset_source)
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    if args.train_max_samples > 0:
        train_ds = train_ds.select(range(min(args.train_max_samples, len(train_ds))))
    if args.test_max_samples > 0:
        test_ds = test_ds.select(range(min(args.test_max_samples, len(test_ds))))

    train_out = _map_split(train_ds, split="train")
    test_out = _map_split(test_ds, split="test")

    train_out.to_parquet(args.output_train)
    test_out.to_parquet(args.output_test)

    print(
        json.dumps(
            {
                "dataset_source": dataset_source,
                "train_samples": len(train_out),
                "test_samples": len(test_out),
                "output_train": args.output_train,
                "output_test": args.output_test,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
