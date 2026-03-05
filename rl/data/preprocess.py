"""Preprocess benchmark samples into verl RL parquet format."""

from __future__ import annotations

import argparse
import json
import os
import string
from pathlib import Path
from typing import Any

import pandas as pd

from rl.prompts.system_prompts import build_hermes_system_prompt


def _is_na(value: Any) -> bool:
    try:
        return pd.isna(value)
    except Exception:
        return False


def _canonical_gt(value: Any) -> str:
    if value is None or _is_na(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if len(text) == 1 and text.upper() in string.ascii_uppercase:
        return text.upper()
    return text


def _extract_ground_truth(row: pd.Series) -> str:
    for key in ["answer", "label", "gt", "target", "correct_answer"]:
        if key in row and not _is_na(row[key]):
            return _canonical_gt(row[key])
    return ""


def _extract_question(row: pd.Series) -> str:
    for key in ["question", "query", "prompt"]:
        if key in row and not _is_na(row[key]):
            return str(row[key])
    return ""


def _extract_options(row: pd.Series) -> dict[str, str]:
    options = {}
    for letter in string.ascii_uppercase:
        if letter in row and not _is_na(row[letter]):
            options[letter] = str(row[letter])
    return options


def _struct_to_user_content_and_images(struct: Any) -> tuple[str, list[Any]]:
    if not isinstance(struct, list):
        return str(struct), []

    chunks = []
    images = []
    for item in struct:
        if not isinstance(item, dict):
            chunks.append(str(item))
            continue
        item_type = item.get("type")
        value = item.get("value")
        if item_type == "image":
            chunks.append("<image>")
            if value is not None:
                images.append(value)
        elif item_type == "text":
            if value is not None:
                chunks.append(str(value))
        else:
            chunks.append(str(value))

    content = "\n".join(part for part in chunks if part is not None and str(part).strip() != "")
    return content, images


def build_row(dataset_name: str, row: pd.Series, system_prompt: str, idx: int) -> dict[str, Any]:
    # Re-use dataset prompt builder so task formatting matches benchmark evaluation.
    struct = row.get("_prompt_struct")
    user_content, images = _struct_to_user_content_and_images(struct)

    question = _extract_question(row)
    answer = _extract_ground_truth(row)
    options = _extract_options(row)

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return {
        "data_source": dataset_name,
        "agent_name": "tool_agent",
        "prompt": prompt,
        "images": images,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": int(row.get("index", idx)),
            "question": question,
            "answer": answer,
            "options": options,
            # Keep a non-empty mapping so pyarrow can infer a stable struct schema.
            # ToolAgentLoop only reads entries for known tool names, so this placeholder is ignored.
            "need_tools_kwargs": False,
            "tools_kwargs": {
                "__placeholder__": {
                    "create_kwargs": {"note": "no per-tool kwargs"},
                }
            },
        },
    }


_DEFAULT_VAL_FRAC = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert benchmark dataset to verl RL parquet format.")
    parser.add_argument("--dataset", required=True, help="Dataset name accepted by build_dataset()")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit number of samples")
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="all",
        help="Which split to emit. 'all' uses the full dataset (legacy behaviour).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=_DEFAULT_VAL_FRAC,
        help=f"Fraction of data reserved for val (default {_DEFAULT_VAL_FRAC})",
    )
    return parser.parse_args()


def _load_project_env() -> None:
    """Load `<project_root>/.env` so LMUData is available in standalone scripts."""
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        # Keep backward compatibility if python-dotenv is unavailable.
        return
    load_dotenv(dotenv_path=env_file, override=False)


def main() -> None:
    args = parse_args()
    _load_project_env()
    if not os.environ.get("LMUData"):
        raise EnvironmentError(
            "LMUData is not set. Define LMUData in your shell or integrate-verl/.env."
        )

    # Import after LMUData validation: dataset module initialization may access LMUData at import-time.
    from mini_vlm_toolkit.dataset import build_dataset

    dataset = build_dataset(args.dataset)
    if dataset is None:
        raise RuntimeError(f"Failed to build dataset: {args.dataset}")

    df = dataset.data.copy()

    # Deterministic train/val split when requested.
    if args.split in ("train", "val"):
        shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        n_val = max(1, int(len(shuffled) * args.val_frac))
        if args.split == "val":
            df = shuffled.iloc[:n_val].copy()
        else:
            df = shuffled.iloc[n_val:].copy()

    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    prompt_structs = []
    for _, row in df.iterrows():
        prompt_structs.append(dataset.build_prompt(row))
    df["_prompt_struct"] = prompt_structs

    system_prompt = build_hermes_system_prompt(tool_schemas=[])

    records = []
    for idx, (_, row) in enumerate(df.iterrows()):
        records.append(build_row(args.dataset, row, system_prompt, idx))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_parquet(args.output, index=False)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "samples": len(out_df),
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
