#!/usr/bin/env python3
import os
import json
import argparse
from collections import Counter


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_tool_calls(traj_json):
    """
    Extract tool names from trajectory JSON.
    Expected format:
      data["messages"][i]["role"] == "assistant"
      data["messages"][i]["tool_calls"] = [...]
    """
    tool_names = []

    messages = traj_json.get("messages", [])
    if not isinstance(messages, list):
        return tool_names

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            continue

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue

            fn = tc.get("function", {})
            if not isinstance(fn, dict):
                continue

            name = fn.get("name")
            if name:
                tool_names.append(name)

    return tool_names


def get_json_files(folder):
    files = []
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            files.append(os.path.join(folder, fname))

    # sort like 0.json, 1.json, 2.json, ..., 10.json
    def sort_key(path):
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    files.sort(key=sort_key)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="/data/<user>/mini-vlm-toolkit/outputs/Qwen3-VL-8B-Instruct/baseagentwithtools/agent_v1_0215_judge_v1_0215_20260328_000824_default/RealWorldQA/full_trajectories", help="trajectory folder path")
    parser.add_argument(
        "--tool_name",
        type=str,
        nargs="+",
        default=["Crop_Tool", "Visual_Grounding_Tool", "Python_Executor_Tool", "ZoomInImageByBBox_Tool", "OCR_Tool", "SAM3_Segment_Tool", "Depth_Estimation_Tool", "Python_Coding_Agent_Tool"],
        help="tool name(s) to count, e.g. Visual_Grounding_Tool OCR_Tool",
    )
    args = parser.parse_args()

    folder = args.folder
    tool_names = args.tool_name  # None means all tools

    if not os.path.isdir(folder):
        raise ValueError(f"{folder} is not a valid directory")

    json_files = get_json_files(folder)

    total_counter = Counter()
    per_file_result = {}
    bad_files = []

    for path in json_files:
        try:
            data = load_json(path)
            tools = extract_tool_calls(data)

            if tool_names is None:
                count = Counter(tools)
                per_file_result[path] = count
                total_counter.update(count)
            else:
                count = Counter(t for t in tools if t in tool_names)
                per_file_result[path] = count
                total_counter.update(count)

        except Exception as e:
            bad_files.append((path, str(e)))

    print(f"Scanned {len(json_files)} files from: {folder}")
    print("=" * 80)

    if tool_names is None:
        print("Total tool call counts:")
        for name, cnt in total_counter.most_common():
            print(f"{name}: {cnt}")

        print("\nPer-file breakdown:")
        for path in json_files:
            base = os.path.basename(path)
            cnts = per_file_result.get(path, Counter())
            if len(cnts) == 0:
                print(f"{base}: 0")
            else:
                summary = ", ".join(f"{k}={v}" for k, v in cnts.items())
                print(f"{base}: {summary}")
    else:
        for tn in tool_names:
            print(f"Total calls for tool [{tn}]: {total_counter[tn]}")

    if bad_files:
        print("\nFailed files:")
        for path, err in bad_files:
            print(f"{path}: {err}")


if __name__ == "__main__":
    main()