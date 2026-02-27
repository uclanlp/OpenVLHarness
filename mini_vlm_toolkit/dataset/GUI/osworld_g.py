import ast
import json
import os
import os.path as osp
import re
from collections import defaultdict

import pandas as pd

from PIL import Image
from tqdm import tqdm

from .screenspot import ScreenSpot, USER_INSTRUCTION
from ...smp import *

logger = get_logger("RUN")

SYSTEM_PROMPT_OSWORLD_G = """You are a helpful assistant.

You must answer by returning exactly one XML tool call block:
<tool_call>
{"name": "computer_use", "arguments": {"action": "<click_or_wait>", "coordinate": [x, y]}}
</tool_call>

Rules:
- Use action "click" when the target is visible.
- Use action "wait" when the target is not visible or task is infeasible.
- For click, x and y must be numbers in [0, 1000], where (0,0) is top-left and (1000,1000) is bottom-right.
- Do not output any extra text outside <tool_call>...</tool_call>.
"""


def parse_osworldg_click_point(response):
    if not isinstance(response, str):
        return [0.0, 0.0]

    try:
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.S)
        if match:
            payload = json.loads(match.group(1))
            args = payload.get("arguments", {}) if isinstance(payload, dict) else {}
            action = str(args.get("action", "")).strip().lower()
            if action == "wait":
                return [-1.0, -1.0]
            coord = args.get("coordinate")
            if isinstance(coord, list) and len(coord) >= 2:
                return [float(coord[0]), float(coord[1])]
    except Exception:
        pass

    # Fallback: tolerate legacy "x=..., y=..." style output.
    match = re.search(r"x=([\d.]+), y=([\d.]+)", response)
    if match:
        return [float(match.group(1)), float(match.group(2))]

    return [0.0, 0.0]


def normalize_click_point(click_point, img_size):
    x, y = float(click_point[0]), float(click_point[1])
    if x < 0 or y < 0:
        return [x, y]
    if x <= 1.0 and y <= 1.0:
        return [x, y]
    if x <= 1000.0 and y <= 1000.0:
        return [x / 1000.0, y / 1000.0]
    w, h = img_size
    if w > 0 and h > 0:
        return [x / w, y / h]
    return [0.0, 0.0]


class OSWorldG(ScreenSpot):
    MODALITY = "IMAGE"
    TYPE = "GUI"
    DATASET_URL = {}
    DATASET_MD5 = {}
    EVAL_TYPE = "point"
    RE_TYPE = "functional"

    _SUPPORTED_DATASETS = ["OSWorld_G", "OSWorld-G"]

    def __init__(
        self,
        dataset="OSWorld_G",
        skip_noimg=True,
        skeleton=False,
        re_type="functional",
    ):
        self.dataset_name = dataset
        self.RE_TYPE = re_type
        lmu_root = LMUDataRoot()
        self.img_root = os.environ.get(
            "OSWORLD_G_IMAGE_ROOT",
            osp.join(lmu_root, "images", "OSWorld-G"),
        )
        self._ann_root = os.environ.get(
            "OSWORLD_G_ANN_ROOT",
            osp.join(lmu_root, "OSWorld-G"),
        )
        self._ann_file = os.environ.get("OSWORLD_G_ANN_FILE")
        if skeleton:
            return

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and "image_path" in data:
            data = data[~pd.isna(data["image_path"])]

        data["index"] = [str(x) for x in data["index"]]
        self.meta_only = True
        self.parse_response_func = parse_osworldg_click_point
        self.data = data

    @classmethod
    def supported_datasets(cls):
        return cls._SUPPORTED_DATASETS

    def _resolve_annotation_file(self):
        root = LMUDataRoot()
        ann_candidates = [
            self._ann_file,
            osp.join(self._ann_root, "OSWorld-G.json"),
            osp.join(self._ann_root, "OSWorld-G_refined.json"),
            osp.join(root, "OSWorld-G", "OSWorld-G.json"),
            osp.join(root, "OSWorld-G", "OSWorld-G_refined.json"),
        ]
        for p in ann_candidates:
            if p and osp.isfile(p):
                return osp.abspath(p)
        raise FileNotFoundError(
            "OSWorld-G annotation file not found. "
            "Set OSWORLD_G_ANN_FILE or place OSWorld-G.json under "
            f"{self._ann_root}."
        )

    def load_data(self, dataset):
        ann_path = self._resolve_annotation_file()
        raw = load(ann_path)
        if isinstance(raw, pd.DataFrame):
            records = raw.to_dict("records")
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError(f"Unsupported annotation format in {ann_path}: {type(raw)}")

        rows = []
        for i, item in enumerate(records):
            if not isinstance(item, dict):
                continue
            box_type = str(item.get("box_type", "bbox")).lower()
            bbox = item.get("box_coordinates", None)
            if box_type != "bbox" or not isinstance(bbox, list) or len(bbox) != 4:
                continue

            gui_types = item.get("GUI_types", [])
            if isinstance(gui_types, list) and len(gui_types):
                data_type = str(gui_types[0]).lower()
            else:
                data_type = "gui"

            rows.append(
                {
                    "index": item.get("id", i),
                    "question": item.get("instruction", ""),
                    "image_path": item.get("image_path", ""),
                    "bbox": bbox,
                    "data_type": data_type,
                    "data_source": "OSWorld-G",
                    "category": ",".join(gui_types) if isinstance(gui_types, list) else str(gui_types),
                }
            )

        if len(rows) == 0:
            raise ValueError(f"No valid bbox samples found in {ann_path}")
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        instruction = USER_INSTRUCTION.format(instruction=line["question"])
        msgs = [dict(role="system", type="text", value=SYSTEM_PROMPT_OSWORLD_G)]
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs.append(dict(type="image", value=tgt_path))
        msgs.append(dict(type="text", value=instruction))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        return self.evaluate_point(eval_file, **judge_kwargs)

    def evaluate_point(self, eval_file, **judge_kwargs):
        # -1: format_err, 0: wrong, 1: correct
        stats = defaultdict(list)
        data = load(eval_file)
        assert "bbox" in data and "prediction" in data
        lines = [data.iloc[i] for i in range(len(data))]
        for line in tqdm(lines):
            bbox = line["bbox"] if isinstance(line["bbox"], list) else ast.literal_eval(line["bbox"])
            # Ground-truth bbox is in xywh absolute pixels.
            x1, y1, w, h = bbox
            bbox = (x1, y1, x1 + w - 1, y1 + h - 1)
            image = Image.open(os.path.join(self.img_root, line["image_path"]))
            img_size = image.size
            bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
            key = line["data_type"] if "category" not in line else line["category"] + ":" + line["data_type"]
            prediction = str(line["prediction"])
            try:
                click_point = self.parse_response_func(prediction)
                click_point = normalize_click_point(click_point, img_size)
                if click_point[0] < 0 or click_point[1] < 0:
                    match = False
                    stats[key].append(0)
                else:
                    match = (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3])
                    stats[key].append(1 if match else 0)
                is_wrong_format = False
            except Exception as e:
                logger.warning(f"exception in osworld-g eval: {e}")
                stats[key].append(-1)
                match, is_wrong_format = False, True

        final_score_dict = {}
        final_score_dict.update({k + ":cnt": len(stats[k]) for k in stats})
        full_stats = [x for values in stats.values() for x in values]
        final_score_dict["Overall_Accuracy"] = np.mean([x > 0 for x in full_stats]) * 100
        final_score_dict["Format_Err_Rate"] = np.mean([x < 0 for x in full_stats]) * 100
        text_stats = [v for k, values in stats.items() if k.endswith("text") for v in values]
        icon_stats = [v for k, values in stats.items() if k.endswith("icon") for v in values]
        final_score_dict["Text_Accuracy"] = np.mean([x > 0 for x in text_stats]) * 100 if len(text_stats) else 0.0
        final_score_dict["Icon_Accuracy"] = np.mean([x > 0 for x in icon_stats]) * 100 if len(icon_stats) else 0.0
        score_pth = get_intermediate_file_path(eval_file, "_score", "json")
        dump(final_score_dict, score_pth)
        return final_score_dict
