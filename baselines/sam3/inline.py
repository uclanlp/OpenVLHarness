"""In-process SAM3 inference for the baseline runner.

Loads the SAM3 image model once at construction time and exposes a
`segment(prompt, image_path)` call that returns the same dict shape the
SAM3 FastAPI server returns (see
`mini_vlm_toolkit/tool_infra/src/tools/sam3_segment/server.py:174`). This
lets the baseline bypass the unified tool server entirely.

Two knobs are set once per process at construction time:

- `use_presence`: include the presence-token sigmoid multiply (canonical
  SAM3 grounding). When False (default), mirrors the server's
  `_forward_grounding_no_presence` patch
  (`tools/sam3_segment/server.py:34`), which in turn matches
  `use_presence=False` in `sam3/eval/postprocessors.py:45`.
- `iom_nms_thresh`: post-hoc IoM mask-overlap removal threshold (server
  uses 0.3). Pass `None` to skip NMS entirely.
"""

import types
from typing import Any, Dict, List, Optional

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image

from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
from sam3.model import box_ops
from sam3.model.data_misc import interpolate
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


@torch.inference_mode()
def _forward_grounding_no_presence(self, state):
    """Copy of `Sam3Processor._forward_grounding` minus the presence multiply."""
    outputs = self.model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=self.find_stage,
        geometric_prompt=state["geometric_prompt"],
        find_target=None,
    )
    out_bbox = outputs["pred_boxes"]
    out_masks = outputs["pred_masks"]
    out_probs = outputs["pred_logits"].sigmoid().squeeze(-1)

    keep = out_probs > self.confidence_threshold
    out_probs = out_probs[keep]
    out_masks = out_masks[keep]
    out_bbox = out_bbox[keep]

    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    img_h, img_w = state["original_height"], state["original_width"]
    scale = torch.tensor([img_w, img_h, img_w, img_h], device=self.device)
    boxes = boxes * scale[None, :]

    out_masks = interpolate(
        out_masks.unsqueeze(1),
        (img_h, img_w),
        mode="bilinear",
        align_corners=False,
    ).sigmoid()

    state["masks_logits"] = out_masks
    state["masks"] = out_masks > 0.5
    state["boxes"] = boxes
    state["scores"] = out_probs
    return state


def _mask_to_rle(mask: np.ndarray) -> str:
    m = mask
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    m = (m > 0).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(m))
    return rle["counts"].decode("utf-8")


def _sort_left_to_right(boxes, masks, scores):
    if not boxes:
        return boxes, masks, scores
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    return (
        [boxes[i] for i in order],
        [masks[i] for i in order] if masks else [],
        [scores[i] for i in order] if scores else [],
    )


class Sam3Inline:
    def __init__(
        self,
        *,
        use_presence: bool = False,
        iom_nms_thresh: Optional[float] = 0.3,
    ):
        self.use_presence = use_presence
        self.iom_nms_thresh = iom_nms_thresh
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        if not use_presence:
            self.processor._forward_grounding = types.MethodType(
                _forward_grounding_no_presence, self.processor
            )

    def segment(self, prompt: str, image_path: str) -> Dict[str, Any]:
        with Image.open(image_path) as raw:
            pil_img = raw.convert("RGB")
        orig_img_w, orig_img_h = pil_img.size

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(pil_img)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        boxes_list: List[Any] = []
        scores_list: List[float] = []
        if boxes is not None:
            boxes_list = (
                boxes.detach().cpu().tolist()
                if isinstance(boxes, torch.Tensor)
                else np.array(boxes).tolist()
            )
        if scores is not None:
            scores_list = (
                scores.detach().cpu().tolist()
                if isinstance(scores, torch.Tensor)
                else np.array(scores).tolist()
            )

        rle_masks: List[str] = []
        if masks is not None:
            masks_np = (
                masks.detach().cpu().numpy()
                if isinstance(masks, torch.Tensor)
                else np.array(masks)
            )
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                rle_masks = [_mask_to_rle(masks_np[i]) for i in range(masks_np.shape[0])]
            elif masks_np.ndim == 3:
                rle_masks = [_mask_to_rle(masks_np[i]) for i in range(masks_np.shape[0])]
            else:
                rle_masks = [_mask_to_rle(masks_np)]

        num_masks = len(rle_masks)

        if self.iom_nms_thresh is not None and num_masks >= 2:
            filtered = remove_overlapping_masks(
                {
                    "orig_img_h": orig_img_h,
                    "orig_img_w": orig_img_w,
                    "pred_masks": rle_masks,
                    "pred_boxes": boxes_list,
                    "pred_scores": scores_list,
                },
                iom_thresh=self.iom_nms_thresh,
            )
            rle_masks = filtered["pred_masks"]
            boxes_list = filtered["pred_boxes"]
            scores_list = filtered["pred_scores"]
            num_masks = len(rle_masks)

        boxes_list, rle_masks, scores_list = _sort_left_to_right(
            boxes_list, rle_masks, scores_list
        )

        return {
            "pred_boxes": boxes_list,
            "pred_scores": scores_list,
            "pred_masks": rle_masks,
            "orig_img_w": orig_img_w,
            "orig_img_h": orig_img_h,
            "num_masks": num_masks,
        }
