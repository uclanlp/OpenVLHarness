import base64
import io
import types
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import pycocotools.mask as mask_utils
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model import box_ops
from sam3.model.data_misc import interpolate
from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks

from openvlharness.services.tools.sam3_segment.viz.visualizer import Visualizer

MODEL_NAME = "sam3-image"
IOM_THRESH = 0.3  # post-hoc IoM mask-overlap removal threshold

app = FastAPI(title="SAM3 Segmentation Server", version="0.1")

# Load once at startup (important for performance)
model = build_sam3_image_model()
processor = Sam3Processor(model)


# Disable presence token: replace _forward_grounding with a copy of
# sam3/model/sam3_image_processor.py:182-222 minus the presence multiply.
# Mirrors the canonical `use_presence=False` switch in sam3/eval/postprocessors.py:45.
@torch.inference_mode()
def _forward_grounding_no_presence(self, state):
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


processor._forward_grounding = types.MethodType(
    _forward_grounding_no_presence, processor
)


def _pil_from_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise ValueError("Empty image upload.")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def _mask_to_rle(mask: np.ndarray) -> str:
    """
    Encode a numpy mask directly to a COCO RLE counts string.
    Expected mask shape: (H, W) or (1, H, W).
    """
    m = mask
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    m = (m > 0).astype(np.uint8)
    m_fortran = np.asfortranarray(m)
    rle = mask_utils.encode(m_fortran)
    return rle["counts"].decode("utf-8")


def _sort_left_to_right(
    boxes: List[List[float]],
    masks: List[str],
    scores: List[float],
) -> Tuple[List[List[float]], List[str], List[float]]:
    """Sort boxes, RLE masks, and scores left-to-right by box x1 coordinate."""
    if not boxes:
        return boxes, masks, scores
    sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_masks = [masks[i] for i in sorted_indices] if masks else []
    sorted_scores = [scores[i] for i in sorted_indices] if scores else []
    return sorted_boxes, sorted_masks, sorted_scores


def _render_masks(
    pil_img: Image.Image,
    pred_boxes: List[List[float]],
    rle_masks: List[str],
    orig_h: int,
    orig_w: int,
) -> str:
    """
    Render masks onto the image and return a base64-encoded PNG string.

    Uses hardcoded viz params (already tuned):
      - font_size_multiplier=1.2, border_width_multiplier=0.0
      - alpha=0.15, label_mode="1"
    """
    img_rgb = np.array(pil_img)

    # Build RLE dicts and binary masks for the Visualizer
    rle_dicts = [
        {"size": (orig_h, orig_w), "counts": rle} for rle in rle_masks
    ]
    binary_masks = [mask_utils.decode(rle_dict) for rle_dict in rle_dicts]
    boxes_np = np.array(pred_boxes) if pred_boxes else np.empty((0, 4))

    viz = Visualizer(
        img_rgb,
        font_size_multiplier=1.2,
        border_width_multiplier=0.0,
    )
    viz.overlay_instances(
        boxes=boxes_np,
        masks=rle_dicts,
        binary_masks=binary_masks,
        assigned_colors=None,
        alpha=0.15,
        label_mode="1",
    )
    rendered_rgb = viz.output.get_image()
    rendered_pil = Image.fromarray(rendered_rgb)

    buf = io.BytesIO()
    rendered_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/segment")
def segment(
    prompt: str = Form(...),
    image: UploadFile = File(...),
) -> JSONResponse:
    """
    Multipart/form-data request:
      - prompt: text prompt
      - image: uploaded image file

    Returns JSON with pred_boxes, pred_masks (RLE), pred_scores,
    orig_img_h, orig_img_w, num_masks, and rendered_image_b64.
    All masks/boxes/scores are sorted left-to-right by box x1.
    """
    try:
        pil_img = _pil_from_upload(image)
        orig_img_w, orig_img_h = pil_img.size

        # need to enable this on sam3.1 update
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = processor.set_image(pil_img)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        # Convert to numpy / python-native types
        masks_np = None
        if masks is not None:
            try:
                if isinstance(masks, torch.Tensor):
                    masks_np = masks.detach().cpu().numpy()
                else:
                    masks_np = np.array(masks)
            except Exception:
                masks_np = np.array(masks)

        boxes_list: List[Any] = []
        scores_list: List[float] = []

        if boxes is not None:
            try:
                if isinstance(boxes, torch.Tensor):
                    boxes_list = boxes.detach().cpu().tolist()
                else:
                    boxes_list = np.array(boxes).tolist()
            except Exception:
                boxes_list = np.array(boxes).tolist()

        if scores is not None:
            try:
                if isinstance(scores, torch.Tensor):
                    scores_list = scores.detach().cpu().tolist()
                else:
                    scores_list = np.array(scores).tolist()
            except Exception:
                scores_list = np.array(scores).tolist()

        # Convert masks to RLE strings
        rle_masks: List[str] = []
        if masks_np is not None:
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                for i in range(masks_np.shape[0]):
                    rle_masks.append(_mask_to_rle(masks_np[i]))
            elif masks_np.ndim == 3:
                for i in range(masks_np.shape[0]):
                    rle_masks.append(_mask_to_rle(masks_np[i]))
            else:
                rle_masks.append(_mask_to_rle(masks_np))

        num_masks = len(rle_masks)

        # Post-hoc IoM mask-overlap removal (greedy keep by score)
        if num_masks >= 2:
            filtered = remove_overlapping_masks(
                {
                    "orig_img_h": orig_img_h,
                    "orig_img_w": orig_img_w,
                    "pred_masks": rle_masks,
                    "pred_boxes": boxes_list,
                    "pred_scores": scores_list,
                },
                iom_thresh=IOM_THRESH,
            )
            rle_masks = filtered["pred_masks"]
            boxes_list = filtered["pred_boxes"]
            scores_list = filtered["pred_scores"]
            num_masks = len(rle_masks)

        # Sort left-to-right by box x1
        boxes_list, rle_masks, scores_list = _sort_left_to_right(
            boxes_list, rle_masks, scores_list
        )

        # Render visualization
        rendered_image_b64 = ""
        if num_masks > 0:
            rendered_image_b64 = _render_masks(
                pil_img, boxes_list, rle_masks, orig_img_h, orig_img_w
            )

        resp: Dict[str, Any] = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "pred_boxes": boxes_list,
            "pred_scores": scores_list,
            "pred_masks": rle_masks,
            "orig_img_h": orig_img_h,
            "orig_img_w": orig_img_w,
            "num_masks": num_masks,
            "rendered_image_b64": rendered_image_b64,
        }

        return JSONResponse(resp)

    except Exception as e:
        return JSONResponse({"model": MODEL_NAME, "error": str(e)}, status_code=400)
