import json
import os
import urllib.request

from PIL import Image

from mini_vlm_toolkit.client.client import ToolClient
from mini_vlm_toolkit.client.parser import parse_tool_execute_args
from mini_vlm_toolkit.config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from mini_vlm_toolkit.llm.engine import LLMEngine
from mini_vlm_toolkit.tools.sam3_segment.tool import SAM3_Segment_Tool
from mini_vlm_toolkit.tools.sam3_visualize.tool import (
    SAM3_VisualizeAllMasks_Tool,
    SAM3_VisualizeSingleMask_Tool,
)
from tests.utils import (
    build_test_id,
    decode_base64_image,
    ensure_sam3_server,
    ensure_tool_server,
    save_test_output,
)


SAM3_SERVER_URL = os.getenv("SAM3_SERVER_URL", "http://127.0.0.1:8083")
TOOL_SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8101")
DEFAULT_TEST_IMAGE_URL = (
    "https://raw.githubusercontent.com/facebookresearch/sam3/main/assets/images/test_image.jpg"
)
PROMPT_CANDIDATES = [
    "person",
    "people",
    "man",
    "woman",
    "shoe",
    "sneaker",
    "vest",
    "blue vest",
    "red vest",
]


def log_step(title):
    print(f"\n== {title} ==")


def log_output(label, payload):
    print(f"{label}:\n{json.dumps(payload, indent=2, sort_keys=True)}")


def ensure_sam3_demo_image():
    image_path = os.getenv("SAM3_TEST_IMAGE")
    if image_path:
        return image_path

    test_dir = os.path.dirname(__file__)
    data_dir = os.path.join(test_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    image_path = os.path.join(data_dir, "sam3_demo.jpg")
    if os.path.isfile(image_path):
        return image_path

    image_url = os.getenv("SAM3_TEST_IMAGE_URL", DEFAULT_TEST_IMAGE_URL)
    log_step("Downloading SAM3 test image")
    print(f"Fetching {image_url}")
    try:
        urllib.request.urlretrieve(image_url, image_path)
    except Exception as exc:
        print(f"Failed to download SAM3 test image: {exc}")
        return None
    return image_path


def select_prompt_with_multiple_masks(result_lookup):
    for prompt in PROMPT_CANDIDATES:
        result = result_lookup(prompt)
        if "error" in result:
            raise AssertionError(f"SAM3 server error for prompt '{prompt}': {result['error']}")
        num_masks = len(result.get("masks_png_b64") or [])
        if num_masks >= 2:
            return prompt, result
    fallback_prompt = PROMPT_CANDIDATES[0]
    fallback_result = result_lookup(fallback_prompt)
    if "error" in fallback_result:
        raise AssertionError(
            f"SAM3 server error for prompt '{fallback_prompt}': {fallback_result['error']}"
        )
    return fallback_prompt, fallback_result


def build_tool_call_prompt(tool_name, tool_metadata, user_request, image_path):
    return (
        "Generate a single Python code block that calls this tool.\n"
        "The code must include exactly one line: execution = tool.execute(...).\n"
        "Use keyword arguments only.\n"
        f"Tool name: {tool_name}\n"
        f"Tool metadata: {tool_metadata}\n"
        f"Image path to use: {image_path}\n"
        f"User request: {user_request}\n"
        "Return only the code block."
    )


def test_direct_invocation():
    log_step("SAM3 visualize direct invocation (real image)")
    image_path = ensure_sam3_demo_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping SAM3 visualize direct test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    segment_tool = SAM3_Segment_Tool()

    def run_segment(prompt):
        return segment_tool.execute(
            prompt=prompt,
            image_path=image_path,
            return_masks_png_b64=True,
            max_masks=20,
            server_url=SAM3_SERVER_URL,
        )

    prompt, result = select_prompt_with_multiple_masks(run_segment)
    log_output("Segment prompt", {"prompt": prompt})
    log_output("SAM3 result summary", {"num_masks": len(result.get("masks_png_b64") or [])})

    tool_all = SAM3_VisualizeAllMasks_Tool()
    result_all = tool_all.execute(
        image_path=image_path,
        predictions=result,
        predictions_format="sam3_server",
    )
    all_image = decode_base64_image(result_all["image_base64"])
    save_test_output(all_image, build_test_id(__file__, "test_direct_invocation"), suffix="all")

    tool_single = SAM3_VisualizeSingleMask_Tool()
    result_single = tool_single.execute(
        image_path=image_path,
        predictions=result,
        predictions_format="sam3_server",
        mask_index=0,
    )
    overlay = decode_base64_image(result_single["overlay_image_base64"])
    zoomed = decode_base64_image(result_single["zoomed_image_base64"])
    save_test_output(overlay, build_test_id(__file__, "test_direct_invocation"), suffix="single")
    save_test_output(zoomed, build_test_id(__file__, "test_direct_invocation"), suffix="zoomed")


def test_client_invocation():
    log_step("SAM3 visualize client invocation via server (real image)")
    image_path = ensure_sam3_demo_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping SAM3 visualize client test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    client = ToolClient(TOOL_SERVER_URL)

    def run_segment(prompt):
        response = client.execute(
            "SAM3_Segment_Tool",
            {
                "prompt": prompt,
                "image_path": image_path,
                "return_masks_png_b64": True,
                "max_masks": 20,
                "server_url": SAM3_SERVER_URL,
            },
        )
        return response["result"]

    prompt, sam3_result = select_prompt_with_multiple_masks(run_segment)
    log_output("Segment prompt", {"prompt": prompt})
    log_output("SAM3 result summary", {"num_masks": len(sam3_result.get("masks_png_b64") or [])})

    response_all = client.execute(
        "SAM3_VisualizeAllMasks_Tool",
        {
            "image_path": image_path,
            "predictions": sam3_result,
            "predictions_format": "sam3_server",
        },
    )
    all_image = decode_base64_image(response_all["result"]["image_base64"])
    save_test_output(all_image, build_test_id(__file__, "test_client_invocation"), suffix="all")

    response_single = client.execute(
        "SAM3_VisualizeSingleMask_Tool",
        {
            "image_path": image_path,
            "predictions": sam3_result,
            "predictions_format": "sam3_server",
            "mask_index": 0,
        },
    )
    overlay = decode_base64_image(response_single["result"]["overlay_image_base64"])
    zoomed = decode_base64_image(response_single["result"]["zoomed_image_base64"])
    save_test_output(overlay, build_test_id(__file__, "test_client_invocation"), suffix="single")
    save_test_output(zoomed, build_test_id(__file__, "test_client_invocation"), suffix="zoomed")


def test_llm_invocation():
    log_step("SAM3 visualize LLM-assisted invocation via server (real image)")
    image_path = ensure_sam3_demo_image()
    if not image_path:
        print("SAM3 test image unavailable; skipping SAM3 visualize LLM test.")
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"SAM3_TEST_IMAGE not found: {image_path}")

    model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    api_key = os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY)
    log_output(
        "LLM config",
        {
            "model": model,
            "base_url": base_url,
            "api_key_set": bool(api_key),
        },
    )
    engine = LLMEngine(model=model, base_url=base_url, api_key=api_key)

    client = ToolClient(TOOL_SERVER_URL)

    def run_segment(prompt):
        response = client.execute(
            "SAM3_Segment_Tool",
            {
                "prompt": prompt,
                "image_path": image_path,
                "return_masks_png_b64": True,
                "max_masks": 20,
                "server_url": SAM3_SERVER_URL,
            },
        )
        return response["result"]

    prompt_used, sam3_result = select_prompt_with_multiple_masks(run_segment)
    log_output("Segment prompt", {"prompt": prompt_used})

    tools = client.list_tools()
    metadata = tools["SAM3_VisualizeAllMasks_Tool"]
    log_output("SAM3_VisualizeAllMasks_Tool metadata", metadata)

    prompt = build_tool_call_prompt(
        "SAM3_VisualizeAllMasks_Tool",
        metadata,
        "Overlay all masks on the image with default settings.",
        image_path,
    )
    llm_output = engine.generate(prompt, max_tokens=8192, temperature=0.4)
    log_output("LLM output", llm_output)
    try:
        args = parse_tool_execute_args(llm_output)
    except Exception as exc:
        log_output("LLM parse error", {"error": str(exc)})
        args = {}
    args.setdefault("image_path", image_path)
    args["predictions"] = sam3_result
    args["predictions_format"] = "sam3_server"

    response = client.execute("SAM3_VisualizeAllMasks_Tool", args)
    log_output("Response", response)
    all_image = decode_base64_image(response["result"]["image_base64"])
    save_test_output(all_image, build_test_id(__file__, "test_llm_invocation"))


def main():
    required_tools = [
        "SAM3_Segment_Tool",
        "SAM3_VisualizeAllMasks_Tool",
        "SAM3_VisualizeSingleMask_Tool",
    ]
    if not ensure_tool_server(TOOL_SERVER_URL, required_tools=required_tools):
        return
    if not ensure_sam3_server(SAM3_SERVER_URL):
        return
    test_direct_invocation()
    test_client_invocation()
    test_llm_invocation()
    print("SAM3 visualize tool test completed.")


if __name__ == "__main__":
    main()
